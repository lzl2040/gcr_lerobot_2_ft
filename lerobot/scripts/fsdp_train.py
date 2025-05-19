#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import time
import os
import glob
import json
import functools
from pathlib import Path
from datetime import datetime
from pprint import pformat
from termcolor import colored
from typing import Any
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    always_wrap_policy
)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.api import StateDictType, FullStateDictConfig

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer, Qwen2_5_VLVisionBlock
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.transforms import ImageTransforms
from lerobot.common.datasets.lerobot_dataset import MultiDatasetforDistTraining, extra_collate_fn
from lerobot.common.datasets.sampler import EpisodeAwareSampler, DistEpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy

def init_logger(cfg, rank):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO if rank == 0 else logging.WARN)
    
    if rank == 0:
        formatter = logging.Formatter(
            f'[%(asctime)s] [rank: {rank}] [%(levelname)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        log_path = Path(cfg.log_dir) / f"fsdp_logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0

def save_fsdp_checkpoint(model, optim, output_dir, step):
    # 使用 StateDictType.FULL_STATE_DICT 替代 FSDP.FULL_STATE_DICT
    save_policy = StateDictType.FULL_STATE_DICT
    full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    # 所有进程统一进入状态字典收集阶段
    with FSDP.state_dict_type(model, save_policy, full_state_dict_config):
        model_state_dict = model.state_dict()
    
    # 所有进程同步，防止部分进程提前退出
    dist.barrier()

    # 仅主进程保存模型和优化器状态
    if get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        ckpt_path = os.path.join(output_dir, f"step{step}.pt")

        # 可选：保存优化器状态
        # optim_state_dict = FSDP.full_optim_state_dict(model, optim)

        # torch.save({
        #     'model': model_state_dict,
        #     'optimizer': optim_state_dict,
        #     'step': step,
        # }, ckpt_path)
        torch.save(model_state_dict, ckpt_path)

        logging.info(f"Checkpoint saved at {ckpt_path}")

def train_step(model, batch, scaler, optimizer):
    """执行单个训练步骤"""
    # 前向传播
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        loss, output_dict = model(batch)
    
    # 反向传播
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()
    
    # 梯度裁剪（可选）
    # scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # 参数更新
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad()
    
    # 梯度平均，用于记录
    if dist.is_initialized():
        dist.all_reduce(grad_norm, op=dist.ReduceOp.SUM)
        grad_norm /= dist.get_world_size()
    
    return loss, grad_norm, output_dict

@parser.wrap()
def train(cfg: TrainPipelineConfig):
    # 初始化分布式环境
    # os.environ["NODE_RANK"] = "0"
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_rank = int(os.environ["RANK"])
    node_rank = int(os.environ["NODE_RANK"])
    master_ip = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]
    master_uri = "tcp://%s:%s" % (master_ip, master_port)
    dist.init_process_group(
        backend="nccl",
        init_method=master_uri,
        world_size=world_size,
        timeout=timedelta(minutes=60),
        rank=world_rank,
    )
    # dist.init_process_group(backend="nccl")
    # world_size = dist.get_world_size()
    rank = dist.get_rank()
    # local_rank = rank
    # local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    # local_rank = node_rank
    torch.cuda.set_device(local_rank)
    
    # 初始化配置
    cfg.validate()
    logger = init_logger(cfg, rank)
    logger.info(f"DIST INFO: world_size={world_size}, local_rank={local_rank}, world_rank={world_rank}, node_rank={node_rank}, master_uri={master_uri}")
    
    step = 1
    seed = cfg.seed + rank
    if cfg.resume:
        logger.info("Resume is set, will try to load from checkpoint...")
        os.makedirs(cfg.output_dir, exist_ok=True)
        pts = sorted(glob.glob(os.path.join(cfg.output_dir, "*.pt")))
        logger.info(f"Found {len(pts)} checkpoints, names are {pts}")
        if pts:
            if cfg.resume_force_path is not None:
                if cfg.resume_force_path in pts:
                    step = int(os.path.basename(cfg.resume_force_path).split(".")[0].split("step")[1]) + 1
                    logger.info(f"Force resume path {cfg.resume_force_path} found in {pts}")
                else:
                    steps = [int(os.path.basename(pt).split(".")[0].split("step")[1]) for pt in pts]
                    step = sorted(steps)[-1] + 1
                    
                    logger.info(f"Force resume path {cfg.resume_force_path} not found in {pts}, will resume from the latest checkpoint from {cfg.output_dir} at step {step}")
                seed += (step-1)
            else:
                
                steps = [int(os.path.basename(pt).split(".")[0].split("step")[1]) for pt in pts]
                step = sorted(steps)[-1] + 1
                logger.info(f"Resuming from checkpoint {pts[-1]} at step {step}")
                seed += (step-1)
        else:
            cfg.resume = False
            logger.info("No checkpoint found, starting from scratch.")
            
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(pformat(cfg.to_dict()))
        if cfg.wandb.enable and cfg.wandb.project:
            wandb_logger = WandBLogger(cfg)
        else:
            wandb_logger = None
            logger.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))
    else:
        wandb_logger = None
    
    # 设置随机种子
    if cfg.seed is not None:
        set_seed(cfg.seed)
    
        
    # 数据集初始化
            
    image_transforms = (ImageTransforms(cfg.dataset.image_transforms))
    dataset = MultiDatasetforDistTraining(
        cfg=cfg, 
        image_transforms=image_transforms,
        seed=seed,
        data_mix=cfg.dataset.data_mix,
        vla2root_json="vla2root.json",
        # vla2root_json="vla2root_bak_single.json"
    )
    
    # Policy setup
    logger.info("Creating policy...")
    if hasattr(cfg.policy, "tokenizer_max_length"):
        logger.info("Setting model's tokenizer_max_length to 100")
        cfg.policy.tokenizer_max_length=100
    logger.info("Still creating policy...")
    
    # 模型初始化
    policy = make_policy(
        cfg=cfg.policy,
        device="cpu",
        ds_meta=dataset.meta,
        weight_pt_path="/mnt/wangxiaofa/original_qw/flow+04_0509_df100_full_Prometheus/step50000.pt"
        # weight_pt_path="/Data/lzl/qwen_pi0/step50000.pt"
    )
    
    # 训练状态初始化
    if cfg.resume:
        if pts:
            cfg.resume = os.path.join(cfg.output_dir, f"step{step-1}.pt")
            logger.info(f"Resuming from checkpoint {cfg.resume} at step {step}")
            model_state_dict = torch.load(cfg.resume, map_location="cpu")
            policy.load_state_dict(model_state_dict, strict=True)
            
    # 设置模型全部参数为BF16
    logger.info("Setting model parameters to BF16...")
    for params in policy.parameters():
        params.data = params.data.bfloat16()
        # params.data = params.data.to(dtype=torch.float16)
    
    # FSDP包装配置
    # auto_wrap_policy = functools.partial(
    #     transformer_auto_wrap_policy,
    #     transformer_layer_cls={
    #                     Qwen2DecoderLayer,
    #                     Qwen2_5_VLDecoderLayer,
    #                     # Qwen2RMSNorm
    #                     },  
    # )
    # auto_wrap_policy = functools.partial(
    #     size_based_auto_wrap_policy,
    #     min_num_params=10000000,
    #     exclude_wrap_modules={
    #         Qwen2RMSNorm
    #     }
    # )
    auto_wrap_policy = functools.partial(
        always_wrap_policy,
    )
    
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
        keep_low_precision_grads=True
    )
    # mixed_precision = None
    
    model = FSDP(
        policy,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=local_rank,
        use_orig_params=True
    )
    
    # 优化器和学习率调度器
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, model)
    
    logger.info(model)
    
    # 数据加载器
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=cfg.seed+rank,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=2,
        collate_fn=extra_collate_fn,
        pin_memory=False,
    )
    
    # 混合精度scaler
    # scaler = ShardedGradScaler()
    scaler = None
    
    # Metrics setup
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
        "optim_s": AverageMeter("optim_s", ":.3f"),
    }
    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=int(step)
    )
    
    # 主训练循环
    if rank == 0:
        logger.info(f"Starting FSDP training on {world_size} devices")
        logger.info(pformat(cfg.to_dict()))
    
    model.train()
    dataloader_iter = cycle(dataloader)
    
    fwd_bwd_time = 0
    dataloading_s = 0
    
    if cfg.resume:
        logger.info("Setting up learning rate scheduler...")
        for _ in range(step-1):
            lr_scheduler.step()
    
    if rank == 0:
        logger.info("Starting training loop...")
        
    
    while step < cfg.steps:
        batch_start = time.perf_counter()
        batch = next(dataloader_iter)
        data_time = time.perf_counter() - batch_start
        dataloading_s += data_time
        
        step_start = time.perf_counter()
        loss, grad_norm, outputs = train_step(model, batch, scaler, optimizer)
        step_time = time.perf_counter() - step_start
        fwd_bwd_time += step_time
        
        # 更新指标
        train_tracker.dataloading_s = dataloading_s
        train_tracker.update_s = fwd_bwd_time
        
        loss_value = loss.detach().mean().item()
        grad_norm_value = grad_norm.item() if grad_norm is not None else 0.0
        
        train_tracker.loss = loss_value
        train_tracker.grad_norm = grad_norm_value
        train_tracker.lr = optimizer.param_groups[0]["lr"]
        train_tracker.step()
        
        fwd_bwd_time = 0
        dataloading_s = 0
        
        
        # 学习率调度
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        # 日志记录
        if step % cfg.log_freq == 0:
            dist.barrier(device_ids=[local_rank])
            if rank == 0:
                logger.info(train_tracker)
                
                if wandb_logger:
                    wandb_log_dict = train_tracker.to_dict()
                    if outputs:
                        wandb_log_dict.update(outputs)
                    wandb_logger.log_dict(wandb_log_dict, step)
                train_tracker.reset_averages()
        
        # 保存检查点
        if step % cfg.save_freq == 0:
            save_fsdp_checkpoint(model, optimizer, cfg.output_dir, step)
        
        
        step += 1
    
    # 最终保存
    if rank == 0:
        save_fsdp_checkpoint(model, optimizer, cfg.output_dir, "final")
        logger.info("Training completed successfully")

if __name__ == "__main__":
    # # 设置环境变量
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # os.environ["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
    # os.environ["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"
    os.environ['WANDB_API_KEY'] = '9e1c3ac77856b8ebb5573c4e1e250c84aabfb904'
    
    # 启动训练
    train()