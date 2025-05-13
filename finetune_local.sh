torchrun --nnodes=1 \
    --nproc_per_node=4 \
    --master_port=9911 \
    lerobot/scripts/fsdp_train.py \
    --policy.type="qwen" \
    --save_freq=100 \
    --dataset.repo_id="whatever" \
    --dataset.processor="/Data/lzl/qwen2.5_vl_7b/Qwen2.5-VL-7B-Instruct" \
    --dataset.parent_dir="/Data/lerobot_data/simulated" \
    --dataset.data_mix="libero" \
    --output_dir="qwen_flow" \
    --batch_size=2 \
    --steps=60_0000 \
    --policy.scheduler_warmup_steps=2_0000 \
    --policy.scheduler_decay_steps=60_0000 \
    --policy.optimizer_lr=2.5e-5 \
    --policy.train_main_layers=0 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false
    