
# ========== 配置区域 ==========
# MODE: "resume" = 加载作者checkpoint继续训练(恢复优化器/epoch)
#       "finetune" = 用作者checkpoint作为初始化,从epoch 0重新训练
MODE="resume"

# 作者提供的训练好的checkpoint路径
PRETRAINED_CKPT=/group/40143/howu/AIDE/pretrained_ckpts/genimage_train.pth
# ==============================

GPU_NUM=8
WORLD_SIZE=1
RANK=0
MASTER_ADDR=localhost
MASTER_PORT=29512

DISTRIBUTED_ARGS="
    --nproc_per_node $GPU_NUM \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# 根据 MODE 设置参数
if [ "$MODE" = "resume" ]; then
    echo "[INFO] 模式: 继续训练 (resume) - 恢复优化器状态和epoch"
    MODE_ARGS="--resume ${PRETRAINED_CKPT}"
    OUTPUT_DIR=/group/40143/howu/AIDE/results/genimage_resume
elif [ "$MODE" = "finetune" ]; then
    echo "[INFO] 模式: 重新训练 (finetune) - 仅加载模型权重,从epoch 0开始"
    MODE_ARGS="--finetune ${PRETRAINED_CKPT}"
    OUTPUT_DIR=/group/40143/howu/AIDE/results/genimage_finetune
else
    echo "[ERROR] 未知 MODE: $MODE, 请设置为 resume 或 finetune"
    exit 1
fi

PY_ARGS=${@:1}  # Any other arguments

python -m torch.distributed.launch $DISTRIBUTED_ARGS main_finetune.py \
    --model AIDE \
    --batch_size 32 \
    --blr 1e-4 \
    --epochs 20 \
    --data_path /group/40143/howu/AIDE/dataset/GenImage/train \
    --eval_data_path /group/40143/howu/AIDE/dataset/GenImage/train/Midjourney \
    --resnet_path /group/40143/howu/AIDE/pretrained_ckpts/resnet50.pth \
    --convnext_path /group/40143/howu/AIDE/pretrained_ckpts/open_clip_pytorch_model.bin \
    --output_dir ${OUTPUT_DIR} \
    ${MODE_ARGS} \
    ${PY_ARGS}
