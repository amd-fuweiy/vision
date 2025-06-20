unset MIOPEN_FIND_MODE
unset MIOPEN_FIND_ENFORCE

COMPILE_AGRS="--compile"    #COMPILE_ARGS=""
FAKE_DATA_ARGS="--use-fake-data" # FAKE_DATA_ARGS=""
USE_NHWC_LAYOUT=false

if [ "$USE_NHWC_LAYOUT" == "true" ]; then
    export PYTORCH_MIOPEN_SUGGEST_NHWC=1
    export PYTORCH_MIOPEN_SUGGEST_NHWC_BATCHNORM=1
    LAYOUT_ARGS="--channel-last"
    echo "Training with NHWC layout"
else
    export PYTORCH_MIOPEN_SUGGEST_NHWC=0
    export PYTORCH_MIOPEN_SUGGEST_NHWC_BATCHNORM=0
    LAYOUT_ARGS=""
    echo "Training with NCHW layout"
fi

DATAPATH=/mnt/nvme0/data/datasets/imagenet
MODEL=efficientnet_b7
GBS=128
torchrun --nproc_per_node=8 references/classification/train.py --data-path $DATAPATH \
        --model $MODEL --epochs 300 --batch-size $GBS --opt adamw --lr 0.001 --weight-decay 0.05 --norm-weight-decay 0.0  --bias-weight-decay 0.0 --transformer-embedding-decay 0.0 --lr-scheduler cosineannealinglr --lr-min 0.00001 --lr-warmup-method linear  --lr-warmup-epochs 20 --lr-warmup-decay 0.01 --amp --label-smoothing 0.1 --mixup-alpha 0.8 --clip-grad-norm 5.0 --cutmix-alpha 1.0 --random-erase 0.25 --interpolation bicubic --auto-augment ta_wide --model-ema --ra-sampler --ra-reps 4  --val-resize-size 224 $LAYOUT_ARGS $COMPILE_ARGS $FAKE_DATA_ARGS --profile
