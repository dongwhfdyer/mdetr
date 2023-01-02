#CUDA_VISIBLE_DEVICES=0,1 LOGLEVEL=DEBUG python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --dataset_config configs/pretrain.json --ema
LOGLEVEL=DEBUG python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --dataset_config configs/pretrain.json --ema
#LOGLEVEL=DEBUG python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --dataset_config configs/pretrain.json --ema --batch_size 1
#CUDA_LAUNCH_BLOCKING=1 LOGLEVEL=DEBUG python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --dataset_config configs/pretrain.json --ema --batch_size 1
