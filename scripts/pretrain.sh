export CUDA_VISIBLE_DEVICES=0,1,2,3
n_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
nproc_per_node=$n_devices
n_threads=$((n_devices * 2))

python -u -m torch.distributed.run --nproc_per_node=$nproc_per_node \
    --nnodes=1 \
    --master_port 12312 train_kpgt.py \
    --save_path ../models/ \
    --n_threads $n_threads \
    --n_devices $n_devices \
    --config KG-GCPT \
    --n_steps 200000 \
    --batch_size 512 \
    --gradient_accumulate_steps 2 \
    --pretrain1_path ../chembl29 \
    --data_aug1 subgraph \
    --data_aug1_rate 0.2 \
    --data_aug2 drop_nodes \
    --data_aug2_rate 0.2 \
    --save_name test
    # --wandb_key None