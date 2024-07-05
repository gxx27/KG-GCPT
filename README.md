# Pre-training on Drug Transformer

## Datasets
Datasets with extracted features are available at the following Google Drive links:

Pubchem: https://drive.google.com/file/d/14YJIlgHEu4Qrp1asYxgzZNCu3GZxi2l5/view?usp=drive_link

Chembl: https://drive.google.com/file/d/1Vo8X0MN_Ni7H1HJRR4NKrV1lzQN4A_fu/view?usp=drive_link

finetune datasets: https://drive.google.com/file/d/1jAU9SIkXOtEmtl-w45kjl6dClExhW5Fx/view?usp=drive_link

## Pretrained Model
Pretrained model is available with the Google drive link:https://drive.google.com/file/d/15emQKReJgt34HxpSCdDY2OEGhALUtjty/view?usp=drive_link

## KPGT
First, you need to download the dataset and pre-trained model in the path KPGT/

and then build the conda environment
```shell
conda env create
conda activate KPGT
```

For pre-training, run the following command
```shell
cd scripts
./pretrain.sh
```
to do the pretraining.

For fine-tuning, run the following command
```shell
cd scripts
./finetune_classification.sh # downstream task is classification

# or
./finetune_regression.sh # downstream task is regression
```
to do the fine-tuning and reproduce the results.