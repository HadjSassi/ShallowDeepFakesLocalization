# IEEE TSYP11 CS-YP Challenge
## IEEE Higher National Engineering School Of Tunis (ENSIT) SB

### This work is a fork for the work of Shallow- and Deep-fake Image Manipulation Localization Using Deep Learning

## Datasets

### First Dataset source

The first dataset to reproduce the same results of the work of Shallow- and Deep-fake Image Manipulation Localization Using Deep Learning from [here](https://www.dropbox.com/s/o5410tl5v4vxsth/ICNC2023-Deepfakes.tar.xz?dl=0).

### Second Dataset source

The second dataset need to be extended to the first dataset and updating the paths to reproduce the results of our work is from here [here](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images/data)

## Usage
We need NVIDIA Graphical card to launch this project, so replace {Number Of GPUs} with 
the number of GPUs cores you want to execute the project.

If you want to test the previous work just replace {x} with 1, if you want to test the 
newest work just replace {x] with 2.

### Training

Run the following code to train the network.

```
python -m torch.distributed.launch --nproc_per_node={Number of GPUs} train_torch.py --paths_file .\Paths{x}\paths_train_df.txt --val_paths_file .\Paths{x}\paths_val_df.txt --model ours
```

### Testing

Run the following code to evaluate the network. It's recommended to put the trained Model in the pretrainedModel and test it

```
python -m torch.distributed.launch --nproc_per_node={Number of GPUs}  evaluate.py --paths_file .\Paths{x}\paths_test_df.txt --load_path .\pretrainedModel\model.path --model ours
```