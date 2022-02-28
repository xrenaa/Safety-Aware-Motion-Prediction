## Safety-aware Motion Prediction with Unseen Vehicles for Autonomous Driving

<a href="https://arxiv.org/abs/2109.01510"><img src="https://img.shields.io/badge/arXiv-2102.10543-b31b1b.svg"></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>

> **Safety-aware Motion Prediction with Unseen Vehicles for Autonomous Driving** <br>
> *Xuanchi Ren, Tao Yang, Li Erran Li, Alexandre Alahi, and Qifeng Chen*<br>
> *ICCV 2021*<br>
> 
[[Paper](https://arxiv.org/pdf/2109.01510.pdf)]
[[Supplementary material]()]

## Recent Updates   
<!-- I am sorry that I am busying with application, and I am planning to release code ASAP. -->
<!-- :white_check_mark: Update StyleGAN2   -->
:white_check_mark: Update data preprocessing code  
:white_check_mark: Update model  
:black_square_button: Training script  

## Installation ##

### Cloning ###

1. Clone this repository with the following command:

```
git clone https://github.com/xrenaa/Safety-Aware-Motion-Prediction.git
cd experiments/nuScenes
git clone https://github.com/nutonomy/nuscenes-devkit.git
git checkout 12fb09169eb8ebf04bc39a30cd50334215769c3e
```

2. Replace `experiments/nuScenes/nuscenes-devkit/python-sdk/nuscenes/prediction/input_representation/static_layers.py` with the file [Here](https://drive.google.com/file/d/1Q_lqbj78Nv3Js9FNwWJvay77UaghfVNa/view?usp=sharing).

### Environment Setup ###
First, we'll create a conda environment to hold the dependencies.
```
conda create --name safeDrive python=3.6 -y
source activate safeDrive
pip install -r requirements.txt
```


### Data Setup ###
#### nuScenes Dataset ####
1. Download the nuScenes dataset (this requires signing up on [their website](https://www.nuscenes.org/)). Extract the downloaded zip file's contents and place them in the `experiments/nuScenes` directory. Then, download the map expansion pack (v1.2) and copy the contents of the extracted `maps` folder into the `experiments/nuScenes/maps` folder. Eventually you should have the following folder structure:

```
experiments/nuscenes
    samples	-	Sensor data for keyframes.
    sweeps	-	Sensor data for intermediate frames.
    maps	-	Folder for all map files: rasterized .png images and vectorized .json files.
    v1.0-trainval	-	JSON tables that include all the meta data and annotations.
    process_data.py - Our provided data processing script.
```

2. Finally, process them into a data format that our model can work with.

```
cd experiments/nuScenes

# For the tranval nuScenes dataset, use the following
python process_data.py --data ../nuScenes --split train --img_size 128
python process_data.py --data ../nuScenes --split train_val --img_size 128
python process_data.py --data ../nuScenes --split val --img_size 128
```
We provide a [notebook](experiments/nuScenes/data_playground.ipynb) to visualize the processed data.

## Citation
```bibtex
@inproceedings{ren2021unseen,
  title   = {Safety-aware Motion Prediction with Unseen Vehicles for Autonomous Driving},
  author  = {Ren, Xuanchi, and Yang, Tao, and Li, Li Erran, and Alahi, Alexandre, and Chen, Qifeng},
  booktitle = {ICCV},
  year    = {2021}
}
```
