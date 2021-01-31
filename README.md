# SLATracker
This repository hosts our code for our paper ***Spatial-Attention Location-Aware Multi-Object Tracking***.  
The code will be released after the paper published.  
![](https://github.com/JunnHan/SLATracker/blob/main/assets/MOT17-03.gif)  
### Requirements
- Python3.6
- Pytorch 1.6.0, torchvision 0.7.0
- [detectron2](https://github.com/facebookresearch/detectron2)
- python-opencv
- py-motmetrics
- cython-bbox
### Installation
See [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)
### Quick Start
#### Dataset Zoo
See [DATASET_ZOO.md](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md)  
**Note that we transform all used datasets to COCO format for training convenience.**
#### Training
`python3 train_net.py --config-file config/faster_rcnn_R_50_FPN_1x.yaml`  
#### Visualization
A trained model is available at [GoogleDrive](https://drive.google.com/drive/folders/1bfaB9MRSyv_2AfmJYhu2WgiiMA2w8yje?usp=sharing)  
`python3 demo/vis_track.py --config-file config/ --opts MODEL.WEIGHTS output/model_final.pth`
### MOTChallenge Results
[Official MOTChallenge website](https://motchallenge.net/)
Benchmark | MOTA | IDF1 | MOTP | MT | ML | FP | FN | IDSw |
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
2DMOT15_public | 47.0 | 57.9 | 75.3 | 22.6 | 27.2 | 9044 | 22986 | 558 |
MOT16_public | 60.6 | 59.5 | 78.0 | 24.2 | 29.1 | 5783 | 65469 | 643 |
MOT17_public | 59.7 | 63.4 | 77.7 | 24.0 | 31.1 | 16644 | 209318 | 1647 |
2DMOT15_private | 57.4 | 62.3 | 75.9 | 38.6 | 15.5 | 5910 | 19716 | 560 |
MOT16_private | 72.0 | 69.6 | 77.9 | 37.3 | 20.9 | 7242 | 43147 | 740 |
MOT17_private | 71.8 | 69.0 | 77.8 | 38.0 | 20.5 | 19077 | 137700 | 2493 |
### Acknowledgement
A large part of the code is borrowed from [Zhongdao/Towards-Realtime-MOT](https://github.com/Zhongdao/Towards-Realtime-MOT) and [DeanChan/HOIM-PyTorch](https://github.com/DeanChan/HOIM-PyTorch). Thanks for their wonderful works.
