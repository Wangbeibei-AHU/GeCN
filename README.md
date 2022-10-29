# GeCN Implementation
This is a TensorFlow implementation of the TPAMI 2021 paper "GeCNs: Graph Elastic Convolutional Networks for Data Representation" for the task of (semi-supervised) classification task.

## Requirements
- python>=3.6
- tensorflow-gpu>=1.13.1


## Examples
Here are the instruction commands for running the codes.
* Command
```
cd GeCN
python train.py --dataset cora 
```

## Introduction
In this repo, we provide GeCN's code with the Cora and Citeseer datasets as example. The graph convolution method used in this code is provided by Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017).


## Cite
Please cite our paper if you use this code in your own work:

```
@article{jiang2021gecns,
  title={Gecns: Graph elastic convolutional networks for data representation},
  author={Jiang, Bo and Wang, Beibei and Tang, Jin and Luo, Bin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022},
  volume={44},
  number={9},
  pages={4935-4947},
  publisher={IEEE}
}
```
