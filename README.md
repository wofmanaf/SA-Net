# SA-Net: Shuffle Attention for Deep Convolutional Neural Networks <sub>([paper](https://arxiv.org/pdf/2102.00240.pdf))</sub>
By Qing-Long Zhang and Yu-Bin Yang

[State Key Laboratory for Novel Software Technology at Nanjing University]

## Approach
<div align="center">
  <img src="https://github.com/wofmanaf/SA-Net/blob/main/figures/sa.png">
</div>
<p align="center">
  Figure 1: The Diagram of a shuffle attention module.
</p>

## Image Classification

We provide baseline sa_resnet models pretrained on ImageNet-1k.

| name | acc@1 | #params (M) | url |
| --- | --- | --- | --- |
| sa_resnet50 | 77.88 | 25.56 | [BaiduDrive(474p)](https://pan.baidu.com/s/1-TEXeMjprUDyx013l3vZIQ)|
| sa_resnet101 | 78.95 | 44.55 | [BaiduDrive(6nxm)](https://pan.baidu.com/s/16L0enJxKMd9LJ8B4dbGmtw)|


## Evaluation
To evaluate a pre-trained sa_resnet50 on ImageNet val with a single GPU run:
```
python main.py -a sa_resnet50 -e --resume /path/to/sa_resnet50.pth.tar /path/to/imagenet
```
This should give
```
 * Acc@1 77.882 Acc@5 93.892
```