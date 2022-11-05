# HyCondITM

Code for paper *[Hybrid Conditional Deep Inverse Tone Mapping](https://dl.acm.org/doi/abs/10.1145/3503161.3548129)*.


## Introduction

This project is developed based on [BasicSR](https://github.com/xinntao/BasicSR).

For matters not covered in this document, please refer to the detailed documentation of the original project.


### Structure of directories
```
|-- basicsr      # core source code
   |-- archs
      |-- condition_module.py       # conditioned feature modulation module
      |-- hyconditm_arch.py         # main network architecture
      |-- histogram_module.py       # histogram feature module（deprecated）
   |-- data
      |-- paired_image_dataset.py   # paired dataset class
      |-- unpaired_image_dataset.py # unpaired dataset class
   |-- losses
      |-- color_diff_loss.py        # color difference loss
      |-- histogram_loss.py         # histogram loss（deprecated）
   |-- metrics
   |-- models
      |-- itm_model.py              # definition of model
   |-- utils
      |-- hdr_util.py               # utils for representation conversion, and etc.
   |-- test.py
   |-- train.py
|-- datasets     # datasets
|-- options      # configs of experiments
   |-- test
      |-- HyCondITM
   |-- train
      |-- HyCondITM
      |-- HyCondITM_unpaired
```


## Dataset

HDRTV1K dataset \[[GitHub](https://github.com/chxy95/HDRTVNet)\]

Download link:

- [Baidu Netdisk](https://pan.baidu.com/s/1TwXnBzeV6TlD3zPvuEo8IQ) （access code: 6qvu）
- [OneDrive](https://uofmacau-my.sharepoint.com/:f:/g/personal/yc17494_umac_mo/Ep6XPVP9XX9HrZDUR9SmjdkB-t1NSAddMIoX3iJmGwqW-Q?e=dNODeW) （access code: HDRTVNet）


## How To Run

Train

```shell
CUDA_VISIBLE_DEVICES=2,4 basicsr/train.py -opt options/train/HyCondITM/train_HyCondITM_v1_L1only_stepLR.yml
```

Test

```shell
CUDA_VISIBLE_DEVICES=1 basicsr/test.py -opt options/test/HyCondITM/test_HyCondITMv1.yml
```


## Citation

> Tong Shao, Deming Zhai, Junjun Jiang, and Xianming Liu. 2022. Hybrid Conditional Deep Inverse Tone Mapping. In Proceedings of the 30th ACM International Conference on Multimedia (MM '22). Association for Computing Machinery, New York, NY, USA, 1016–1024. https://doi.org/10.1145/3503161.3548129