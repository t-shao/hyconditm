# HyCondITM 项目文档

## 方法介绍
见论文

## 数据集

HDRTV1K 数据集 \[[GitHub](https://github.com/chxy95/HDRTVNet)\]

下载地址： [百度网盘](https://pan.baidu.com/s/1TwXnBzeV6TlD3zPvuEo8IQ) （提取码：6qvu）、[OneDrive](https://uofmacau-my.sharepoint.com/:f:/g/personal/yc17494_umac_mo/Ep6XPVP9XX9HrZDUR9SmjdkB-t1NSAddMIoX3iJmGwqW-Q?e=dNODeW) （提取码：HDRTVNet）

## 源码说明

源码基于开源项目 BasicSR \[[GitHub](https://github.com/xinntao/BasicSR)\] 构建，本文档未尽事宜请查阅原项目的详细文档。

### 目录结构
```
|-- basicsr      # 核心代码
   |-- archs
      |-- condition_module.py       # 条件特征调制
      |-- hyconditm_arch.py         # 网络结构
      |-- histogram_module.py       # 直方图特征模块（depr）
   |-- data
      |-- paired_image_dataset.py   # 配对数据集
      |-- unpaired_image_dataset.py # 非配对数据集
   |-- losses
      |-- color_diff_loss.py        # 色差loss
      |-- histogram_loss.py         # 直方图loss（depr）
   |-- metrics
   |-- models
      |-- itm_model.py              # 模型定义
   |-- utils
      |-- hdr_util.py               # 传递函数、色域转换等
   |-- test.py
   |-- train.py
|-- datasets     # 数据集
|-- options      # 实验配置文件
   |-- test
      |-- HyCondITM
   |-- train
      |-- HyCondITM
      |-- HyCondITM_unpaired
```

### 运行命令

训练

```shell
CUDA_VISIBLE_DEVICES=2,4 basicsr/train.py -opt options/train/HyCondITM/train_HyCondITM_v1_L1only_stepLR.yml
```

测试

```shell
CUDA_VISIBLE_DEVICES=1 basicsr/test.py -opt options/test/HyCondITM/test_HyCondITMv1.yml
```

