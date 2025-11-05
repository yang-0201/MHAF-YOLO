# MHAF-YOLO
We have optimized and designed MAF-YOLOv2(MHAF-YOLO) based on the latest YOLO framework. This model achieves exceptionally high parameter efficiency and has reached state-of-the-art performance among all YOLO variants.

最新论文为：[MHAF-YOLO: Multi-Branch Heterogeneous Auxiliary Fusion YOLO for accurate object detection](https://arxiv.org/abs/2502.04656) 

我们改进的Dual-MHAF-YOLO框架在 [**阿里天池 “睿创杯”首届高校创新创业大赛赛道一：轻量化双光(可见光-红外)目标检测**](https://tianchi.aliyun.com/competition/entrance/532344?spm=5176.12281973.J_6-HJZaSjQocH7SIdvbK02.3.784b3b74HDwofS) 竞赛中取得冠军！<br><br>

<div align="center">
 <a href="./">
     <img src="https://github.com/user-attachments/assets/aced26b3-e4a1-4288-a68b-4c2b3b86c5e2" width="80%"/>
</a>
 
</div>

## UPDATES 🔥
- 2025/2/7: Add Paper
- 2025/1/22: Add MAF-YOLOv2-cls, MAF-YOLOv2-seg
- 2024/11/11: Add MAF-YOLOv2
<div align="center">
    <a href="./">
        <img src="https://github.com/user-attachments/assets/075bd591-4851-424d-a627-535a938e88e7" width="40%"/>
        <img src="https://github.com/user-attachments/assets/b3623f16-d99e-4803-b65d-cd7069b3dd62" width="40%"/>
    </a>
</div>



## Model Zoo
### Detection
MS COCO

| Model             | Test Size | #Params | FLOPs |     AP<sup>val</sup>     |   AP<sub>50</sub><sup>val</sup>    | Latency|
|:------------------|:----:|:-------:|:-----:|:------------------------:|:----------------------------------:|:----------------------------------:|
| [MAF-YOLOv2-Lite-N](https://github.com/yang-0201/MAF-YOLOv2/releases/download/v1.0.0/MAF-YOLOv2-Lite-N.pt) |   640  |  1.4M   | 4.7G  |          38.5%            |      53.7%                |        1.11ms 
| [MAF-YOLOv2-N](https://github.com/yang-0201/MAF-YOLOv2/releases/download/v1.0.0/MAF-YOLOv2-N.pt)      |   640  |  2.2M   | 7.2G  |          42.3%           |               58.5%   |1.28ms                   | 
| [MAF-YOLOv2-S](https://github.com/yang-0201/MAF-YOLOv2/releases/download/v1.0.0/MAF-YOLOv2-S.pt)      |   640  |  7.1M   | 25.3G |      48.9%      |               65.9%    |     1.67ms    |             | 
| [MAF-YOLOv2-M](https://github.com/yang-0201/MAF-YOLOv2/releases/download/v1.0.0/MAF-YOLOv2-M.pt)      |   640  |  15.3M  | 65.2G |      52.7%       |               69.5%                |      2.79ms   | 

MS COCO with ImageNet Pretrain

| Model             | Test Size | #Params | FLOPs |     AP<sup>val</sup>     |   AP<sub>50</sub><sup>val</sup>    | 
|:------------------|:----:|:-------:|:-----:|:------------------------:|:----------------------------------:|
| [MAF-YOLOv2-N-pretrained](https://github.com/yang-0201/MAF-YOLOv2/releases/download/v1.0.0/MAF-YOLOv2-N-pretrained.pt)      |   640  |  2.2M   | 7.2G  |          43.1%           |               59.3%                | 
| [MAF-YOLOv2-S-pretrained](https://github.com/yang-0201/MAF-YOLOv2/releases/download/v1.0.0/MAF-YOLOv2-S-pretrained.pt)      |   640  |  7.1M   | 25.3G |      49.4%      |               66.5%                | 


### Segmentation
COCO-seg

| Model                                                                                                   | Test Size | #Params | FLOPs | AP<sub>bbox</sub> | AP<sub>mask</sub> | 
|:--------------------------------------------------------------------------------------------------------|:----:|:-------:|:-----:|:-----------------:|:-----------------:|
| [MAF-YOLOv2-N-seg](https://github.com/yang-0201/MHAF-YOLO/releases/download/v1.0.0/MAF-YOLOv2-N-Seg.pt) |   640  |  2.4M   | 14.8G |       42.5%       |       35.0%       | 
| [MAF-YOLOv2-S-seg](https://github.com/yang-0201/MHAF-YOLO/releases/download/v1.0.0/MAF-YOLOv2-S-Seg.pt) |   640  |  7.8M   | 40.4G |       48.8%       |       39.7%       | 

### Classification 
ImageNet

| Model                                                                                                                | size<br><sup>(pixels) | acc<br><sup>top1 | acc<br><sup>top5 | params<br><sup>(M) | FLOPs<br><sup>(G) |
|:----------------------------------------------------------------------------------------------------------------------:|:-----------------------:|:------------------:|:------------------:|:--------------------:|:-------------------:|
| [MAF-YOLOv2-N-cls](https://github.com/yang-0201/MHAF-YOLO/releases/download/v1.0.0/MAF-YOLOv2-N-cls.pt)              | 224                   | 71.2             | 90.3             | 2.8                | 0.4               |
| [MAF-YOLOv2-S-cls](https://github.com/yang-0201/MHAF-YOLO/releases/download/v1.0.0/MAF-YOLOv2-S-cls.pt)              | 224                   | 75.5             | 92.6             | 5.6                | 1.4               |
| [MAF-YOLOv2-N-cls-finetune-384](https://github.com/yang-0201/MHAF-YOLO/releases/download/v1.0.0/MAF-YOLOv2-N-cls-finetune-384.pt) | 384                   | 73.8             | 91.8             | 2.8                | 1.3               |
| [MAF-YOLOv2-S-cls-finetune-384](https://github.com/yang-0201/MHAF-YOLO/releases/download/v1.0.0/MAF-YOLOv2-S-cls-finetune-384.pt) | 384                   | 77.5             | 93.8             | 5.6                | 4.2               |

### Rotated Object Detection
DOTA-v1.0

| Model                                                                                                   | size<br><sup>(pixels) | MS   | Pretrain | params<br><sup>(M) | FLOPs<br><sup>(G) | mAP  |
|:---------------------------------------------------------------------------------------------------------:|:-----------------------:|:----------:|:----------:|:--------------------:|:-------------------:|:------:|
| [MAF-YOLOv2-S-obb](https://github.com/yang-0201/MHAF-YOLO/releases/download/v1.0.0/MAF-YOLOv2-S-obb.pt) | 1024                  | ✔| ✘    | 7.3                | 67.3              | 81.1 |


### Others
| Model             | Test Size | #Params | FLOPs |     AP<sup>val</sup>     |   AP<sub>50</sub><sup>val</sup>    | AP<sub>s</sub><sup>val</sup>    | AP<sub>m</sub><sup>val</sup>    | AP<sub>l</sub><sup>val</sup>    | Epochs|
|:------------------|:----:|:-------:|:-----:|:------------------------:|:----------------------------------:|:----------------------------------:|:----------------------------------:|:----------------------------------:|:----------------------------------:|
| YOLOv12n |   640  |  2.6M   | 6.5G  |          40.6%            |      56.7%            | 20.2%|45.2% |  58.4%   |        600
| [YOLOv12n + MAFPN](https://github.com/yang-0201/MAF-YOLOv2/releases/download/v1.0.0/YOLOv12n_MAFPN.pt)      |   640  |  2.6M   | 8.8G  |          41.6%(+1.0)           |               57.7%(+1.0)   | 22.0%(+1.8) |45.7%(+0.5) | 58.5%(+0.1) |500(-100)                  | 

Directly replace the PAFPN in YOLOv12n with MAFPN.   [yolov12n-MAFPN](ultralytics/cfg/models/v12/yolov12n-MAFPN.yaml)


## Installation

```
conda create -n mafyolov2 python==3.9
conda activate mafyolov2
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```

## Train
Single GPU training
```python
# train.py
from ultralytics import YOLOv10
if __name__ == '__main__':
    model = YOLOv10('MAF-YOLOv2-n.yaml')
    model.train(data='coco.yaml', batch=16, device=0)

```
## Val
```python
# val.py
from ultralytics import YOLOv10
if __name__ == '__main__':
    model = YOLOv10('MAF-YOLOv2-N.pt')
    model.val(data='coco.yaml', device=0,split='val', save_json=True, batch=8)
```
## Export
End-to-End ONNX
```python
yolo export model=MAF-YOLOv2-N.pt format=onnx opset=13 simplify
```
End-to-End TensorRT
```python
yolo export model=MAF-YOLOv2-N.pt format=engine half=True simplify opset=13 workspace=16
```
or 
```python
trtexec --onnx=MAF-YOLOv2-N.onnx --saveEngine=MAF-YOLOv2-N.engine --fp16
```
Evaluation speed
```python
trtexec --loadEngine=MAF-YOLOv2-N.engine --fp16
```
## Problems and Improvements
<details><summary>Problems</summary>
<details><summary>1. Multi-GPU distributed training</summary>
One of the issues with the YOLOv10 framework is that during multi-GPU training, there is a certain probability that the program cannot be completely stopped, requiring manual intervention to kill the process.
</details>
<details><summary>2. Failed to load some pretrained weights.</summary>
 </details>
</details>

<details><summary>Improvements</summary>
<details><summary>1. Try to replace nms free</summary>
 MHAF-YOLO, like YOLOv10, uses a one-to-one head by default to achieve an NMS-free effect. However, in some smaller models or smaller datasets, using NMS combined with a one-to-many head can lead to significant improvements. For example, on the COCO dataset, the nano model shows a 1% improvement, and on private smaller-scale datasets, it can even reach over 2%. If your model isn’t concerned about the speed overhead of NMS, you can make the following modification to see the accuracy improvement:

Edit the file ultralytics/models/yolov10/val.py and uncomment lines 11 to 19.
  </details>
 </details>
 
## Citation

If our code or model is helpful to your work, please cite our paper and consider giving us a star. We would be very grateful!

```BibTeX
@article{yang2025mhaf,
  title={MHAF-YOLO: Multi-Branch Heterogeneous Auxiliary Fusion YOLO for accurate object detection},
  author={Yang, Zhiqiang and Guan, Qiu and Yu, Zhongwen and Xu, Xinli and Long, Haixia and Lian, Sheng and Hu, Haigen and Tang, Ying},
  journal={arXiv preprint arXiv:2502.04656},
  year={2025}
}
```

## Acknowledgements
* [https://github.com/THU-MIG/yolov10](https://github.com/THU-MIG/yolov10)
* [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

