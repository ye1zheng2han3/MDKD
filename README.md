# Improve Knowledge Distillation Via Multi-level Normalization And Multi-level Decoupling

## Usage

The code is built on [mdistiller](<https://github.com/megvii-research/mdistiller>), [Logit-Standardization-in-Knowledge-Distillation](<https://github.com/sunshangquan/logit-standardization-KD>) and [CTKD](<https://github.com/zhengli97/CTKD>).

<img src=.github/MDKD+.jpg width=100% /> 
Optimizal Normalization:

### Installation

Environments:

- Python 3.8
- PyTorch 1.7.0

Install the package:

```
sudo pip3 install -r requirements.txt
sudo python setup.py develop
```

## Distilling CNNs

### CIFAR-100

- Download the [`cifar_teachers.tar`](<https://github.com/megvii-research/mdistiller/releases/tag/checkpoints>) and untar it to `./download_ckpts` via `tar xvf cifar_teachers.tar`.

1. For MDKD+ and MDKD(ours)

  ```bash
  # MDKD+
  python tools/train.py --cfg configs/cifar100/mdkd/resnet32x4_resnet8x4.yaml --logit-stand --base-temp 4 --kd-weight 9 
  # MDKD
  python tools/train.py --cfg configs/cifar100/mdkd/resnet32x4_resnet8x4.yaml
  ```

2. For KD

  ```bash
  # KD
  python tools/train.py --cfg configs/cifar100/kd/resnet32x4_resnet8x4.yaml
  ```

3. For DKD

  ```bash
  # DKD
  python tools/train.py --cfg configs/cifar100/dkd/resnet32x4_resnet8x4.yaml 
  ```
4. For MLKD

  ```bash
  # MLKD
  python tools/train.py --cfg configs/cifar100/mlkd/resnet32x4_resnet8x4.yaml
  ```

5. For CTKD

Please refer to [CTKD](./CTKD).

