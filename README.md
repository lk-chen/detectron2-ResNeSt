# ResNeSt (Detectron2)

This repo contains an implementation of [ResNeSt](https://hangzhang.org/files/resnest.pdf) on detection and instance segmentation, using [detectron2](https://github.com/facebookresearch/detectron2) framework.


## Object Detection
<table class="tg">
  <tr>
    <th class="tg-0pky">Method</th>
    <th class="tg-0pky">Backbone</th>
    <th class="tg-0pky">mAP%</th>
    <th class="tg-0pky">download</th>
  </tr>
  <tr>
    <td rowspan="4" class="tg-0pky">Faster R-CNN</td>
    <td class="tg-0pky">ResNet-50</td>
    <td class="tg-0pky">39.25</td>
    <td class="tg-0lax">config | model | log </td>
  </tr>
  <tr>
    <td class="tg-0lax">ResNet-101</td>
    <td class="tg-0lax">41.37</td>
  </tr>
  <tr>
    <td class="tg-0lax">ResNeSt-50 (<span style="color:red">ours</span>)</td>
    <td class="tg-0lax"><b>42.33</b></td>
  </tr>
  <tr>
    <td class="tg-0lax">ResNeSt-101 (<span style="color:red">ours</span>)</td>
    <td class="tg-0lax"><b>44.72</b></td>
  </tr>
  <tr>
    <td rowspan="5" class="tg-0lax">Cascade R-CNN</td>
    <td class="tg-0lax">ResNet-50</td>
    <td class="tg-0lax">42.52</td>
  </tr>
  <tr>
    <td class="tg-0lax">ResNet-101</td>
    <td class="tg-0lax">44.03</td>
  </tr>
  <tr>
    <td class="tg-0lax">ResNeSt-50 (<span style="color:red">ours</span>)</td>
    <td class="tg-0lax"><b>45.41</b></td>
  </tr>
  <tr>
    <td class="tg-0lax">ResNeSt-101 (<span style="color:red">ours</span>)</td>
    <td class="tg-0lax"><b>47.50</b></td>
  </tr>
  <tr>
    <td class="tg-0lax">ResNeSt-200 (<span style="color:red">ours</span>)</td>
    <td class="tg-0lax"><b>49.03</b></td>
  </tr>
</table>

We train all models with FPN, SyncBN and image scale augmentation (short size of a image is pickedrandomly from 640 to 800). 1x learning rate schedule is used. All of them are reported on COCO-2017 validation dataset.



## Instance Segmentation
<table class="tg">
  <tr>
    <th class="tg-0pky">Method</th>
    <th class="tg-0pky">Backbone</th>
    <th class="tg-0pky">bbox</th>
    <th class="tg-0lax">mas</th>
    <th class="tg-0pky">download</th>
  </tr>
  <tr>
    <td rowspan="4" class="tg-0pky">Mask R-CNN</td>
    <td class="tg-0pky">ResNet-50</td>
    <td class="tg-0pky">39.97</td>
    <td class="tg-0lax">36.05</td>
    <td class="tg-0lax">config | model | log </td>
</tr>
  <tr>
    <td class="tg-0lax">ResNet-101</td>
    <td class="tg-0lax">41.78</td>
    <td class="tg-0lax">37.51</td>
  </tr>
  <tr>
    <td class="tg-0lax">ResNeSt-50 (<span style="color:red">ours</span>)</td>
    <td class="tg-0lax"><b>42.81</b></td>
    <td class="tg-0lax"><b>38.14</td>
  </tr>
  <tr>
    <td class="tg-0lax">ResNeSt-101 (<span style="color:red">ours</span>)</td>
    <td class="tg-0lax"><b>45.75</b></td>
    <td class="tg-0lax"><b>40.65</b></td>
  </tr>
  <tr>
    <td rowspan="4" class="tg-0lax">Cascade R-CNN</td>
    <td class="tg-0lax">ResNet-50</td>
    <td class="tg-0lax">43.06</td>
    <td class="tg-0lax">37.19</td>
  </tr>
  <tr>
    <td class="tg-0lax">ResNet-101</td>
    <td class="tg-0lax">44.79</td>
    <td class="tg-0lax">38.52</td>
  </tr>
  <tr>
    <td class="tg-0lax">ResNeSt-50 (<span style="color:red">ours</span>)</td>
    <td class="tg-0lax"><b>46.19</b></td>
    <td class="tg-0lax"><b>39.55</b></td>
  </tr>
  <tr>
    <td class="tg-0lax">ResNeSt-101 (<span style="color:red">ours</span>)</td>
    <td class="tg-0lax"><b>48.30</b></td>
    <td class="tg-0lax"><b>41.56</b></td>
  </tr>
</table>

All models are trained along with FPN and SyncBN. For data augmentation,input images’ shorter side are randomly scaled to one of (640, 672, 704, 736, 768, 800). All of them are reported on COCO-2017 validation dataset.




## Training and Inference
Please follow [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) to install detecron2. 

To train a model with 8 gpus, please run
```shell
python tools/train_net.py  --num-gpus 8 --config-file your_config.yaml
```

For inference
```shell
python tools/train_net.py  \
                --config-file your_config.yaml
                --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```


## Reference

**ResNeSt: Split-Attention Networks** [[arXiv](https://arxiv.org/pdf/2004.08955.pdf)]

Hang Zhang, Chongruo Wu, Zhongyue Zhang, Yi Zhu, Zhi Zhang, Haibin Lin, Yue Sun, Tong He, Jonas Muller, R. Manmatha, Mu Li and Alex Smola

```
@article{zhang2020resnest,
title={ResNeSt: Split-Attention Networks},
author={Zhang, Hang and Wu, Chongruo and Zhang, Zhongyue and Zhu, Yi and Zhang, Zhi and Lin, Haibin and Sun, Yue and He, Tong and Muller, Jonas and Manmatha, R. and Li, Mu and Smola, Alexander},
journal={arXiv preprint arXiv:2004.08955},
year={2020}
}
```

## Related
* [ResNeSt (main)](https://github.com/zhanghang1989/ResNeSt)
* ResNeSt on ImageNet ([Gluon](https://gluon-cv.mxnet.io/model_zoo/classification.html), [pytorch](https://hangzhang.org/PyTorch-Encoding/model_zoo/imagenet.html) )
* ResNeSt on Semantic Segmentation ([Gluon](https://gluon-cv.mxnet.io/model_zoo/segmentation.html), [pytorch](https://hangzhang.org/PyTorch-Encoding/model_zoo/segmentation.html))

### Contributors
[Chongruo Wu](https://github.com/chongruo), [Zhongyue Zhang](http://zhongyuezhang.com/), [Hang Zhang](https://hangzhang.org/)
