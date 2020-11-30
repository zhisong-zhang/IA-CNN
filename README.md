# IA-CNN

## Introduction
In recent years, convolutional neural networks
(CNNs) have been widely used in security, autonomous driving,
and healthcare. Even though CNNs have shown state-of-the-art
performance, they produce results that are difficult to explain and
in some cases out of control. The black-box nature of CNN makes
it lack trust. Our paper proposes an interpretable CNN structure
with attention mechanism (IA-CNN) that highly improves the
interpretability of the CNN models. Each filter of the last convlayer
only has one response (one key point) of the target object,
which is directly connected to the output. We also combine
attention mechanism to weakly supervise the last conv-layer. In
this way, our model can clearly show which features the model
extracted are the key to the output prediction. Meanwhile, our
IA-CNN structure can be used in various classical models with
higher performance in fine-grained classification and comparative
performance in the ordinary classification task. Note that our
IA-CNN structure is an end-to-end model, the last conv-layer of
which can extract key points from images automatically and is
connected to the output prediction linearly.

## Citation

## Enviroment
Our code is based on Python and Pytorch. Requirements of enviroments are as follow:
* Python: 3.7
* Pytorch: 1.5.0
* torchvision: 0.6.0

## How to use
python3 main.py --model resnet --dataset CUB --model_type ex_gradcam

## Acknowledgement
Part of the code referred to the following open source code:
* https://github.com/ducminhkhoi/InterpretableCNN
* https://github.com/machine-perception-robotics-group/attention_branch_network
* https://blog.csdn.net/weixin_41735859/article/details/106474768

