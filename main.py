# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 09:43:06 2020

@author: Administrator
G`"""

import torch
import torch.nn as nn
from torch import  optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets,models
import torch.nn.functional as F
from tqdm import tqdm
import os
from dataset import CUBDataset,VOCPartDataset,CarsDataset,CovidCTDataset
from dogs import Dogs
import argparse
import numpy as np
from vgg16 import VGG_interpretable,VGG_atten,VGG_gradcam, VGG_interpretable_atten,VGG_interpretable_gradcam, VGGNet,VGG_interpretable_gradcam2
from resnet import Resnet,Resnet_interpretable,Resnet_interpretable_gradcam,Resnet_interpretable_atten
#from mobilenet import MobileNet,Mobile_interpretable_gradcam
#from alexnet import Alexnet,Alexnet_interpretable_gradcam,Alexnet_interpretable
#from inception import Inception
from eval import accuracy, Logger, savefig
import math
from utils import Cutout
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
'''定义超参数'''
batch_size = 32 
batch_size_t = 32       # 批的大小
learning_rate = 1e-2    # 学习率
learning_rate_basemodel = 1e-2
num_epoches = 90        # 遍历训练集的次数
#explanation = False
model_half = True
lambda_ = 0.001
pretrained =False


transform_train = transforms.Compose([
    transforms.RandomSizedCrop(224),
    #transforms.CenterCrop(224),
    #transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(30),
    transforms.ToTensor(),
    #transforms.Lambda(lambda x: x.repeat(3,1,1)),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std  = [ 0.229, 0.224, 0.225 ]),
    ])

transform_test = transforms.Compose([
    #transforms.RandomSizedCrop(224),
    transforms.Resize((224, 224)),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Lambda(lambda x: x.repeat(3,1,1)),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std  = [ 0.229, 0.224, 0.225 ]),
    ])

# 定义获取梯度的函数
def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())

# 定义获取特征图的函数
def farward_hook(module, input, output):
    fmap_block.append(output)
'''————————————————
版权声明：本文为CSDN博主「景唯acr」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/weixin_41735859/article/details/106474768'''
# calculate grad_cam map
def grad_cam(grad_block,fmap_block):
    grad_block = grad_block[0]
    fmap_block = fmap_block[0]
    a,b,c,d = grad_block.size()
    atten = np.zeros((a,b,c,d), dtype=np.float32)
    for i in range(len(grad_block)):
        grads = grad_block[i].cpu().data.numpy()
        fmap = fmap_block[i].cpu().data.numpy()
        cam = np.zeros(fmap.shape[1:], dtype=np.float32)
        grads = grads.reshape([grads.shape[0],-1])
        weights = np.mean(grads, axis=1)
        for j, w in enumerate(weights):
            cam += w * fmap[j, :, :]	
        cam = np.maximum(cam, 0)
        if cam.max() != 0:
            cam = cam / cam.max()
        cam = cam + 1.0
        if args.model == 'vgg':
            atten[i] = np.ones((512,1,1)) * cam[np.newaxis,:]
        elif args.model == 'resnet':
            atten[i] = np.ones((2048,1,1)) * cam[np.newaxis,:]
        elif args.model == 'mobilenet':
            atten[i] = np.ones((1024,1,1)) * cam[np.newaxis,:]
        elif args.model == 'alexnet':
            atten[i] = np.ones((256,1,1)) * cam[np.newaxis,:]
    atten = torch.from_numpy(atten).cuda().half()
    return atten

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CapsNet')
    parser.add_argument('--dataset', type=str, default='CUB', metavar='N',
                        help='name of dataset: CIFAR10 or CUB or VOCPart or CarsDataset or Covid or dogs')
    parser.add_argument('--np_save', type=str, default='F', metavar='N',
                        help='name of T or F')
    parser.add_argument('--model_type', type=str, default='ex_gradcam', metavar='N',
                        help='norm, atten, ex or ex_atten or ex_gradcam or ex_gradcam5')
    parser.add_argument('--model', type=str, default='resnet', metavar='N',
                        help='vgg or resnet or mobilenet or alexnet or inception')
    parser.add_argument('--model_init', action='store_true', default=False, help='use Xavier Initialization')
    
    
    args = parser.parse_args()
    
    if args.cutout:
        transform_train.transforms.append(Cutout(args.cutout_length))
    
    '''下载训练集 '''
    if args.dataset == 'CIFAR10':
        train_dataset = datasets.CIFAR10('./data/CIFAR10', train=True, transform=transform_train, download=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = datasets.CIFAR10('./data/CIFAR10', train=False, transform=transform_test, download=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size_t, shuffle=False)
        num_classe = 10
        weight_folder = './model/CIFAR10/'
        title = 'CIFAR-10-' + 'vgg16'

    elif args.dataset == 'CIFAR100':
        train_dataset = datasets.CIFAR100('./data/CIFAR100', train=True, transform=transform_train, download=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = datasets.CIFAR100('./data/CIFAR100', train=False, transform=transform_test, download=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size_t, shuffle=False)
        num_classe = 100
        weight_folder = './model/CIFAR100/'
        title = 'CIFAR-100-' + 'vgg16'
        
    elif args.dataset == 'CUB':
        train_dataset = CUBDataset('./data/CUB', train=True, transform=transform_train, download=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = CUBDataset('./data/CUB', train=False, transform=transform_test, download=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size_t, shuffle=False)
        num_classe = 200
        weight_folder = './model/CUB/'
        title = 'CUB-200-' + 'vgg16'

    elif args.dataset == 'VOCPart':
        train_dataset = VOCPartDataset('./data', train=True, transform=transform_train, download=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = VOCPartDataset('./data', train=False, transform=transform_test, download=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size_t, shuffle=False)
        num_classe = 6
        weight_folder = './model/VOCPart/'
        title = 'VOCPart-' + 'vgg16'
        
    elif args.dataset == 'CarsDataset':
        #train_dataset = CarsDataset('test', './data/Cars/testing/extracted/', './data/Cars/cars_metas/cars_train_annos.mat')
        train_dataset = CarsDataset('train', './data/Cars/training/extracted/', './data/Cars/cars_metas/cars_train_annos.mat')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = CarsDataset('test', './data/Cars/testing/extracted/', './data/Cars/cars_metas/cars_test_annos_withlabels.mat')
        test_loader = DataLoader(test_dataset, batch_size=batch_size_t, shuffle=False)
        num_classe = 196
        weight_folder = './model/Cars/'
        title = 'Cars-' + 'vgg16'

    elif args.dataset == 'dogs':
        train_dataset = Dogs('./data/dogs', train=True, transform=transform_train, download=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = Dogs('./data/dogs', train=False, transform=transform_test, download=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size_t, shuffle=False)
        num_classe = 120
        weight_folder = './model/dogs/'
        title = 'VOCPart-' + 'vgg16'
        
    elif args.dataset == 'Covid':
        train_dataset = CovidCTDataset(root_dir='./data/Covid/Images-processed',
                              txt_COVID='./data/Covid/Data-split/COVID/trainCT_COVID.txt',
                              txt_NonCOVID='./data/Covid/Data-split/NonCOVID/trainCT_NonCOVID.txt',
                              transform= transform_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = CovidCTDataset(root_dir='./data/Covid/Images-processed',
                              txt_COVID='./data/Covid/Data-split/COVID/testCT_COVID.txt',
                              txt_NonCOVID='./data/Covid/Data-split/NonCOVID/testCT_NonCOVID.txt',
                              transform= transform_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size_t, shuffle=False)
        num_classe = 2
        weight_folder = './model/Covid/'
        title = 'Covid-' + 'vgg16'
        
    if not os.path.isdir(weight_folder):
        print('No weight_folder found!')
        os.makedirs(weight_folder)
    
    
    '''创建model实例对象，并检测是否支持使用GPU'''
    if args.model == 'vgg':
        if args.model_type == 'ex_atten':
            model = VGG_interpretable_atten(num_classes=num_classe)
        elif args.model_type == 'ex':
            model = VGG_interpretable(num_classes=num_classe)
        elif args.model_type == 'atten':
            model = VGG_atten(num_classes=num_classe)
        elif args.model_type == 'gradcam':
            model = VGG_gradcam(num_classes=num_classe)
        elif args.model_type == 'ex_gradcam':
            model = VGG_interpretable_gradcam(num_classes=num_classe)
        elif args.model_type == 'ex_gradcam5':
            model = VGG_interpretable_gradcam(num_classes=num_classe)
        elif args.model_type == 'ex_gradcam2':
            model = VGG_interpretable_gradcam2(num_classes=num_classe)
        else:
            model = VGGNet(num_classes=num_classe)
            #model = VGG_gradcam(num_classes=num_classe)
    elif args.model == 'resnet':
        if args.model_type == 'ex_atten':
            model = Resnet_interpretable_atten(num_classes=num_classe)
        elif args.model_type == 'ex':
            model = Resnet_interpretable(num_classes=num_classe)
        elif args.model_type == 'ex_gradcam':
            model = Resnet_interpretable_gradcam(num_classes=num_classe)
        elif args.model_type == 'ex_gradcam2':
            model = VGG_interpretable_gradcam2(num_classes=num_classe)
        elif args.model_type == 'norm':
            model = Resnet(num_classes=num_classe)
    use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
    if use_gpu:
        model = model.cuda()
    if model_half:
        model = model.half()
    if args.model_init:
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                
    '''定义loss和optimizer'''
    criterion = nn.CrossEntropyLoss()
    base_params = list(map(id, model.base_model.parameters()))
    logits_params = filter(lambda p: id(p) not in base_params, model.parameters())
    params = [
        {"params": logits_params, "lr": learning_rate},
        {"params": model.base_model.parameters(), "lr": learning_rate_basemodel},
    ]
    optimizer = optim.SGD(params)
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate,weight_decay=0.01)
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    #optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.9)
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    #scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch+1))
    scheduler = optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1, last_epoch=-1)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, 60, gamma=0.1)

    if pretrained:
        weight_file = weight_folder + 'model_best.pth'
        model.load_state_dict(torch.load(weight_file))
        #weight_file = weight_folder + "/model_best_full.pth"
        #checkpoint = torch.load(weight_file)
        #model.load_state_dict(checkpoint['net'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        #start_epoch = checkpoint['epoch'] + 1
        logger = Logger(os.path.join(weight_folder, 'log.txt'), title=title, resume=True)
        print('model load finish!')
    
    logger = Logger(os.path.join(weight_folder, 'log.txt'), title=title)
    logger.set_names(['Train Loss', 'Valid Loss', 'Train Acc_1.', 'Train Acc_5.', 'Valid Acc_1.', 'Valid Acc_5.'])
    
    
    
    
    '''训练模型'''
    steps = math.ceil(len(train_dataset) / batch_size)
    steps_test = math.ceil(len(test_dataset) / batch_size_t)
    best_acc = 0.0
    if args.np_save == 'T':
        loss_train = []
        acc1_train = []
        acc5_train = []
        loss_test = []
        acc1_test = []
        acc5_test = []
    for epoch in range(num_epoches):
        print('*' * 25, 'epoch {}'.format(epoch + 1), '*' * 25)      # .format为输出格式，formet括号里的即为左边花括号的输出
        running_loss = 0.0
        running_loss_1 = 0.0
        running_loss_2 = 0.0
        running_acc_1 = 0.0
        running_acc_5 = 0.0
        with tqdm(total=steps) as pbar:
            for data in train_loader:
                if args.dataset == 'Covid':
                    img, targets = data['img'],data['label']
                else:
                    img, targets = data
                if model_half:
                    img = img.half()
                # cuda
                if use_gpu:
                    img = img.cuda()
                    targets = targets.cuda()
                img = Variable(img)
                targets = Variable(targets.squeeze())
                # 向前传播
                if args.model_type == 'ex':
                    out, x1, loss_1= model(img)
                    loss = criterion(out, targets)
                    loss = loss + lambda_ * (loss_1.sum())
                elif args.model_type == 'atten':
                    att_outputs,out= model(img)
                    loss = criterion(out, targets)
                    att_loss = criterion(att_outputs, targets)
                    loss = loss + att_loss
                elif args.model_type == 'ex_atten':
                    att_outputs,out, x1, loss_1= model(img)
                    loss = criterion(out, targets)
                    att_loss = criterion(att_outputs, targets)
                    loss = loss + att_loss #+lambda_ * loss_1.sum()
                elif args.model_type == 'gradcam':
                    grad_block = list()
                    fmap_block = list()
                    if args.model == 'vgg':
                        handle_feat = model.avgpool.register_forward_hook(farward_hook)
                        handle_grad = model.avgpool.register_backward_hook(backward_hook)
                    elif args.model != 'vgg':
                        handle_feat = model.base_model.register_forward_hook(farward_hook)
                        handle_grad = model.base_model.register_backward_hook(backward_hook)
                    out= model(img,img)
                    optimizer.zero_grad()
                    class_loss = 0.0
                    for i in range(len(out)):
                        idx = np.argmax(out[i].cpu().data.numpy())
                        class_loss += out[i,idx]
                    class_loss = class_loss/len(out)
                    class_loss.backward(retain_graph=True)
                    handle_feat.remove()
                    handle_grad.remove()
                    atten = grad_cam(grad_block,fmap_block)
                    att_loss = criterion(out, targets)
                    out= model(img,cam=False,att=atten)
                    loss = criterion(out, targets)
                    #loss = (1-(1/(epoch+1))) * loss + (1/(epoch+1)) * att_loss + (1-(1/(epoch+1))) * lambda_ * loss_1.sum()
                    loss = loss
                elif args.model_type == 'ex_gradcam':
                    grad_block = list()
                    fmap_block = list()
                    if args.model == 'vgg':
                        handle_feat = model.avgpool.register_forward_hook(farward_hook)
                        handle_grad = model.avgpool.register_backward_hook(backward_hook)
                    elif args.model != 'vgg':
                        handle_feat = model.base_model.register_forward_hook(farward_hook)
                        handle_grad = model.base_model.register_backward_hook(backward_hook)
                    out,_,_ = model(img,img)
                    optimizer.zero_grad()
                    class_loss = 0.0
                    for i in range(len(out)):
                        idx = np.argmax(out[i].cpu().data.numpy())
                        class_loss += out[i,idx]
                    class_loss = class_loss/len(out)
                    class_loss.backward(retain_graph=True)
                    handle_feat.remove()
                    handle_grad.remove()
                    atten = grad_cam(grad_block,fmap_block)
                    att_loss = criterion(out, targets)
                    out, x1, loss_1= model(img,cam=False,att=atten)
                    loss = criterion(out, targets)
                    loss = (1-(1/(epoch+1))) * loss + (1/(epoch+1)) * att_loss + (1-(1/(epoch+1))) * lambda_ * loss_1.sum()
                    #loss = 0.6 * loss + 0.4 * att_loss + lambda_ * loss_1.sum()
                elif args.model_type == 'ex_gradcam5':
                    grad_block = list()
                    fmap_block = list()
                    if args.model == 'vgg':
                        handle_feat = model.avgpool.register_forward_hook(farward_hook)
                        handle_grad = model.avgpool.register_backward_hook(backward_hook)
                    elif args.model != 'vgg':
                        handle_feat = model.base_model.register_forward_hook(farward_hook)
                        handle_grad = model.base_model.register_backward_hook(backward_hook)
                    out,_,_= model(img,img)
                    optimizer.zero_grad()
                    class_loss = 0.0
                    for i in range(len(out)):
                        #idx = np.argmax(out[i].cpu().data.numpy())
                        idex = np.argsort(out[i].cpu().data.numpy())
                        for k in range(5):
                            class_loss += out[i,idex[k]]
                    class_loss = class_loss/len(out)
                    class_loss.backward(retain_graph=True)
                    handle_feat.remove()
                    handle_grad.remove()
                    atten = grad_cam(grad_block,fmap_block)
                    att_loss = criterion(out, targets)
                    out, x1, loss_1= model(img,cam=False,att=atten)
                    loss = criterion(out, targets)
                    loss = (1-(1/(epoch+1))) * loss + (1/(epoch+1)) * att_loss + (1-(1/(epoch+1))) * lambda_ * loss_1.sum()
                    #loss = (1-(1/(epoch+1))) * loss + 0.5*(1/(epoch+1)) * att_loss

                elif args.model_type == 'ex_gradcam2':
                    grad_block = list()
                    fmap_block = list()
                    handle_feat = model.avgpool.register_forward_hook(farward_hook)
                    handle_grad = model.avgpool.register_backward_hook(backward_hook)
                    out,_,_,_,_ = model(img,img)
                    optimizer.zero_grad()
                    class_loss = 0.0
                    for i in range(len(out)):
                        idx = np.argmax(out[i].cpu().data.numpy())
                        class_loss += out[i,idx]
                    class_loss = class_loss/len(out)
                    class_loss.backward(retain_graph=True)
                    handle_feat.remove()
                    handle_grad.remove()
                    atten = grad_cam(grad_block,fmap_block)
                    att_loss = criterion(out, targets)
                    out, x1, x2, loss_1, loss_2= model(img,cam=False,att=atten)
                    loss = criterion(out, targets)
                    #loss = (1-(1/(epoch+1))) * loss + (1/(epoch+1)) * att_loss + (1-(1/(epoch+1))) * lambda_ * loss_1.sum()
                    loss = 0.4 * loss + 0.6 * att_loss + lambda_ * (loss_1.sum() + loss_2.sum())
                else:
                    out = model(img)
                    loss = criterion(out, targets)
                if args.model_type != 'norm' and args.model_type != 'gradcam'and args.model_type != 'atten':
                    running_loss_1 += loss_1.sum().item() * targets.size(0)
                    #running_loss_2 += loss_2.sum().item() * targets.size(0)
                running_loss += loss.item() * targets.size(0)
                if args.dataset == 'Covid':
                    prec1, prec5 = accuracy(out.data, targets.data, topk=(1, 2))
                else:
                    prec1, prec5 = accuracy(out.data, targets.data, topk=(1, 5))
                #_, pred = torch.max(out, 1)     # 预测最大值所在的位置标签
                #num_correct = (pred == targets).sum()
                #accuracy = (pred == targets).float().mean()
                #running_acc += num_correct.item()
                running_acc_1 += prec1.item()
                running_acc_5 += prec5.item()
                # 向后传播
                optimizer.zero_grad()
                
                if args.model_type != 'norm' and args.model_type != 'gradcam'and args.model_type != 'atten':
                    #compute and add gradient
                    x1.retain_grad()
                    #x2.retain_grad()
            
                    loss_1.backward(Variable(torch.ones(*loss_1.size()).cuda(0)), retain_graph=True)
                    x1_grad = x1.grad.clone()
                    optimizer.zero_grad()
                    
                    #loss_2.backward(Variable(torch.ones(*loss_2.size()).cuda(0)), retain_graph=True)
                    #x2_grad = x2.grad.clone()
                    #optimizer.zero_grad()
                
                loss.backward()
                if args.model_type != 'norm' and args.model_type != 'gradcam'and args.model_type != 'atten':
                    x1.grad = lambda_ * x1.grad
                    #x2.grad += lambda_ * x2_grad
                optimizer.step()
                if args.model_type == 'norm' or args.model_type == 'gradcam'or args.model_type == 'atten':
                    pbar.set_postfix(loss=loss.data.cpu().numpy(), acc_prec1=prec1.cpu().numpy(),acc_prec5=prec5.cpu().numpy())
                elif args.model_type == 'ex':
                    pbar.set_postfix(loss=loss.data.cpu().numpy(),loss_1=loss_1.sum().data.cpu().numpy(), acc_prec1=prec1.cpu().numpy(),acc_prec5=prec5.cpu().numpy())
                else:
                    pbar.set_postfix(loss=loss.data.cpu().numpy(),att_loss=att_loss.data.cpu().numpy(),loss_1=loss_1.sum().data.cpu().numpy(), acc_prec1=prec1.cpu().numpy(),acc_prec5=prec5.cpu().numpy())
                pbar.update()
                #break
        if args.model_type != 'norm' and args.model_type != 'gradcam'and args.model_type != 'atten':
            print('Finish {} epoch, Loss: {:.6f}, Acc_prec1: {:.6f}, Acc_prec5: {:.6f}, loss_1:{:.6f},lr: {:.6f}'.format(
                epoch + 1, running_loss / (len(train_dataset)), running_acc_1 / (steps), 
                running_acc_5 / (steps), running_loss_1 / (steps),optimizer.param_groups[0]['lr']))
        else:
            print('Finish {} epoch, Loss: {:.6f}, Acc_prec1: {:.6f}, Acc_prec5: {:.6f}, lr: {:.6f}'.format(
                epoch + 1, running_loss / (len(train_dataset)), running_acc_1 / (steps), 
                running_acc_5 / (steps), optimizer.param_groups[0]['lr']))
        scheduler.step()
        if args.np_save == 'T':
            loss_train.append(running_loss / (len(train_dataset)))
            acc1_train.append(running_acc_1 / (steps))
            acc5_train.append(running_acc_5 / (steps))
        
        model.eval()    # 模型评估
        eval_loss = 0
        eval_acc_1 = 0
        eval_acc_5 = 0
        for data in test_loader:      # 测试模型
            if args.dataset == 'Covid':
                img, targets = data['img'],data['label']
            else:
                img, targets = data
            #with torch.no_grad():
            if model_half:
                img = img.half()
            if use_gpu:
                #img = Variable(img, volatile=True).cuda()
                #targets = Variable(targets, volatile=True).cuda()
                img = img.cuda()
                targets = targets.squeeze().cuda()
            #else:
            img = Variable(img)
            targets = Variable(targets)
            if args.model_type == 'ex_atten':
                _, out, x1, loss_1= model(img)
                loss = criterion(out, targets)
                loss = loss + lambda_ * loss_1.sum()
            elif args.model_type == 'ex':
                out, x1, loss_1= model(img)
                loss = criterion(out, targets)
                loss = loss + lambda_ * (loss_1.sum())
            elif args.model_type == 'atten':
                _,out= model(img)
                loss = criterion(out, targets)
                loss = loss
            elif args.model_type == 'ex_gradcam':
                grad_block = list()
                fmap_block = list()
                if args.model == 'vgg':
                    handle_feat = model.avgpool.register_forward_hook(farward_hook)
                    handle_grad = model.avgpool.register_backward_hook(backward_hook)
                elif args.model != 'vgg':
                    handle_feat = model.base_model.register_forward_hook(farward_hook)
                    handle_grad = model.base_model.register_backward_hook(backward_hook)
                out,_,_ = model(img,img)
                model.zero_grad()
                class_loss = 0.0
                for i in range(len(out)):
                    idx = np.argmax(out[i].cpu().data.numpy())
                    class_loss += out[i,idx]
                class_loss = class_loss/len(out)
                #class_loss.requires_grad = True
                class_loss.backward()
                handle_feat.remove()
                handle_grad.remove()
                atten = grad_cam(grad_block,fmap_block)
                out, x1, loss_1= model(img,cam=False,att=atten)
                loss = criterion(out, targets)
                #loss = loss + lambda_ * loss_1.sum()
                loss = (1-(1/(epoch+1))) * loss + (1-(1/(epoch+1))) * lambda_ * loss_1.sum()
            elif args.model_type == 'ex_gradcam5':
                grad_block = list()
                fmap_block = list()
                handle_feat = model.avgpool.register_forward_hook(farward_hook)
                handle_grad = model.avgpool.register_backward_hook(backward_hook)
                out,_,_ = model(img,img)
                model.zero_grad()
                class_loss = 0.0
                for i in range(len(out)):
                    #idx = np.argmax(out[i].cpu().data.numpy())
                    idex = np.argsort(out[i].cpu().data.numpy())
                    for k in range(5):
                        class_loss += out[i,idex[k]]
                class_loss = class_loss/len(out)
                #class_loss.requires_grad = True
                class_loss.backward()
                handle_feat.remove()
                handle_grad.remove()
                atten = grad_cam(grad_block,fmap_block)
                out,_,_ = model(img,cam=False,att=atten)
                loss = criterion(out, targets)
                #loss = loss + lambda_ * loss_1.sum()
                loss = loss
            elif args.model_type == 'gradcam':
                grad_block = list()
                fmap_block = list()
                handle_feat = model.avgpool.register_forward_hook(farward_hook)
                handle_grad = model.avgpool.register_backward_hook(backward_hook)
                out = model(img,img)
                model.zero_grad()
                class_loss = 0.0
                for i in range(len(out)):
                    idx = np.argmax(out[i].cpu().data.numpy())
                    class_loss += out[i,idx]
                class_loss = class_loss/len(out)
                #class_loss.requires_grad = True
                class_loss.backward()
                handle_feat.remove()
                handle_grad.remove()
                atten = grad_cam(grad_block,fmap_block)
                out = model(img,cam=False,att=atten)
                loss = criterion(out, targets)
                #loss = loss + lambda_ * loss_1.sum()
                loss = loss
            else:
                out = model(img)
                out_result = out.data.cpu().numpy()
                loss = criterion(out, targets)
            eval_loss += loss.item() * targets.size(0)
            if args.dataset == 'Covid':
                prec1, prec5 = accuracy(out.data, targets.data, topk=(1, 2))
            else:
                prec1, prec5 = accuracy(out.data, targets.data, topk=(1, 5))
            #_, pred = torch.max(out, 1)
            #num_correct = (pred == targets).sum()
            #eval_acc += num_correct.item()
            eval_acc_1 += prec1.item()
            eval_acc_5 += prec5.item()
        print('Test Loss: {:.6f}, Acc_prec1: {:.6f}, Acc_prec5: {:.6f}'.format(eval_loss / (len(
            test_dataset)), eval_acc_1 / steps_test, 
            eval_acc_5 / steps_test))
        if args.np_save == 'T':
            loss_test.append(eval_loss / (len(test_dataset)))
            acc1_test.append(eval_acc_1 / steps_test)
            acc5_test.append(eval_acc_5 / steps_test)
    
        # 保存模型
        if eval_acc_1 / steps_test >= best_acc:
            state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch }
            torch.save(state, weight_folder + "/model_best_full.pth")
            torch.save(model.state_dict(), weight_folder + "/model_best.pth")
            best_acc = eval_acc_1 / steps_test
        torch.save(model.state_dict(), weight_folder + "/model.pth")
        logger.append([running_loss / (len(train_dataset)), 
                       eval_loss / (len(test_dataset)), 
                       running_acc_1 / (steps), 
                       running_acc_5 / (steps), 
                       eval_acc_1 / steps_test, 
                       eval_acc_5 / steps_test])
    logger.close()
    logger.plot()
    savefig(os.path.join(weight_folder, 'log.eps'))
    if args.np_save == 'T':
        np.save('loss_train',loss_train)
        np.save('loss_test',loss_test)
        np.save('acc1_train',acc1_train)
        np.save('acc1_test',acc1_test)
        np.save('acc5_train',acc5_train)
        np.save('acc5_test',acc5_test)
