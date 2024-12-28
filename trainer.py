import argparse
import logging
import os
import random
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from utils import DiceLoss, Focal_loss
from torchvision import transforms
from icecream import ic

# 定义计算损失的函数
#这里需要修改
def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight:float=0.8):
    # 从模型输出中提取低分辨率的logits
    low_res_logits = outputs['low_res_logits']
    # 计算交叉熵损失
    loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
    # 计算Dice损失
    loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    # 结合交叉熵损失和Dice损失进行加权
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice


# 训练函数，主要用于训练模型
def trainer_synapse(args, model, snapshot_path, multimask_output, low_res):
    # 导入数据集和随机数据生成器
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    
    # 配置日志记录
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))  # 记录参数配置

    # 初始化学习率、类别数、批量大小等
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    # 加载训练数据集
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size], low_res=[low_res, low_res])]))
    print("The length of train set is: {}".format(len(db_train)))  # 打印训练集大小

    # 数据加载时的随机种子初始化函数
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # 创建DataLoader，用于加载训练数据
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    # 如果使用多个GPU，则使用DataParallel
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()  # 设置模型为训练模式

    # 定义损失函数：交叉熵损失和Dice损失
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes + 1)

    # 如果有warmup，调整初始学习率
    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr

    # 根据选择的优化器类型，使用AdamW或SGD优化器
    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001)

    # 设置TensorBoard日志记录
    writer = SummaryWriter(snapshot_path + '/log')

    # 初始化训练参数
    iter_num = 0
    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    max_iterations = args.max_epochs * len(trainloader)  # 计算最大迭代次数
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    # 训练过程中记录最佳性能
    best_performance = 0.0

    # 训练过程的进度条
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        # 逐批次训练
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # [b, c, h, w], [b, h, w]
            low_res_label_batch = sampled_batch['low_res_label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()  # 将数据转到GPU
            low_res_label_batch = low_res_label_batch.cuda()

            # 确保图像数据正确
            assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'

            # 前向传播：通过模型计算输出
            outputs = model(image_batch, multimask_output, args.img_size)

            # 计算损失
            loss, loss_ce, loss_dice = calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, args.dice_param)

            # 清空梯度，反向传播，更新权重
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 如果进行warmup，动态调整学习率
            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                # 学习率衰减
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            # 更新迭代次数
            iter_num = iter_num + 1

            # 记录训练过程中的指标
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            # 打印日志
            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            # 每20次迭代记录一次训练图像和结果
            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())  # 归一化图像
                writer.add_image('train/Image', image, iter_num)
                
                output_masks = outputs['masks']
                output_masks = torch.argmax(torch.softmax(output_masks, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', output_masks[1, ...] * 50, iter_num)

                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        # 每20个epoch保存一次模型
        save_interval = 20
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        # 最后一轮或停止epoch时保存模型
        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    # 关闭TensorBoard日志
    writer.close()
    return "Training Finished!"
