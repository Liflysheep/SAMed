import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from importlib import import_module

# 导入SAMed模型的LoRA适配器
from sam_lora_image_encoder import LoRA_Sam
# 导入SAM模型注册模块
from segment_anything import sam_model_registry

# 导入训练器
from trainer import trainer_synapse
from icecream import ic

# 解析命令行参数
parser = argparse.ArgumentParser()
# 设置各种训练参数，用户可以通过命令行输入参数来定制实验
parser.add_argument('--root_path', type=str,
                    default='/data/LarryXu/Synapse/preprocessed_data/train_npz', help='root dir for data')
parser.add_argument('--output', type=str, default='/output/sam/results')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=8, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=200, help='maximum epoch number to train')
parser.add_argument('--stop_epoch', type=int,
                    default=160, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=12, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=2, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.005,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=512, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--vit_name', type=str,
                    default='vit_b', help='select one vit model')
parser.add_argument('--ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth',
                    help='Pretrained checkpoint')
parser.add_argument('--lora_ckpt', type=str, default=None, help='Finetuned lora checkpoint')
parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
parser.add_argument('--warmup', action='store_true', help='If activated, warp up the learning from a lower lr to the base_lr')
parser.add_argument('--warmup_period', type=int, default=250,
                    help='Warp up iterations, only valid whrn warmup is activated')
parser.add_argument('--AdamW', action='store_true', help='If activated, use AdamW to finetune SAM model')
parser.add_argument('--module', type=str, default='sam_lora_image_encoder')
parser.add_argument('--dice_param', type=float, default=0.8)
args = parser.parse_args()

if __name__ == "__main__":
    # 设置训练是否为确定性模式
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    # 设置随机种子，确保训练过程可重复
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 配置数据集名称和相关路径
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
        }
    }

    # 设置实验名称
    args.is_pretrain = True
    args.exp = dataset_name + '_' + str(args.img_size)

    # 生成训练结果保存路径
    snapshot_path = os.path.join(args.output, "{}".format(args.exp))
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

    # 如果保存路径不存在，则创建
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    # 注册并加载SAM模型
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                num_classes=args.num_classes,
                                                                checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                pixel_std=[1, 1, 1])

    # 动态导入LoRA适配器模块
    pkg = import_module(args.module)
    # 将LoRA模型加载到GPU
    net = pkg.LoRA_Sam(sam, args.rank).cuda()

    # 如果指定了LoRA微调检查点，则加载
    if args.lora_ckpt is not None:
        net.load_lora_parameters(args.lora_ckpt)

    # 判断是否为多掩码输出任务
    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    # 设置低分辨率参数（通常是图像嵌入大小的四倍）
    low_res = img_embedding_size * 4

    # 配置文件路径
    config_file = os.path.join(snapshot_path, 'config.txt')
    config_items = []
    # 将命令行参数写入配置文件
    for key, value in args.__dict__.items():
        config_items.append(f'{key}: {value}\n')

    # 保存配置文件
    with open(config_file, 'w') as f:
        f.writelines(config_items)

    # 根据数据集名称选择相应的训练器
    trainer = {'Synapse': trainer_synapse}
    # 执行训练过程
    trainer[dataset_name](args, net, snapshot_path, multimask_output, low_res)
