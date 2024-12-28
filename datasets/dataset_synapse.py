import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from einops import repeat
from icecream import ic


# 定义一个函数用于随机旋转和翻转图像和标签
def random_rot_flip(image, label):
    # 随机选择旋转的角度（90度的倍数）
    k = np.random.randint(0, 4)  # 0, 1, 2, or 3 -> 0° 90° 180° 270°
    image = np.rot90(image, k)  # 对图像进行旋转
    label = np.rot90(label, k)  # 对标签进行旋转
    # 随机选择翻转的轴（水平或垂直）
    axis = np.random.randint(0, 2)  # 0表示水平翻转，1表示垂直翻转
    image = np.flip(image, axis=axis).copy()  # 对图像进行翻转
    label = np.flip(label, axis=axis).copy()  # 对标签进行翻转
    return image, label


# 定义一个函数用于随机旋转图像和标签
def random_rotate(image, label):
    # 随机选择旋转角度
    angle = np.random.randint(-20, 20)  # 旋转角度范围：-20到20度
    image = ndimage.rotate(image, angle, order=0, reshape=False)  # 对图像进行旋转
    label = ndimage.rotate(label, angle, order=0, reshape=False)  # 对标签进行旋转
    return image, label


# 定义一个类，用于随机数据增强（旋转、翻转等）
class RandomGenerator(object):
    def __init__(self, output_size, low_res):
        # 初始化方法，设置目标图像尺寸和低分辨率尺寸
        self.output_size = output_size  # 输出图像的目标尺寸
        self.low_res = low_res  # 低分辨率的尺寸

    def __call__(self, sample):
        # 定义当调用该对象时的行为（数据增强过程）
        image, label = sample['image'], sample['label']

        # 随机应用旋转和翻转增强
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        
        # 获取图像的尺寸
        x, y = image.shape
        # 如果图像尺寸与目标尺寸不一致，进行缩放
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # 3表示使用立方体插值
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)  # 标签使用最近邻插值

        # 生成低分辨率标签
        label_h, label_w = label.shape
        low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w), order=0)

        # 将图像数据转换为torch tensor，并添加通道维度
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)  # 形状变为 [1, H, W]
        # 复制图像数据，形成三个通道，模拟RGB图像
        image = repeat(image, 'c h w -> (repeat c) h w', repeat=3)  # 重复图像三次，形成[3, H, W]
        
        # 标签和低分辨率标签也转为tensor
        label = torch.from_numpy(label.astype(np.float32))
        low_res_label = torch.from_numpy(low_res_label.astype(np.float32))

        # 返回一个字典，包括图像、标签和低分辨率标签
        sample = {'image': image, 'label': label.long(), 'low_res_label': low_res_label.long()}
        return sample


# 定义一个用于加载和处理数据的类，继承自Dataset
class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        # 初始化方法，设置基本目录、数据分割、是否进行数据变换等
        self.transform = transform  # 使用torch中的transform
        self.split = split  # 数据分割，通常为"train"或"test"
        # 从文件中读取样本列表
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir  # 数据文件所在的根目录

    def __len__(self):
        # 返回数据集的大小
        return len(self.sample_list)

    def __getitem__(self, idx):
        # 获取指定索引的数据样本
        if self.split == "train":
            # 如果是训练集，从npz文件中读取数据
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)  # 加载npz文件
            image, label = data['image'], data['label']  # 从文件中获取图像和标签
        else:
            # 如果是测试集，从h5文件中读取数据
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)  # 加载h5文件
            image, label = data['image'][:], data['label'][:]  # 获取图像和标签

        # 输入维度应该一致
        # 由于自然图像的通道数通常为3，这里医疗图像的通道数也设置为3（通常单通道的医学图像需要转为3通道）
        
        # 创建一个字典，将图像和标签存入
        sample = {'image': image, 'label': label}
        # 如果设置了transform，应用数据增强
        if self.transform:
            sample = self.transform(sample)
        
        # 将当前样本的名称存入
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
