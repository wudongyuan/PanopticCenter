# @Time    : 2020/10/28 下午4:28
# @Author  : uestcer
# @Software: PyCharm
import os
from glob import glob

import numpy as np
import SimpleITK as sitk
import re

OT_path = []
Flair_path = []
T1_path = []
T1c_path = []
T2_path = []
img_list = []
Path = r'/media/uestcer/Projects/WDY/datasets/Brats/BRATS2015/'

def cutSlice1(path):
    for p in path:
        mha = sitk.ReadImage(p)
        (filename, extension) = os.path.splitext(p)
        img_array = sitk.GetArrayFromImage(mha)
        index = 1
        for i in img_array:
            img = sitk.GetImageFromArray(i)
            name = (filename + '_' + str(index) + '.nii.gz').replace('BRATS2015', 'BRATS2015_slice')
            (filepath, tempfilename) = os.path.split(name)
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            sitk.WriteImage(sitk.Cast(img, sitk.sitkUInt8), name)
            index = index + 1


# 只包含肿瘤
def cutSlice2(path, have_tumor_data):
    for p in path:
        mha = sitk.ReadImage(p)
        (filename, extension) = os.path.splitext(p)
        name = filename.split('/')[10]
        img_array = sitk.GetArrayFromImage(mha)
        for f in have_tumor_data:
            if name in f:
                tumor_index = f.split('-')[2].split(' ')
                for index in tumor_index:
                    img = sitk.GetImageFromArray(img_array[int(index)-1])
                    name = (filename + '_' + str(index) + '.nii.gz').replace('BRATS2015', 'BRATS2015_slice')
                    (filepath, tempfilename) = os.path.split(name)
                    if not os.path.exists(filepath):
                        os.makedirs(filepath)
                    sitk.WriteImage(sitk.Cast(img, sitk.sitkUInt8), name)


for LorH in os.listdir(Path+'Training'):
    path = os.path.join(Path+'Training', LorH)
    T1_path = glob(path + '/*/*/*T1.*.mha')
    T1c_path = glob(path + '/*/*/*T1c*.mha')
    Flair_path = glob(path + '/*/*/*Flair*.mha')
    T2_path = glob(path + '/*/*/*T2*.mha')
    OT_path = glob(path + '/*/*/*OT*.mha')

    with open("/media/uestcer/Projects/WDY/datasets/Brats/BRATS2015_slice/train.txt", "r") as f:  # 打开文件
        data = f.readlines()
        have_tumor_data= []
        for line in (data):
            have_tumor_data.append(line[0:-2])
        cutSlice2(T1_path, have_tumor_data)
        cutSlice2(T1c_path, have_tumor_data)
        cutSlice2(Flair_path, have_tumor_data)
        cutSlice2(T2_path, have_tumor_data)
        cutSlice2(OT_path, have_tumor_data)

