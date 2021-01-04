import numpy as np
import matplotlib as mpl
import matplotlib.cm
import imageio
import os
import os.path


def color_image(image, num_classes=5):
    norm = mpl.colors.Normalize(vmin=0., vmax=num_classes)
    # mycm = mpl.cm.get_cmap('Set1')
    mycm = mpl.cm.get_cmap('GnBu')
    return mycm(norm(image))


def dice_coef(y_true, y_pred):
    y_true_f = y_true
    y_pred_f = y_pred
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + 0.0004) / (np.sum(y_true_f) + np.sum(y_pred_f) + 0.0004)


def precision_coef(y_true, y_pred):
    y_true_f = y_true
    y_pred_f = y_pred
    intersection = np.sum(y_true_f * y_pred_f)
    return (intersection + 0.0004) / (np.sum(y_pred_f) + 0.0004)


def recall_coef(y_true, y_pred):
    y_true_f = y_true
    y_pred_f = y_pred
    intersection = np.sum(y_true_f * y_pred_f)
    return (intersection + 0.0004) / (np.sum(y_true_f) + 0.0004)


def create_gif(gif_name, path, duration=0.1):
    '''
    生成gif文件，原始图片仅支持png格式
    gif_name ： 字符串，所生成的 gif 文件名，带 .gif 后缀
    path :      需要合成为 gif 的图片所在路径
    duration :  gif 图像时间间隔
    '''
    frames = []
    pngFiles = os.listdir(path)
    image_list = [os.path.join(path, f) for f in pngFiles]
    for image_name in image_list:
        # 读取 png 图像文件
        frames.append(imageio.imread(image_name))
    # 保存为 gif
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return


def Region1(data, label):
    completeTumor = np.zeros(data.shape)
    syntheticData = np.zeros(label.shape)
    completeTumor[data != 0] = 1
    syntheticData[label != 0] = 1
    return completeTumor, syntheticData


def Region2(data, label):
    completeTumor = np.zeros(data.shape)
    syntheticData = np.zeros(label.shape)
    completeTumor[np.logical_not(np.logical_or((data == 0), (data == 2)))] = 1
    syntheticData[np.logical_not(np.logical_or((label == 0), (label == 2)))] = 1
    return completeTumor, syntheticData


def Region3(data, label):
    completeTumor = np.zeros(data.shape)
    syntheticData = np.zeros(label.shape)
    completeTumor[data == 4] = 1
    syntheticData[label == 4] = 1
    return completeTumor, syntheticData
