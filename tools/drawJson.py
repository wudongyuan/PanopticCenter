# @Time    : 2020/9/29 下午5:07
# @Author  : uestcer
# @Software: PyCharm
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

fig, ax = plt.subplots()
polygons = []
jsonDict = []

with open('/media/uestcer/Projects/WDY/projects/panoptic-deeplab/datasets/cityscapes/gtFine/train/bochum/bochum_000000_001519_gtFine_polygons.json','r',encoding='utf8') as fp:
    strF = fp.read()
    if len(strF) > 0:
        datas = json.loads(strF)
    else:
        datas = {}
    imgWidth = datas['imgWidth']
    imgHeight = datas['imgHeight']
    # plt.figure(figsize=(imgWidth, imgHeight))
    objects = datas['objects']
    label=objects[5]['label']
    gemfield_polygons = [objects[5]['polygon']]

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'

    # x, y = np.loadtxt('test.txt', delimiter=',', unpack=True)
    x=[]
    y=[]
    for i in gemfield_polygons[0]:
        x.append(i[0])
        y.append(i[1])
    x.append(gemfield_polygons[0][0][0])
    y.append(gemfield_polygons[0][0][1])

    plt.figure(figsize=(imgWidth/100, imgHeight/100))
    plt.plot(x, y, color='red', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(label)
    plt.legend()
    plt.show()
