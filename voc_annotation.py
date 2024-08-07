import xml.etree.ElementTree as ET
import numpy as np

# ----------------------------------------------VOC数据集中所有类-----------------------------------------------------------#
classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

nums = np.zeros(len(classes))


def extract_annotation(year, image_id, list_file):  # 提取xml中的信息
    in_file = open('VOC%s/Annotations/%s.xml' % (year, image_id), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    filename = root.find('filename').text  # 图片名
    for obj in root.iter('object'):
        cls = obj.find('name').text  # 类别
        difficult = 0
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text
        if cls not in classes or int(difficult) == 1:  # 不属于classes中的类或者检测难度较大的则舍弃
            continue
        cls_id = classes.index(cls)  # 类别转成数值id
        xmlbox = obj.find('bndbox')  # 坐标信息
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        filename = filename + " " + ",".join([str(a) for a in b]) + ',' + str(cls_id)
    list_file.write(filename + '\n')  # 写入信息


if __name__ == "__main__":
    voc_year = 2007  # 下载的VOC数据集版本 以VOC2007为例
    with open('VOC%s/ImageSets/Main/train.txt' % (voc_year)) as f:  # 获得训练集的所有样本名
        train_names = f.readlines()
        train_names = [line.rstrip('\n') for line in train_names]

    with open('VOC%s/train.txt' % (voc_year), 'w') as g:  # 提取信息
        for name in train_names:
            extract_annotation(voc_year, name, g)

    with open('VOC%s/ImageSets/Main/val.txt' % (voc_year)) as f:  # 获得验证集的所有样本名
        val_names = f.readlines()
        val_names = [line.rstrip('\n') for line in val_names]

    with open('VOC%s/val.txt' % (voc_year), 'w') as g:
        for name in val_names:
            extract_annotation(voc_year, name, g)

    with open('VOC%s/ImageSets/Main/test.txt' % (voc_year)) as f:  # 获得测试集的所有样本名
        test_names = f.readlines()
        test_names = [line.rstrip('\n') for line in test_names]

    with open('VOC%s/test.txt' % (voc_year), 'w') as g:
        for name in test_names:
            extract_annotation(voc_year, name, g)
