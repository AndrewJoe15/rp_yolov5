import xml.etree.ElementTree as ET
import os
import random
from shutil import copyfile, rmtree
from utils.general import Path, check_dataset
import argparse


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory

def convert(size, box):
    dw, dh = 1. / size[0], 1. / size[1]
    x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh


def convert_annotation(xml_dir, txt_dir, image_id, names):
    if not os.path.isfile(xml_dir + '%s.xml' % image_id):
        return False
    in_file = open(xml_dir + '%s.xml' % image_id, encoding='UTF-8')
    out_file = open(txt_dir + '%s.txt' % image_id, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in names or int(difficult) == 1:
            continue
        cls_id = names.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    in_file.close()
    out_file.close()
    return True


# 保证文件夹存在且是空的
def check_dir(file_dir):
    if os.path.isdir(file_dir):
        rmtree(file_dir)
    os.mkdir(file_dir)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='', help='dataset folder name')
    parser.add_argument('--ratio', type=int, default=80, help='trainSet/dataset percent 1-100')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt):
    # 数据集文件夹
    dataset_base_dir = opt.name
    # 训练集划分比例
    train_ratio = opt.ratio
    # 类别数组
    data_yaml = opt.name + '.yaml'
    data_yaml = ROOT / '../data' / data_yaml
    names = []
    cls_dirt = check_dataset(data_yaml)['names']
    for i in range(len(cls_dirt)):
        names.append(cls_dirt[i])

    # 数据集路径
    data_base_dir = "../datasets/"
    # data_base_dir = Path(yaml['path'])
    # 项目数据集路径
    work_sapce_dir = os.path.join(data_base_dir, dataset_base_dir)
    annotation_dir = os.path.join(work_sapce_dir, "Annotations/")
    image_dir = os.path.join(work_sapce_dir, "Images/")
    yolo_labels_dir = os.path.join(work_sapce_dir, "Labels/")
    # 训练/验证集路径
    # 图片
    yolov5_images_dir = os.path.join(data_base_dir, "images/")
    yolov5_images_train_dir = os.path.join(yolov5_images_dir, "train/")
    yolov5_images_val_dir = os.path.join(yolov5_images_dir, "val/")
    # 标签
    yolov5_labels_dir = os.path.join(data_base_dir, "labels/")
    yolov5_labels_train_dir = os.path.join(yolov5_labels_dir, "train/")
    yolov5_labels_val_dir = os.path.join(yolov5_labels_dir, "val/")

    # 检查训练数据文件目录，有则清空，则没有则创建
    check_dir(yolov5_images_dir)
    check_dir(yolov5_labels_dir)
    check_dir(yolov5_images_train_dir)
    check_dir(yolov5_images_val_dir)
    check_dir(yolov5_labels_train_dir)
    check_dir(yolov5_labels_val_dir)

    list_imgs = os.listdir(image_dir)  # 图片文件列表

    sum_train = sum_val = 0

    for i in range(0, len(list_imgs)):
        path = os.path.join(image_dir, list_imgs[i])
        if os.path.isfile(path):
            image_path = image_dir + list_imgs[i]
            image_name = list_imgs[i]
            (nameWithoutExtention, extention) = os.path.splitext(os.path.basename(image_path))
            annotation_name = nameWithoutExtention + '.xml'
            annotation_path = os.path.join(annotation_dir, annotation_name)
            label_name = nameWithoutExtention + '.txt'
            label_path = os.path.join(yolo_labels_dir, label_name)
        # 转换xml格式为yolo格式标签
        if convert_annotation(annotation_dir, yolo_labels_dir, nameWithoutExtention, names):
            # 放到训练集或验证集
            prob = random.randint(1, 100)  # 用来划分数据集的随机数
            if prob < train_ratio:  # train dataset
                if os.path.exists(annotation_path):
                    sum_train += 1
                    copyfile(image_path, yolov5_images_train_dir + image_name)
                    copyfile(label_path, yolov5_labels_train_dir + label_name)
            else:  # val dataset
                if os.path.exists(annotation_path):
                    sum_val += 1
                    copyfile(image_path, yolov5_images_val_dir + image_name)
                    copyfile(label_path, yolov5_labels_val_dir + label_name)
    print('标注文件转换与数据集划分完成，训练集数量：' + sum_train.__str__() + ', 验证集数量：' + sum_val.__str__())


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
