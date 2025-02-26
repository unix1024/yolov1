# import sys
# sys.path.append("..")

from dataset.transform import *

from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import json
import os


def get_file_name(root, layout_txt):
    with open(os.path.join(root, layout_txt)) as layout_txt:
        file_name = layout_txt.read().split('\n')[:-1]
    return file_name


def xml2dict(xml):
    data = {c.tag: None for c in xml}
    for c in xml:
        def add(data, tag, text):
            if data[tag] is None:
                data[tag] = text
            elif isinstance(data[tag], list):
                data[tag].append(text)
            else:
                data[tag] = [data[tag], text]
            return data

        if len(c) == 0:
            data = add(data, c.tag, c.text)
        else:
            data = add(data, c.tag, xml2dict(c))
    return data


class VOC0712Dataset(Dataset):
    def __init__(self, root, class_path, transforms, B, S, image_size, mode, data_range=None, get_info=False):
        # label: xmin, ymin, xmax, ymax, class

        with open(class_path, 'r') as f:
            json_str = f.read()
            self.classes = json.loads(json_str)
        layout_txt = None
        if mode == 'train':
            root = [root[0], root[0], root[1], root[1]]
            layout_txt = [r'ImageSets/Main/train.txt', r'ImageSets/Main/val.txt',
                          r'ImageSets/Main/train.txt', r'ImageSets/Main/val.txt']
        elif mode == 'test':
            if not isinstance(root, list):
                root = [root]
            layout_txt = [r'ImageSets/Main/test.txt']
        assert layout_txt is not None, 'Unknown mode'

        self.transforms = transforms
        self.get_info = get_info
        self.b = B
        self.s = S
        self.num_class = len(self.classes)
        self.image_size = image_size

        self.image_list = []
        self.annotation_list = []
        for r, txt in zip(root, layout_txt):
            self.image_list += [os.path.join(r, 'JPEGImages', t + '.jpg') for t in get_file_name(r, txt)]
            self.annotation_list += [os.path.join(r, 'Annotations', t + '.xml') for t in get_file_name(r, txt)]

        if data_range is not None:
            self.image_list = self.image_list[data_range[0]: data_range[1]]
            self.annotation_list = self.annotation_list[data_range[0]: data_range[1]]

    def __len__(self):
        return len(self.annotation_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx])
        org_image_size = image.size
        label = self.label_process(self.annotation_list[idx])

        if self.transforms is not None:
            image, label = self.transforms(image, label)

        # label [n * 5] ->  target [49 * 6]
        target = self.label_direct2grid(label)

        if self.get_info:
            return image, label, target, os.path.basename(self.image_list[idx]).split('.')[0], org_image_size
        else:
            return image, label, target

    def label_process(self, annotation):
        xml = ET.parse(os.path.join(annotation)).getroot()
        data = xml2dict(xml)['object']
        if isinstance(data, list):
            label = [[float(d['bndbox']['xmin']), float(d['bndbox']['ymin']),
                     float(d['bndbox']['xmax']), float(d['bndbox']['ymax']),
                     self.classes[d['name']]]
                     for d in data]
        else:
            label = [[float(data['bndbox']['xmin']), float(data['bndbox']['ymin']),
                     float(data['bndbox']['xmax']), float(data['bndbox']['ymax']),
                     self.classes[data['name']]]]
        label = np.array(label)
        return label

    def label_direct2grid(self, label):
        """
        :param label: dataset type: (pixel unit) [ n_bbox * 5 ]
                                    xmin, ymin, xmax, ymax, class
        :return: label: grid type: (% unit) [ s^2 * 6 ]
                                    cx, cy, w, h, conf, class
        """
        output = torch.zeros(self.s ** 2, 6)
        size = self.image_size // self.s  # 一个grid cell的尺寸

        n_bbox = label.size(0)
        label_c = torch.zeros_like(label)

        # 把框转换成中心点和宽高，单位是像素
        label_c[:, 0] = (label[:, 0] + label[:, 2]) / 2
        label_c[:, 1] = (label[:, 1] + label[:, 3]) / 2
        label_c[:, 2] = abs(label[:, 0] - label[:, 2])
        label_c[:, 3] = abs(label[:, 1] - label[:, 3])
        label_c[:, 4] = label[:, 4]

        # 计算框所在的grid cell
        idx_x = [int(label_c[i][0]) // size for i in range(n_bbox)]
        idx_y = [int(label_c[i][1]) // size for i in range(n_bbox)]

        # 单位转换成百分比
        label_c[:, 0] = torch.div(torch.fmod(label_c[:, 0], size), size)
        label_c[:, 1] = torch.div(torch.fmod(label_c[:, 1], size), size)
        label_c[:, 2] = torch.div(label_c[:, 2], self.image_size)
        label_c[:, 3] = torch.div(label_c[:, 3], self.image_size)

        for i in range(n_bbox):
            idx = idx_y[i] * self.s + idx_x[i]
            # 如果置信度为0，就把框信息保存到grid cell（只保存一个框）
            if output[idx][4] == 0:
                output[idx][:4] = label_c[i][:4]    # 把转换后的坐标保存到对应的grid cell
                output[idx][4] = 1                  # 设置置信度为1
                output[idx][5] = label_c[i][4]      # 保存类别信息

        return output

if __name__ == "__main__":
    from dataset.draw_bbox import draw

    root0712 = [r'VOCdevkit/VOC2007', r'VOCdevkit/VOC2012']

    transforms = Compose([
        ToTensor(),
        RandomHorizontalFlip(1),
        Resize(448)
    ])
    ds = VOC0712Dataset(root0712, 'classes.json', transforms, 2, 7, 448, 'train', get_info=True)
    print(len(ds))

    for i, (image, label, target, org_image_name, org_image_size) in enumerate(ds):


        print(image)
        print(label)
        print(target)
        draw(image, label, ds.classes)

        if i ==5:
            break

        # if i <= 1000:
        #     continue
        # elif i >= 1010:
        #     break
        # else:
        #     print(label.dtype)
        #     print(tuple(image.size()[1:]))
        #     draw(image, label, ds.classes)

    # print('VOC2007Dataset')


