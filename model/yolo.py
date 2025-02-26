# -*- coding: utf-8 -*-

import sys
sys.path.append("..")

import numpy as np

from model.darknet import DarkNet
from model.resnet import resnet_1024ch

import torch
import torch.nn as nn
import torchvision


class yolo(nn.Module):
    def __init__(self, s, cell_out_ch, backbone_name, pretrain=None):
        """
        return: [s, s, cell_out_ch]
        """

        super(yolo, self).__init__()

        self.s = s
        self.backbone = None
        self.conv = None
        if backbone_name == 'darknet':
            self.backbone = DarkNet()
        elif backbone_name == 'resnet':
            self.backbone = resnet_1024ch(pretrained=pretrain)
        self.backbone_name = backbone_name

        assert self.backbone is not None, 'Wrong backbone name'

        self.fc = nn.Sequential(
            nn.Linear(1024 * s * s, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, s * s * cell_out_ch)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = x.view(batch_size, self.s ** 2, -1)
        return x


class yolo_loss:
    def __init__(self, device, s, b, image_size, num_classes):
        self.device = device
        self.s = s
        self.b = b
        self.image_size = image_size
        self.num_classes = num_classes
        self.batch_size = 0

    def __call__(self, input, target):
        """
        :param input: (yolo net output) tensor[s, s, b*5 + n_class] 
                        (c_x, c_y, w, h, obj_conf), (c_x, c_y, w, h, obj_conf), class1_p, class2_p
        :param target: (ground truth)   tensor[s, s, 6]
                        (c_x, c_y, w, h, obj_conf, class_number
        :return: loss tensor
        """
        self.batch_size = input.size(0)

        # conf: [batch * 49 * 2], 分别计算预测框和真实框的iou, 计算出的值会最为 conf 的标签值
        # match: [batch * 49], 标记哪个预测框的iou最大, 为None的表示没有物体的grid cell
        match = []
        conf = []
        for i in range(self.batch_size):
            m, c = self.match_pred_target(input[i], target[i])
            match.append(m)
            conf.append(c)
        
        loss = torch.zeros([self.batch_size], dtype=torch.float, device=self.device)
        xy_loss = torch.zeros_like(loss)
        wh_loss = torch.zeros_like(loss)
        conf_loss = torch.zeros_like(loss)
        class_loss = torch.zeros_like(loss)
        for i in range(self.batch_size):
            loss[i], xy_loss[i], wh_loss[i], conf_loss[i], class_loss[i] = \
                self.compute_loss(input[i], target[i], match[i], conf[i])
        return torch.mean(loss), torch.mean(xy_loss), torch.mean(wh_loss), torch.mean(conf_loss), torch.mean(class_loss)

    def match_pred_target(self, input, target):
        match = []
        with torch.no_grad():
            input_bbox = input[:, :self.b * 5].reshape(-1, self.b, 5)
            ious = [match_get_iou(input_bbox[i], target[i], self.s, i) for i in range(self.s ** 2)]
            for iou in ious:
                if iou is None:
                    match.append(None)
                else:
                    idx = np.argmax(iou)
                    match.append(idx)

        return match, ious

    def compute_loss(self, input, target, match, conf):
        # 计算损失
        ce_loss = nn.CrossEntropyLoss()

        # 49 * 2* 5
        input_bbox = input[:, :self.b * 5].reshape(-1, self.b, 5)
        # 49 * 20
        input_class = input[:, self.b * 5:].reshape(-1, self.num_classes)

        input_bbox = torch.sigmoid(input_bbox)
        loss = torch.zeros([self.s ** 2], dtype=torch.float, device=self.device)
        xy_loss = torch.zeros_like(loss)
        wh_loss = torch.zeros_like(loss)
        conf_loss = torch.zeros_like(loss)
        class_loss = torch.zeros_like(loss)
        # print("\n\n target: ", target)
        # print("\n\n input: ",input)
        # print("\n\n conf: ",conf)
        # print("\n\n match: ",match)
        for i in range(self.s ** 2):
            # 0 xy_loss, 1 wh_loss, 2 conf_loss, 3 class_loss
            l = torch.zeros([4], dtype=torch.float, device=self.device)
            # Neg
            if match[i] == None:
                # λ_noobj = 0.5
                obj_conf_target = torch.zeros([self.b], dtype=torch.float, device=self.device)
                l[2] = torch.sum(torch.mul(0.5, torch.pow(input_bbox[i, :, 4] - obj_conf_target, 2)))
            else:
                # λ_coord = 5
                l[0] = torch.mul(5, torch.pow(input_bbox[i, match[i], 0] - target[i][0], 2) +
                                    torch.pow(input_bbox[i, match[i], 1] - target[i][1], 2) )

                l[1] = torch.mul(5, torch.pow(torch.sqrt(input_bbox[i, match[i], 2]) - torch.sqrt(target[i][2]), 2) +
                                    torch.pow(torch.sqrt(input_bbox[i, match[i], 3]) - torch.sqrt(target[i][3]), 2) )
                
                not_match = 1 - match[i]
                obj_conf_target = torch.zeros([1], dtype=torch.float, device=self.device)

                l[2] = torch.pow(input_bbox[i, match[i], 4] - conf[i][match[i]], 2) + \
                       torch.mul(0.5, torch.pow(input_bbox[i, not_match,  4] - obj_conf_target, 2))

                l[3] = ce_loss(input_class[i], target[i][5].long())

            loss[i] = torch.sum(l)
            xy_loss[i] = l[0]
            wh_loss[i] = l[1]
            conf_loss[i] = l[2]
            class_loss[i] = l[3]
        return torch.sum(loss), torch.sum(xy_loss), torch.sum(wh_loss), torch.sum(conf_loss), torch.sum(class_loss)


def cxcywh2xyxy(bbox):
    """
    :param bbox: [bbox, bbox, ..] tensor c_x(%), c_y(%), w(%), h(%), c
    """
    bbox[:, 0] = bbox[:, 0] - bbox[:, 2] / 2
    bbox[:, 1] = bbox[:, 1] - bbox[:, 3] / 2
    bbox[:, 2] = bbox[:, 0] + bbox[:, 2]
    bbox[:, 3] = bbox[:, 1] + bbox[:, 3]
    return bbox


def match_get_iou(bbox1, bbox2, s, idx):
    """
    :param bbox1: [2*5] tensor c_x(%), c_y(%), w(%), h(%), conf
    :param bbox2: [6] tensor c_x(%), c_y(%), w(%), h(%), conf, class
    :return:
    """

    if bbox1 is None or bbox2[4] == 0:
        return None
    
    bbox2 = bbox2.unsqueeze(dim=0)

    bbox1 = np.array(bbox1.cpu())
    bbox2 = np.array(bbox2.cpu())

    bbox1[:, 0] = bbox1[:, 0] / s
    bbox1[:, 1] = bbox1[:, 1] / s
    bbox2[:, 0] = bbox2[:, 0] / s
    bbox2[:, 1] = bbox2[:, 1] / s

    grid_pos = [(j / s, i / s) for i in range(s) for j in range(s)]
    bbox1[:, 0] = bbox1[:, 0] + grid_pos[idx][0]
    bbox1[:, 1] = bbox1[:, 1] + grid_pos[idx][1]
    bbox2[:, 0] = bbox2[:, 0] + grid_pos[idx][0]
    bbox2[:, 1] = bbox2[:, 1] + grid_pos[idx][1]

    bbox1 = cxcywh2xyxy(bbox1)
    bbox2 = cxcywh2xyxy(bbox2)

    # %
    return get_iou(bbox1, bbox2)


def get_iou(bbox1, bbox2):
    """
    :param bbox1: [bbox, bbox, ..] tensor xmin ymin xmax ymax
    :param bbox2:
    :return: area:
    """

    s1 = abs(bbox1[:, 2] - bbox1[:, 0]) * abs(bbox1[:, 3] - bbox1[:, 1])
    s2 = abs(bbox2[:, 2] - bbox2[:, 0]) * abs(bbox2[:, 3] - bbox2[:, 1])

    ious = []
    for i in range(bbox1.shape[0]):
        xmin = np.maximum(bbox1[i, 0], bbox2[:, 0])
        ymin = np.maximum(bbox1[i, 1], bbox2[:, 1])
        xmax = np.minimum(bbox1[i, 2], bbox2[:, 2])
        ymax = np.minimum(bbox1[i, 3], bbox2[:, 3])

        in_w = np.maximum(xmax - xmin, 0)
        in_h = np.maximum(ymax - ymin, 0)

        in_s = in_w * in_h

        iou = in_s / (s1[i] + s2 - in_s)
        iou = iou.squeeze()
        ious.append(iou)
    ious = np.array(ious)
    return ious


def nms(bbox, conf_th, iou_th):
    bbox = np.array(bbox.cpu())

    bbox[:, 4] = bbox[:, 4] * bbox[:, 5]

    bbox = bbox[bbox[:, 4] > conf_th]
    order = np.argsort(-bbox[:, 4])

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        iou = get_iou(np.array([bbox[i]]), bbox[order[1:]])[0]
        inds = np.where(iou <= iou_th)[0]
        order = order[inds + 1]
    return bbox[keep]


def output_process(output, image_size, s, b, conf_th, iou_th):
    """
    param: output in batch
    :return output: list[], bbox: xmin, ymin, xmax, ymax, obj_conf, classes_conf, classes
    """
    batch_size = output.size(0)
    size = image_size // s

    output = torch.sigmoid(output)

    # Get Class
    classes_conf, classes = torch.max(output[:, :, b * 5:], dim=2)
    classes = classes.unsqueeze(dim=2).repeat(1, 1, 2).unsqueeze(dim=3)
    classes_conf = classes_conf.unsqueeze(dim=2).repeat(1, 1, 2).unsqueeze(dim=3)
    bbox = output[:, :, :b * 5].reshape(batch_size, -1, b, 5)

    bbox = torch.cat([bbox, classes_conf, classes], dim=3)

    # To Direct
    bbox[:, :, :, [0, 1]] = bbox[:, :, :, [0, 1]] * size
    bbox[:, :, :, [2, 3]] = bbox[:, :, :, [2, 3]] * image_size

    grid_pos = [(j * image_size // s, i * image_size // s) for i in range(s) for j in range(s)]

    def to_direct(bbox):
        for i in range(s ** 2):
            bbox[i, :, 0] = bbox[i, :, 0] + grid_pos[i][0]
            bbox[i, :, 1] = bbox[i, :, 1] + grid_pos[i][1]
        return bbox

    bbox_direct = torch.stack([to_direct(b) for b in bbox])
    bbox_direct = bbox_direct.reshape(batch_size, -1, 7)

    # cxcywh to xyxy
    bbox_direct[:, :, 0] = bbox_direct[:, :, 0] - bbox_direct[:, :, 2] / 2
    bbox_direct[:, :, 1] = bbox_direct[:, :, 1] - bbox_direct[:, :, 3] / 2
    bbox_direct[:, :, 2] = bbox_direct[:, :, 0] + bbox_direct[:, :, 2]
    bbox_direct[:, :, 3] = bbox_direct[:, :, 1] + bbox_direct[:, :, 3]

    bbox_direct[:, :, 0] = torch.maximum(bbox_direct[:, :, 0], torch.zeros(1))
    bbox_direct[:, :, 1] = torch.maximum(bbox_direct[:, :, 1], torch.zeros(1))
    bbox_direct[:, :, 2] = torch.minimum(bbox_direct[:, :, 2], torch.tensor([image_size]))
    bbox_direct[:, :, 3] = torch.minimum(bbox_direct[:, :, 3], torch.tensor([image_size]))

    bbox = [torch.tensor(nms(b, conf_th, iou_th)) for b in bbox_direct]
    bbox = torch.stack(bbox)
    return bbox


def collate_fn(batch):
    images, labels, targets = zip(*batch) 
    images = torch.stack(images)
    targets = torch.stack(targets)
    return images, labels, targets


if __name__ == "__main__":
    import torch
    from dataset.draw_bbox import draw
    from dataset.transform import *
    from dataset.data import VOC0712Dataset
    from torch.utils.data import DataLoader

    device = torch.device("mps")

    root0712 = [r'../dataset/VOCdevkit/VOC2007', r'../dataset/VOCdevkit/VOC2012']

    transforms = Compose([
        ToTensor(),
        RandomHorizontalFlip(1),
        Resize(448)
    ])
    ds = VOC0712Dataset(root0712, '../dataset/classes.json', transforms, 2, 7, 448, 'train')
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2, collate_fn=collate_fn)

    iterator = iter(dl)
    images, labels, targets = next(iterator)

    print(images.shape)
    print(len(labels))
    print(targets.shape)

    draw(images[0], labels[0], ds.classes)

    net = yolo(7, 30, 'resnet', pretrain=None)
    net.to(device)
    criterion = yolo_loss("mps", 7, 2, 448, 20)

    images = images.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    outputs = net(images)

    loss = criterion(outputs, targets)
    print(loss)
