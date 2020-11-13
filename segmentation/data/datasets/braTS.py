# ------------------------------------------------------------------------------
# Loads BraTS panoptic dataset.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import json
import os
from glob import glob
import SimpleITK as sitk
import xml.dom.minidom
import numpy as np
from PIL import Image, ImageOps
import torch
import xmltodict

from .base_dataset import BaseDataset
from .utils import DatasetDescriptor
from ..transforms import build_transforms, PanopticTargetGenerator, SemanticTargetGenerator

_BraTS_INFORMATION = DatasetDescriptor(
    splits_to_sizes={'train': 19855,
                     'test': 30},
    num_classes=5,
    ignore_label=255,
)

# A map from data type to folder name that saves the data.
_FOLDERS_MAP = {
    'image': 'leftImg8bit',
    'label': 'gtFine',
}
# A map from data type to filename postfix.
_POSTFIX_MAP = {
    'image': '_leftImg8bit',
    'label': '_gtFine_labelTrainIds',
}

# A map from data type to data format.
_DATA_FORMAT_MAP = {
    'image': 'png',
    'label': 'png',
}

# BRAT_CATEGORIES = [
#     {"color": [0, 0, 0], "isthing": 0, "id": 0, "name": "background"},
#     {"color": [119, 11, 32], "isthing": 1, "id": 1, "name": "Necrosis"},
#     {"color": [0, 0, 142],   "isthing": 1, "id": 2, "name": "Edema"},
#     {"color": [0, 0, 230],   "isthing": 1, "id": 3, "name": "Non-enhancing tumor"},
#     {"color": [106, 0, 228], "isthing": 1, "id": 4, "name": "Enhancing tumor"},
# ]

# Add 1 void label.
_BraTS_PANOPTIC_TRAIN_ID_TO_EVAL_ID = [1, 2, 3, 4, 0]

_BraTS_THING_LIST = [255]


class BraTS(BaseDataset):
    """
    BraTS panoptic segmentation dataset.
    Arguments:
        root: Str, root directory.
        split: Str, data split, e.g. train/val/test.
        is_train: Bool, for training or testing.
        crop_size: Tuple, crop size.
        mirror: Bool, whether to apply random horizontal flip.
        min_scale: Float, min scale in scale augmentation.
        max_scale: Float, max scale in scale augmentation.
        scale_step_size: Float, step size to select random scale.
        mean: Tuple, image mean.
        std: Tuple, image std.
        semantic_only: Bool, only use semantic segmentation label.
        ignore_stuff_in_offset: Boolean, whether to ignore stuff region when training the offset branch.
        small_instance_area: Integer, indicates largest area for small instances.
        small_instance_weight: Integer, indicates semantic loss weights for small instances.
    """
    def __init__(self,
                 root,
                 split,
                 is_train=True,
                 crop_size=(240, 240),
                 mirror=True,
                 min_scale=0.5,
                 max_scale=2.,
                 scale_step_size=0.25,
                 mean=(0.485, 0.456, 0.406, 0.435),
                 std=(0.229, 0.224, 0.225, 0.226),
                 semantic_only=False,
                 ignore_stuff_in_offset=False,
                 small_instance_area=0,
                 small_instance_weight=1,
                 **kwargs):
        super(BraTS, self).__init__(root, split, is_train, crop_size, mirror, min_scale, max_scale,
                                                 scale_step_size, mean, std)

        self.num_classes = _BraTS_INFORMATION.num_classes
        self.ignore_label = _BraTS_INFORMATION.ignore_label
        self.label_pad_value = (0, 0, 0)

        self.has_instance = True
        self.label_divisor = 1000
        self.label_dtype = 'float32'
        self.thing_list = _BraTS_THING_LIST

        # Get image and annotation list.
        if split == 'test':
            self.img_list = self._get_files('image', self.split)
            self.ann_list = None
            self.ins_list = None
        else:
            self.T1_path = []
            self.T1c_path = []
            self.Flair_path = []
            self.T2_path = []
            self.OT_path = []
            # self.img_list = []
            self.ann_list = []
            self.ins_list = []

            dPath = r'/media/uestcer/Projects/WDY/datasets/Brats/BRATS2015_slice/'
            for LorH in os.listdir(dPath+'Training'):
                path = os.path.join(dPath+'Training', LorH)
                # for filename in os.listdir(path):
                #     braPath = os.path.join(path, filename)
                    # self.img_list.append(braPath)
                self.T1_path.append(glob(path + '/*/*T1.*/*.gz'))
                self.T1c_path.append(glob(path + '/*/*T1c.*/*.gz'))
                self.Flair_path.append(glob(path + '/*/*Flair*/*.gz'))
                self.T2_path.append(glob(path + '/*/*T2*/*.gz'))
                self.OT_path.append(glob(path + '/*/*OT*/*.gz'))
            self.T1_path = self.T1_path[0] + self.T1_path[1]
            self.T1c_path = self.T1c_path[0] + self.T1c_path[1]
            self.Flair_path = self.Flair_path[0] + self.Flair_path[1]
            self.T2_path = self.T2_path[0] + self.T2_path[1]
            self.ann_list = self.OT_path[0] + self.OT_path[1]

            for label in os.listdir(dPath+'label'):
                path = os.path.join(dPath+'label', label)
                for i in os.listdir(path):
                    file = glob(os.path.join(path, i) + '/*.xml')[0]
                    annotation = self.xml2json(file)
                    # dom = xml.dom.minidom.parse(file)
                    # root = dom.documentElement
                    # annotation = root.getElementsByTagName('image')
                    for i in annotation['annotation']['image']:
                        if i['segmented'] is not '0':
                            bndboxs = i['bndbox']
                            bbox_list = []
                            if isinstance(bndboxs, list):
                                for bndbox in bndboxs:
                                    xmin = int(bndbox['xmin']) - 1
                                    ymin = int(bndbox['ymin']) - 1
                                    xmax = int(bndbox['xmax'])
                                    ymax = int(bndbox['ymax'])
                                    o_width = abs(xmax - xmin)
                                    o_height = abs(ymax - ymin)
                                    bbox = {"area": o_width * o_height,
                                            'bbox': [xmin, ymin, o_width, o_height],
                                            'id': 1,
                                            'category_id': 1,
                                            'iscrowd': 0,
                                            "segmentation": []}
                                    bbox_list.append(bbox)
                                self.ins_list.append(bbox_list)
                            else:
                                o_width = abs(int(bndboxs['xmax']) - int(bndboxs['xmin']))
                                o_height = abs(int(bndboxs['ymax']) - int(bndboxs['ymin']))
                                bbox = {"area": o_width * o_height,
                                        'bbox': [int(bndboxs['xmin']), int(bndboxs['ymin']), o_width, o_height],
                                        'id': 1,
                                        'category_id': 1,
                                        'iscrowd': 0,
                                        "segmentation": []}
                                bbox_list.append(bbox)
                                self.ins_list.append(bbox_list)

        assert len(self) == _BraTS_INFORMATION.splits_to_sizes[self.split]
        self.transform = build_transforms(self, is_train)
        if semantic_only:
            self.target_transform = SemanticTargetGenerator(self.ignore_label, self.rgb2id)
        else:
            self.target_transform = PanopticTargetGenerator(self.ignore_label, self.rgb2id, _BraTS_THING_LIST,
                                                            sigma=8, ignore_stuff_in_offset=ignore_stuff_in_offset,
                                                            small_instance_area=small_instance_area,
                                                            small_instance_weight=small_instance_weight)

        # Generates semantic label for evaluation.
        self.raw_label_transform = SemanticTargetGenerator(self.ignore_label, self.rgb2id)

    @staticmethod
    def train_id_to_eval_id():
        return _BraTS_PANOPTIC_TRAIN_ID_TO_EVAL_ID

    def __getitem__(self, index):
        # TODO: handle transform properly when there is no label
        dataset_dict = {}
        assert os.path.exists(self.ann_list[index]), 'Path does not exist: {}'.format(self.img_list[index])
        arr_t1 = self.read_image(self.T1_path[index], self.label_dtype)
        arr_t1c = self.read_image(self.T1c_path[index], self.label_dtype)
        arr_flair = self.read_image(self.Flair_path[index], self.label_dtype)
        arr_t2 = self.read_image(self.T2_path[index], self.label_dtype)
        image = np.concatenate((arr_flair, arr_t1, arr_t1c, arr_t2), axis=-1)
        if not self.is_train:
            # Do not save this during training.
            dataset_dict['raw_image'] = image.copy()
        if self.ann_list is not None:
            assert os.path.exists(self.ann_list[index]), 'Path does not exist: {}'.format(self.ann_list[index])
            label = self.read_label(self.ann_list[index], self.label_dtype)
        else:
            label = None
        raw_label = label.copy()
        if self.raw_label_transform is not None:
            raw_label = self.raw_label_transform(raw_label, self.ins_list[index])['semantic']
        if not self.is_train:
            # Do not save this during training
            dataset_dict['raw_label'] = raw_label
        size = image.shape
        dataset_dict['raw_size'] = np.array(size)
        # To save prediction for official evaluation.
        # name = os.path.splitext(os.path.basename(self.ann_list[index]))[0]
        # TODO: how to return the filename?
        # dataset_dict['name'] = np.array(name)

        # Resize and pad image to the same size before data augmentation.
        if self.pre_augmentation_transform is not None:
            image, label = self.pre_augmentation_transform(image, label)
            size = image.shape
            dataset_dict['size'] = np.array(size)
        else:
            dataset_dict['size'] = dataset_dict['raw_size']

        # 这个时候:
        # label: shape = (240, 240, 1), dtype = float32
        # image: shape = (240, 240, 4), dtype = float32
        # print(image[80,:,:])
        # print(image.dtype)
        # 暂时注释掉,输出应该是[240,240,3],但是此处输出为[240,240]
        # Apply data augmentation.
        if self.transform is not None:
            image, label = self.transform(image, label)
            # 手动增加一维
            label = np.expand_dims(label, -1)
        # 这个时候:
        # label: shape = (240, 240), dtype = float32
        # image: shape = torch.Size([4, 240, 240]), dtype = torch.float32
        # print(image.dtype)
        dataset_dict['image'] = image
        if not self.has_instance:
            dataset_dict['semantic'] = torch.as_tensor(label.astype('long'))
            return dataset_dict

        # Generate training target.
        if self.target_transform is not None:
            # print(label.shape)
            label_dict = self.target_transform(label, self.ins_list[index])
            for key in label_dict.keys():
                dataset_dict[key] = label_dict[key]

        return dataset_dict

    # import torchsnooper
    @staticmethod
    # @torchsnooper.snoop()
    def rgb2id(color):
        """Converts the color to panoptic label.
        Color is created by `color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]`.
        Args:
            color: Ndarray or a tuple, color encoded image.
        Returns:
            Panoptic label.
        """
        if isinstance(color, np.ndarray) and len(color.shape) == 3:
            if color.dtype == np.uint8:
                color = color.astype(np.int32)
            a = color[:, :, 0] + 256 * color[:, :, 0] + 256 * 256 * color[:, :, 0]
            # print(color.shape)
            return a
        return int(color[0] + 256 * color[0] + 256 * 256 * color[0])

    def __len__(self):
        return len(self.ann_list)

    @staticmethod
    def read_image(file_name, format=None):
        mha = sitk.ReadImage(file_name)
        image = sitk.GetArrayFromImage(mha)
        mask = image > 0
        temp_img = image[image > 0]
        img_array = (image - temp_img.mean()) / temp_img.std() * mask
        img_array = np.expand_dims(img_array, -1)
        # img_array = np.asarray(img_array)
        return img_array.astype(format)

    @staticmethod
    def read_label(file_name, dtype='uint8'):
        # In some cases, `uint8` is not enough for label
        mha = sitk.ReadImage(file_name)
        image = sitk.GetArrayFromImage(mha)
        image = np.expand_dims(image, -1)
        return np.asarray(image, dtype=dtype)

    def _get_files(self, data, dataset_split):
        """Gets files for the specified data type and dataset split.
        Args:
            data: String, desired data ('image' or 'label').
            dataset_split: String, dataset split ('train', 'val', 'test')
        Returns:
            A list of sorted file names or None when getting label for test set.
        """
        if data == 'label' and dataset_split == 'test':
            return None
        pattern = '*%s.%s' % (_POSTFIX_MAP[data], _DATA_FORMAT_MAP[data])
        search_files = os.path.join(
            self.root, _FOLDERS_MAP[data], dataset_split, '*', pattern)
        # search_files = '/media/uestcer/Projects/W、DY/projects/panoptic-deeplab/datasets/cityscapes/leftImg8bit/train/*/*_leftImg8bit.png'
        filenames = glob(search_files)
        return sorted(filenames)

    @staticmethod
    def xml2json(xml_path):
        xml_file = open(xml_path, 'r')
        xml_str = xml_file.read()
        json = xmltodict.parse(xml_str)
        return json

    @staticmethod
    def create_label_colormap():
        """Creates a label colormap used in CITYSCAPES segmentation benchmark.
        Returns:
            A colormap for visualizing segmentation results.
        """
        colormap = np.zeros((256, 3), dtype=np.uint8)
        colormap[0] = [128, 64, 128]
        colormap[1] = [244, 35, 232]
        colormap[2] = [70, 70, 70]
        colormap[3] = [102, 102, 156]
        colormap[4] = [190, 153, 153]
        colormap[5] = [153, 153, 153]
        colormap[6] = [250, 170, 30]
        colormap[7] = [220, 220, 0]
        colormap[8] = [107, 142, 35]
        colormap[9] = [152, 251, 152]
        colormap[10] = [70, 130, 180]
        colormap[11] = [220, 20, 60]
        colormap[12] = [255, 0, 0]
        colormap[13] = [0, 0, 142]
        colormap[14] = [0, 0, 70]
        colormap[15] = [0, 60, 100]
        colormap[16] = [0, 80, 100]
        colormap[17] = [0, 0, 230]
        colormap[18] = [119, 11, 32]
        return colormap
