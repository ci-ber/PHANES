"""
DataLoader.py (pl.LightningDataModule)

Class for brain MRI data loaders from:
- csv file
"""

from core.DataLoader import DefaultDataset
import torchvision.transforms as transforms
from transforms.preprocessing import *


class BrainLoader(DefaultDataset):
    def __init__(self, data_dir, file_type='', label_dir=None, mask_dir=None, target_size=(256, 256), test=False):
        self.target_size = target_size
        self.RES = transforms.Resize(self.target_size)
        super(BrainLoader, self).__init__(data_dir, file_type, label_dir, mask_dir, target_size, test)

    def get_image_transform(self):
        default_t = transforms.Compose([ReadImage(), To01()
                                        ,Pad((1, 1))
                                        ,AddChannelIfNeeded()
                                        ,AssertChannelFirst()
                                        ,self.RES
                                        ,transforms.ToPILImage(), transforms.RandomAffine(10, (0.1, 0.1), (0.9, 1.1)),
                                        transforms.RandomHorizontalFlip(0.5),
                                        transforms.ToTensor()
                                        ])
        return default_t

    def get_image_transform_test(self):
        default_t_test = transforms.Compose([ReadImage(), To01()
                                        ,Pad((1, 1))
                                        ,AddChannelIfNeeded()
                                        ,AssertChannelFirst(), self.RES
                                        ])
        return default_t_test

    def get_label_transform(self):
        default_t_label = transforms.Compose([ReadImage(),  To01()
                                             ,Pad((1, 1))
                                             ,AddChannelIfNeeded()
                                             ,AssertChannelFirst()
                                             ,self.RES
                                            ])
        return default_t_label