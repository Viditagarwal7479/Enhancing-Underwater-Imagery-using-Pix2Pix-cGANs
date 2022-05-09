import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import config


class MapDataset(Dataset):
    def __init__(self, root_dir, val=False):
        self.root_dir = root_dir
        self.imgs = os.listdir(os.path.join(root_dir, "trainA/"))
        self.targets = os.listdir(os.path.join(root_dir, "trainB/"))
        self.val = val

    def __len__(self):
        if not self.val:
            return (len(self.imgs) * 4) // 5
        else:
            return len(self.imgs) - ((len(self.imgs) * 4) // 5)

    def __getitem__(self, index):
        if not self.val:
            img_path = os.path.join(os.path.join(self.root_dir, "trainA/"), self.imgs[index])
            targ_path = os.path.join(os.path.join(self.root_dir, "trainB/"), self.targets[index])
        else:
            img_path = os.path.join(os.path.join(self.root_dir, "trainA/"),
                                    self.imgs[((len(self.imgs) * 4) // 5) + index])
            targ_path = os.path.join(os.path.join(self.root_dir, "trainB/"),
                                     self.targets[((len(self.imgs) * 4) // 5) + index])
        input_image = np.array(Image.open(img_path))
        target_image = np.array(Image.open(targ_path))

        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image
