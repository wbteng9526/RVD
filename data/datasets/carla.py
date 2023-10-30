import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import torch


class CARLA(Dataset):
    def __init__(self, path, num_of_frame, train=True, transform=None, add_noise=False):
        assert os.path.exists(path), "Invalid path to CITY data set: " + path
        self.path = path
        self.transform = transform
        self.train = train
        if train:
            self.path = os.path.join(path, "training")
            self.seq_list = sorted(os.listdir(self.path))
        else:
            self.path = os.path.join(path, "testing")
            self.seq_list = sorted(os.listdir(self.path))
        self.num_of_frame = num_of_frame
        self.seq_list = sorted(self.seq_list)

    def __getitem__(self, ind):
        # load the images from the ind directory to get list of PIL images
        seq = self.seq_list[idx]

        control_frame_dir = os.path.join(self.path, seq, "source", "image")
        target_frame_dir = os.path.join(self.path, seq, "target", "image"))

        control_frame_list = sorted(os.listdir(control_frame_dir))
        target_frame_list = sorted(os.listdir(target_frame_dir))

        assert len(control_frame_list) == len(target_frame_list)
        seq_length = len(control_frame_list)

        if self.train:
            start_ind = torch.randint(0, seq_length - 1 - self.num_of_frame, (1,)).item()
        else:
            start_ind = 0

        control_imgs = [Image.open(os.path.join(control_frame_dir, control_frame_list[start_ind + i])) for i in range(self.num_of_frame)]
        target_imgs = [Image.open(os.path.join(target_frame_dir, target_frame_list[start_ind + i])) for i in range(self.num_of_frame)]

        if self.transform is not None:
            control_imgs = self.transform(control_imgs)
            target_imgs = self.transform(target_imgs)

        if self.add_noise:
            target_imgs = target_imgs + (torch.rand_like(target_imgs) - 0.5) / 256.0

        return dict(jpg=control_imgs, hint=target_imgs)

    def __len__(self):
        # total number of videos
        return len(self.seq_list)
