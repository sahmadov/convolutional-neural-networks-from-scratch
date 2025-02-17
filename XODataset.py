from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
import torch

class XODataset(Dataset):
    def __init__(self, data_dir, labels_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.labels = pd.read_csv(labels_file, sep=" ", header=None, names=["filename", "label"])
        self.labels["label"] = self.labels["label"].map({"X": 1, "O": 0})  # Convert labels to integers

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.labels.iloc[idx, 0])
        label = self.labels.iloc[idx, 1]

        # Load image and convert to grayscale (1 channel)
        image = Image.open(img_name).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)