import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, root_dir, transforms = None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.image_files = [file for file in os.listdir(root_dir) if file.endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.root_dir, image_name)
        mask_path = os.path.join(self.root_dir, 'Mask', image_name)

        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
        mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2GRAY)

        if self.transforms:
            image = self.transforms(image)
            mask = self.transforms(mask)

        return image,mask

class img_helper:
    def genreate_mask(path: str):
        for file in os.listdir(path):
            if file.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                image = cv2.imread(os.path.join(path,file))
                image_GRAY = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                mask = np.where(image_GRAY>225, image_GRAY, 0).astype('uint8')
                os.makedirs(os.path.join(path,'Mask'), exist_ok=True)
                output_path = os.path.join(path,'Mask',file)
                cv2.imwrite(output_path,mask)

    def show_images(data: list, num_images = None):
        if num_images==None:
            num_images = len(data[0])
        fig, axes = plt.subplots(2, num_images, figsize=(5, 5))
        image , masks = data
        for i in range(num_images): 
            axes[0,i].imshow(image[i].squeeze(),cmap="gray")
            axes[0,i].set_title('Grayscale Image')
            axes[0,i].axis('off')

            axes[1,i].imshow(masks[i].squeeze(), cmap="gray")
            axes[1,i].set_title('Mask')
            axes[1,i].axis('off')

        plt.show()