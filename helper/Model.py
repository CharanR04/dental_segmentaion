import torch
import torch.nn as nn
from tqdm import tqdm

class Conv_block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False, kernel_size: int = 2, padding: int=1, stride:int = 1):
        super(Conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels= in_channels, out_channels= out_channels, kernel_size=kernel_size, stride = stride, padding=padding, bias = bias)
        #self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU(inplace=True)
        self.b_norm = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(2,2)
    def forward(self,x):
        x = self.conv(x)
        #x = self.dropout(x)
        x = self.relu(x)
        x = self.b_norm(x)
        x = self.maxpool(x)
        return x

class Deconv_block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size :int = 2, stride: int = 2, padding: int = 0):
        super(Deconv_block, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        #self.dropout = nn.Dropout(0.2)
        self.b_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.deconv(x)
        #x = self.dropout(x)
        x = self.b_norm(x)
        x = self.relu(x)
        return x
    
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block_input = Conv_block(in_channels= 1,out_channels= 4)
        self.conv_block_1 = Conv_block(in_channels= 4,out_channels= 32)
        self.conv_block_2 = Conv_block(in_channels= 32,out_channels= 128)
        self.conv_block_output = Conv_block(in_channels=128,out_channels=512)

        self.deconv_block_input = Deconv_block(out_channels=128,in_channels=512)
        self.deconv_block_1 = Deconv_block(out_channels = 32,in_channels=128)
        self.deconv_block_2 = Deconv_block(out_channels=4,in_channels=32)
        self.deconv_block_output = Deconv_block(out_channels=1,in_channels=4)
    
    def forward(self,x):
        x_input = self.conv_block_input(x)
        x_1 = self.conv_block_1(x_input)
        x_2 = self.conv_block_2(x_1)
        x_output = self.conv_block_output(x_2)

        x_deconv_input = self.deconv_block_input(x_output)
        
        x_deconv_1 = x_deconv_input + x_2
        x_deconv_1 = self.deconv_block_1(x_deconv_1)
        
        x_deconv_2 = x_deconv_1 + x_1
        x_deconv_2 = self.deconv_block_2(x_deconv_2)
        
        x_output = x_deconv_2 + x_input
        x_output = self.deconv_block_output(x_output)

        return x_output
    
    def train_model(self, epoch: int, dataloader, optimizers, loss_fn):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loss_history = []
        val_loss_history = []
        for e in range(epoch):
            print(f'{e}/{epoch}:', end = ' ')
            self.train()
            self.to(device)
            train_loss = 0.0

            for i, (imgs,masks) in tqdm(enumerate(dataloader), total = len(dataloader), ncols = 60):
                optimizers[0].zero_grad()
                optimizers[1].zero_grad()
                imgs = imgs.to(device)
                masks = masks.to(device)
                outputs = self(imgs)
                loss = loss_fn(outputs,masks)
                loss.backward()
                optimizers[0].step()
                optimizers[1].step()

                train_loss += loss.item()
            
            train_loss /= len(dataloader)
            train_loss_history.append(train_loss)
            self.eval()

            valid_loss = 0.0

            with torch.no_grad():
                for i, (img, mask) in tqdm(enumerate(dataloader), total = len(dataloader), ncols=60):
                    img = img.to(device)
                    mask = mask.to(device)
                    output = self(img)
                    loss = loss_fn(output,mask)
                    valid_loss += loss.item()
                valid_loss /= len(dataloader)
            val_loss_history.append(valid_loss)
        return train_loss_history,val_loss_history