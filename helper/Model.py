import torch
import torch.nn as nn

class Conv_block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False, kernel_size: int = 3, padding: int=0):
        super(Conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels= in_channels, out_channels= out_channels, kernel_size=kernel_size, padding=padding, bias = bias)
        self.relu = nn.ReLU(inplace=True)
        self.b_norm = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(2,2)
    def forward(self,x):
         x = self.conv(x)
         x = self.relu(x)
         x = self.b_norm(x)
         x = self.maxpool(x)
         return x

class Deconv_block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size :int = 3, stride: int = 1, padding: int = 0):
        super(Deconv_block, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.b_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.deconv(x)
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
        self.deconv_block_2 = Deconv_block(out_channels=4,in_channels=32,padding=1)
        self.deconv_block_output = Deconv_block(out_channels=1,in_channels=4)
    
    def forward(self,x):
        x_input = self.conv_block_input(x)
        x_1 = self.conv_block_1(x_input)
        x_2 = self.conv_block_2(x_1)
        x_output = self.conv_block_output(x_2)
        
        x_deconv_input = self.deconv_block_input(x_output)
        x_deconv_input = torch.cat((x_deconv_input, x_2), dim=1)
        
        x_deconv_1 = self.deconv_block_1(x_deconv_input)
        x_deconv_1 = torch.cat((x_deconv_1, x_1), dim=1)
        
        x_deconv_2 = self.deconv_block_2(x_deconv_1)
        x_deconv_2 = torch.cat((x_deconv_2, x_input), dim=1)
        
        x_output = self.deconv_block_output(x_deconv_2)

        return x_output

class Train:
    def __init__(self):
        return

"""
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block_input = Conv_block(in_channels= 1,out_channels= 4)
        self.conv_block_1 = Conv_block(in_channels= 4,out_channels= 32)
        self.conv_block_2 = Conv_block(in_channels= 32,out_channels= 128)
        self.conv_block_output = Conv_block(in_channels=128,out_channels=512)
    def forward(self,x):
        x = self.conv_block_input(x)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_output(x)
        return x
       
class Decoder(nn.Module):
    def __init__(self):
        self.deconv_block_input = Deconv_block(out_channels=128,in_channels=512)
        self.deconv_block_1 = Deconv_block(out_channels = 32,in_channels=128)
        self.deconv_block_2 = Deconv_block(out_channels=4,in_channels=32)
        self.deconv_block_output = Deconv_block(out_channels=1,in_channels=4)
    
    def forward(self,x):
        x = self.deconv_block_input(x)
        x = self.deconv_block_1(x)
        x = self.deconv_block_2(x)
        x = self.deconv_block_output(x)
        return x
"""