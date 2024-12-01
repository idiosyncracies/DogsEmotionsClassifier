import torch
import torch.nn as nn
from torchsummary import summary
from ptflops import get_model_complexity_info  ## pip install ptflops
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Define convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # Define fully connected layers
        self.fc1 = nn.Linear(64 * 28 * 28, 128)  # Adjusted based on input image size
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 4)  # Output class is 4

    def forward(self, x):
        # Forward pass through convolutional layers
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        # Flatten the tensor for fully connected layers
        x = x.view(-1, 64 * 28 * 28)
        # Forward pass through fully connected layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

if __name__ == '__main__':
    model = CNN()

    ### 以下代码检查网络大小
    # print(summary(model, (3, 224, 224)))  ## 打印网络结构和大小
    macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    # 检查是否有 None 值
    macs = macs if macs is not None else "Not Available"
    params = params if params is not None else "Not Available"
    print('{:<30}  {:<8}'.format('Number of parameters:', params))
    print('{:<30}  {:<8}'.format('Computational complexity:', macs))
