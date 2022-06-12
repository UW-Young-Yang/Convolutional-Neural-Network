# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms


class LeNet(nn.Module):
    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()
        # certain definitions
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fl = nn.Flatten()
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        # {1: torch.Size([32, 6, 14, 14]), 2: torch.Size([32, 16, 5, 5]), 3: torch.Size([32, 400]), 4: torch.Size([32, 256]), 5: torch.Size([32, 128]), 6: torch.Size([32, 100])}
        shape_dict = {}
        out = self.pool(self.relu(self.conv1(x)))
        shape_dict[1] = out.shape
        out = self.pool(self.relu(self.conv2(out)))
        shape_dict[2] = out.shape
        out = self.fl(out)
        shape_dict[3] = out.shape
        out = self.relu(self.fc1(out))
        shape_dict[4] = out.shape
        out = self.relu(self.fc2(out))
        shape_dict[5] = out.shape
        out = self.fc3(out)
        shape_dict[6] = out.shape
        return out, shape_dict


def count_model_params():
    '''
    return the number of trainable parameters of LeNet.
    '''
    model = LeNet()
    model_params = 0.0
    model_name = list(model.named_parameters())
    for params in model_name:
        model_params += params[1].numel()
    return model_params / 1e6


def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        ###################################
        # fill in the standard training loop of forward pass,
        # backward pass, loss computation and optimizer step
        ###################################

        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc

if __name__ == '__main__':
    # model_name = count_model_params()
    # params_dict = dict()
    # for params in model_name:
    #     name = params[0]
    #     param = params[1]
    #     params_dict[name] = param.numel() # Returns the total number of elements in the tensor
    print(count_model_params())