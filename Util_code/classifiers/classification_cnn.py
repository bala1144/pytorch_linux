"""ClassificationCNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationCNN(nn.Module):
    """
    conv - relu - 2x2 max pool - fc - dropout - relu - fc

    The network operates on mini-batches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, kernel_size=7,
                 stride_conv=1, weight_scale=0.001, pool=2, stride_pool=2, hidden_dim=100,
                 num_classes=10, dropout=0.0):

        super(ClassificationCNN, self).__init__()
        channels, height, width = input_dim

        pad = int((kernel_size-1) / 2)
        # compute height and width of input to first fully connected layer
        height_fc1 = int((1 + (height + 2*pad - kernel_size)/stride_conv) / pool)
        width_fc1 = int((1 + (width + 2*pad - kernel_size)/stride_conv) / pool)


        # initialize layers
        #-------------------
        # Convolutional layer
        self.conv = nn.Conv2d(channels, num_filters, kernel_size, stride_conv, pad)
        #nn.init.xavier_uniform(self.conv.weight.data)
        self.conv.weight.data.mul_(weight_scale)

        # Fully Connected Layers
        self.fc1 = nn.Linear(num_filters * height_fc1 * width_fc1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.linear(hidden_dim,num_classes)
        # Other Layers
        self.dropout = nn.Dropout(p=dropout)
        self.maxpool = nn.MaxPool2d(pool, stride=stride_pool)

        
    
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def forward(self, x):
        """
        Forward pass of the neural network. Should not be called manually but by
        calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """

        #1 Convolutional Layer
        x = self.conv(x)
        #2 ReLU Layer
        x = F.relu(x)
        #3 Max Pool Layer
        x = self.maxpool(x)
        #4 Fully Connected Layer
        x = x.view(x.size()[0], -1)     # flatten data for fc
        x = self.fc1(x)
        #5 Dropout Layer
        x = self.dropout(x)
        #6 ReLu Layer
        x = F.relu(x)
        #7 Fully Connected Layer
        x = self.fc2(x)
        #3rd fully connected layer
        x=self.fc3(F.relu(self.dropout(x)))
    
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
