import pickle as pickle
import numpy as np
import os
import torch
import torch.utils.data as data

#class to import the X,Y into main process
class CIFAR10Data(data.Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        img = self.X[index]
        label = self.y[index]

        img = torch.from_numpy(img)
        return img, label

    def __len__(self):
        return len(self.y)

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        # load with encoding because file was pickled with Python 2
        datadict = pickle.load(f, encoding='latin1')
        X = np.array(datadict['data'])
        Y = np.array(datadict['labels'])
        X = X.reshape(-1, 3, 32, 32).transpose(0,2,3,1).astype("float")
        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    f = os.path.join(ROOT, 'cifar10_train.p')
    #print (os.path(ROOT))
    print (f)
    Xtr, Ytr = load_CIFAR_batch(f)
    return Xtr, Ytr


def get_CIFAR10_datasets(num_training=48000, num_validation=1000,
                         num_test=1000, dtype=np.float32):
    """
    Load and preprocess the CIFAR-10 dataset.
    """
    path = 'datasets/cifar10_train.p'
    with open(path, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = np.array(datadict['data'])
        y = np.array(datadict['labels'])
        X = X.reshape(-1, 3, 32, 32).astype(dtype)

    X /= 255.0
    # Normalize the data: subtract the mean image
    mean_image = np.mean(X, axis=0)
    X -= mean_image

    # Subsample the data
    mask = range(num_training)
    X_train = X[mask]
    y_train = y[mask]
    mask = range(num_training, num_training + num_validation)
    X_val = X[mask]
    y_val = y[mask]
    mask = range(num_training + num_validation,
                 num_training + num_validation + num_test)
    X_test = X[mask]
    y_test = y[mask]

    return (CIFAR10Data(X_train, y_train),
            CIFAR10Data(X_val, y_val),
            CIFAR10Data(X_test, y_test),
            mean_image)



def scoring_function(x, lin_exp_boundary, doubling_rate):
    assert np.all([x >= 0, x <= 1])
    score = np.zeros(x.shape)
    lin_exp_boundary = lin_exp_boundary
    linear_region = np.logical_and(x > 0.1, x <= lin_exp_boundary)
    exp_region = np.logical_and(x > lin_exp_boundary, x <= 1)
    score[linear_region] = 100.0 * x[linear_region]
    c = doubling_rate
    a = 100.0 * lin_exp_boundary / np.exp(lin_exp_boundary * np.log(2) / c)
    b = np.log(2.0) / c
    score[exp_region] = a * np.exp(b * x[exp_region])
    return score