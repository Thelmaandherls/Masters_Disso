## Imports
import torch
import torchvision ## Contains some utilities for working with the image data
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
#%matplotlib inline
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn.functional as F

## Loading the MNIST dataset 
datatset = MNIST(root = 'data/', download = True)
print(len(datatset)) # 60,00 images 

image, label = datatset[10]
plt.imshow(image, cmap = "gray")
print(f"Label: {label}")
print("-------------------------------")

# Transforming the datatset whilst loading 
# To images with a 28 x 28 tensor
# The first dimension keets track of color channels - MNIST = grayscale so only 1 

training_data = MNIST(root = "data/", train = True, transform = transforms.ToTensor())
print(training_data)
print("-------------------------------")

image_converted_to_tensor, label = training_data[0]
print(image_converted_to_tensor, label)
print("-------------------------------")

print(image_converted_to_tensor[:, 10:15, 10:15])
print(torch.max(image_converted_to_tensor), torch.min(image_converted_to_tensor))
print("-------------------------------")

# Values range from 0 to 1, with 0 representing black, 1 white and the values between different shades of grey
plt.imshow(image_converted_to_tensor[0, 10:15, 10:15], cmap = "gray")

train_data, validation_data = random_split(training_data,[50000, 10000]
)

# Print the length of train and validation datasets
print("Length of Train Datasets: ", len(train_data))
print("Length of Validation Datasets: ", len(validation_data))

'''
While building a machine learning/Deep learning models, it is common to split the dataset into 3 parts:

Training set - The part of the data will be used to train the model,compute the loss and adjust the weights of the model using gradient descent.

Validation set - This part of the dataset will be used to evalute the traing model, adjusting the hyperparameters and pick the best version of the model.

Test set - This part of the dataset is used to final check the model predictions on the new unseen data to evaluate how well the model is performing.
'''

print("-------------------------------")


print("-------------------------------")


print("-------------------------------")


print("-------------------------------")


print("-------------------------------")



