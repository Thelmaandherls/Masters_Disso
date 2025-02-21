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

batch_size = 128
train_loader = DataLoader(train_data, batch_size, shuffle = True)
val_loader = DataLoader(validation_data, batch_size, shuffle = False)


"""
Logistic Regression model is identical to a linear regression model i.e, there are weights and bias matrices, and the output is obtained using simple matrix operations(pred = x@ w.t() + b).

We can use nn.Linear to create the model instead of defining and initializing the matrices manually.

Since nn.Linear expects the each training example to a vector, each 1 X 28 X 28 image tensor needs to be flattened out into a vector of size 784(28 X 28), before being passed into the model.

The output for each image is vector of size 10, with each element of the vector signifying the probability a particular target label(i.e 0 to 9). The predicted label for an image is simply the one with the highest probability.

"""
import torch.nn as nn
input_size = 28 * 28
num_classes = 10

## Logistic regression model
model = nn.Linear(input_size, num_classes)
print(model.weight.shape)
print(model.weight)
print(model.bias.shape)
print(model.bias)
print("-------------------------------")

## The model is simply a linear transformation of the input, and does not take into account the non-linear relationships that may exist within the data
## The output of the model is not a probability distribution over the 10 classes, to convert the output into probabilities, we use the softmax function, which has the following formula:
'''
for images, labels in train_loader:
    print(labels)
    print(images.shape)
    outputs = model(images)
    break
    '''

"""
Note This leads to an error, because our input data does not have the right shape. Our images are of the shape 1X28X28, but we need them to be vectors of size 784 i.e we need to flatten them out. We will use the .reshape() method of a tensor, which will allow us to efficiently view each image as a flat vector, without really changing the underlying data.
RuntimeError: mat1 and mat2 shapes cannot be multiplied (3584x28 and 784x10)

To include this additional functionality within model, we need to define a custom model, by extending the nn.Module class from PyTorch"""
print("-------------------------------")


class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
        
    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        print(xb)
        out = self.linear(xb)
        print(out)
        return(out)

model = MnistModel()
print(model.linear.weight.shape, model.linear.bias.shape)
list(model.parameters())
print("-------------------------------")


for images, labels in train_loader:
    outputs = model(images)
    break
    
print('outputs shape: ', outputs.shape)
print('Sample outputs: \n', outputs[:2].data)
print("-------------------------------")

## Apply softmax for each output row
probs = F.softmax(outputs, dim = 1)

## chaecking at sample probabilities
print("Sample probabilities:\n", probs[:2].data)

print("\n")
## Add up the probabilities of an output row
print("Sum: ", torch.sum(probs[0]).item())
max_probs, preds = torch.max(probs, dim = 1)
print("\n")
print(preds)
print("\n")
print(max_probs)
print("-------------------------------")
print(labels)