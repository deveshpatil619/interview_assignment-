import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from from_root import from_root 
import torch
import multiprocessing
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
import torch.cuda


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_resnet():
    for param in resnet.parameters():
        param.requires_grad = False
        resnet.fc = nn.Linear(512, 10)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(resnet.fc.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = resnet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0


if __name__ == '__main__':

    multiprocessing.freeze_support()

    # Define the directory to save the model
    save_dir = './saved_models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

   
    # Load the saved model
    # Load the pre-trained ResNet-18 model for CIFAR-10 from PyTorch's website
    resnet = models.resnet18(pretrained=True, progress=True)

    # Remove the last two layers of the pre-trained model (which are assumed to be the classification layers)
    resnet = nn.Sequential(*list(resnet.children())[:-2])

    # Replace the last layer with a new layer that has 10 output features (for the 10 CIFAR-10 classes)
    num_ftrs = resnet[-1][-1].in_features
    resnet[-1][-1] = nn.Linear(num_ftrs, 10)

    # Set the model to evaluation mode
    resnet.eval()

    # Print the model architecture
    print(resnet)







    # Define the desired train-test split ratio
    train_ratio = 0.8

    # Set the random seed for reproducibility
    random_seed = 42
    np.random.seed(random_seed)

    # Load the CIFAR10 dataset and split it into train and test subsets
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    cifar10_dataset = torchvision.datasets.CIFAR10(root=os.getcwd(), train=True, download=True, transform=transform_train)

    # Get the total number of samples in the dataset and generate a random list of indices
    num_samples = len(cifar10_dataset)
    indices = list(range(num_samples))
    np.random.shuffle(indices)

    # Calculate the number of samples in the train and test sets based on the desired ratio
    num_train_samples = int(np.floor(train_ratio * num_samples))
    num_test_samples = num_samples - num_train_samples

    # Split the indices into train and test subsets based on the desired ratio
    # Split the indices into train and test subsets based on the desired ratio
    train_indices = indices[:num_train_samples]
    test_indices = indices[num_train_samples:]

    # Use SubsetRandomSampler to extract the samples for the train and test subsets
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

    # Create the dataloaders for the train and test subsets using the subset samplers
    trainloader = torch.utils.data.DataLoader(cifar10_dataset, batch_size=128, sampler=train_sampler, num_workers=2)
    testloader = torch.utils.data.DataLoader(cifar10_dataset, batch_size=100, sampler=test_sampler, num_workers=2)

    # Train the fully connected layer
    train_resnet()

    save_dir = './trained_models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the trained model to a file
    save_path = os.path.join(save_dir, 'resnet_cifar10.pth')
    torch.save(resnet.state_dict(), save_path)

    ## Calculate the accuracy of the model on the test set
    correct = 0
    total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = resnet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the %d test images: %d %%' % (num_test_samples, 100 * correct / total))

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def generate_caption(image_path):
    image = Image.open(image_path)
    image = transform_test(image)
    output = resnet(image.unsqueeze(0))
    _, predicted = torch.max(output, 1)
    return classes[predicted]

image_path = 'cifar10_images_jpg\\11_7.jpg'
print(generate_caption(image_path))

