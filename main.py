import torchvision
import torchvision.transforms as transforms
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



if __name__ == '__main__':
    multiprocessing.freeze_support()


    # Define the directory to save the model
    save_dir = './saved_models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load the saved model
    model_path = os.path.join(save_dir, 'resnet_cifar10.pth')
    resnet = models.resnet18()
    resnet.load_state_dict(torch.load(model_path))


    # Define the rest of your code here

    # Save the trained model to a file
    save_path = os.path.join(save_dir, 'resnet_cifar10.pth')
    torch.save(resnet.state_dict(), save_path)
        

    import random

    # Define the desired train-test split ratio
    train_ratio = 0.8

    # Set the random seed for reproducibility
    random_seed = 42
    np.random.seed(random_seed)

    # Load the CIFAR10 dataset and split it into train and test subsets
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    cifar10_dataset = torchvision.datasets.CIFAR10(root=from_root(), train=True, download=True, transform=transform_train)

    # Get the total number of samples in the dataset and generate a random list of indices
    num_samples = len(cifar10_dataset)
    indices = list(range(num_samples))
    np.random.shuffle(indices)

    # Calculate the number of samples in the train and test sets based on the desired ratio
    num_train_samples = int(np.floor(train_ratio * num_samples))
    num_test_samples = num_samples - num_train_samples

    # Split the indices into train and test subsets based on the desired ratio
    train_indices = indices[:num_train_samples]
    test_indices = indices[num_train_samples:]

    # Use SubsetRandomSampler to extract the samples for the train and test subsets
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

    # Create the dataloaders for the train and test subsets using the subset samplers
    trainloader = torch.utils.data.DataLoader(cifar10_dataset, batch_size=128, sampler=train_sampler, num_workers=2)
    testloader = torch.utils.data.DataLoader(cifar10_dataset, batch_size=100, sampler=test_sampler, num_workers=2)

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
    
    save_dir = './trained_models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save the trained model to a file
    save_path = os.path.join(save_dir, 'resnet_cifar10.pth')
    torch.save(resnet.state_dict(), save_path)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = resnet(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def generate_caption(image_path):
        image = Image.open(image_path)
        image = transform_test(image)
