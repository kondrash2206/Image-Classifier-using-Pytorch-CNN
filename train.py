# make some imports

import argparse
import os

parser = argparse.ArgumentParser(description = 'Train Model')
parser.add_argument('--data_directory', type = str)
parser.add_argument('--arch', default = 'vgg16', type = str)
parser.add_argument('--learning_rate', default = 0.003, type = float)
parser.add_argument('--hidden_units', default = 512, type = int)
parser.add_argument('--epochs', default = 1, type = int)
parser.add_argument('--save_dir', default = 'Model_flowers', type = str)
parser.add_argument('--gpu', default = 'cuda', type = str)
args = parser.parse_args()

def train_model(dat_dir,arch,hid_unit,lr,epoch,device,save_name):
    
    #check whether inputs are valid
    if os.path.isdir(dat_dir) == False:
        print('Please enter a valid path to the pictures')
        return
    
    if (hid_unit < 102) | (hid_unit > 25088): 
        print('Please enter a valid number of Hidden Units: 102 < Hid_Units < 25088')
        return
    
    if lr <= 0:
        print('Learning Rate must be >0')
        return
    
    if epoch < 0:
        print('Epochs number must be >0')
        return
    
    if (device != 'cpu') & (device != 'cuda'):
        print('Please choose "cpu"/"cuda" for --gpu')
        return
    
    # Imports
    import numpy as np
    import torch
    from torch import nn
    from torch import optim
    import torch.nn.functional as F
    from torchvision import datasets, transforms, models

    
    
    data_dir = dat_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'


    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
                                     


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=64, shuffle = False)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle = False)

    if arch == 'vgg13': 
        model = models.vgg13(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    else:
        print('Please choose VGG13/VGG16/VGG19 model')
        return
    
    
    
    # Freeze model parameters and weights
    for param in model.parameters():
        param.requires_grad = False

    # Define 3 output layers
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, hid_unit)),
                                            ('relu1', nn.ReLU()),
                                            ('dropout1', nn.Dropout(p = 0.2)),
                                            ('fc4', nn.Linear(hid_unit, 102)),
                                            ('output', nn.LogSoftmax(dim=1))]))
    # Assign new output layers to a model
    model.classifier = classifier
    
    #Define Loss Criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr = lr)
    
    # Send Model to Device
    model.to(device)

    # Loop over Epochs
    for e in range(epoch):
    
        # Null the Stat variables
        running_loss = 0
        test_loss = 0
        accuracy = 0
    
        # Loop over Training Batches
        for images, labels in trainloader:
        
            # Switch on Train Mode to use gradients
            model.train()
            #Send images and labels to GPU/CPU
            images, labels = images.to(device), labels.to(device)
        
            # NN Routine
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step ()
            # Calculate Stat
            running_loss += loss.item()
        # Validation Step
        else:
            #Switch off the Gradient calculations (to increase speed)
            with torch.no_grad():
                # Evaluation Regime (to increase speed)
                model.eval()
            
                #Loop over test Batches
                for images1, labels1 in validationloader:
                    #Send images and labels to GPU / CPU
                    images1, labels1 = images1.to(device), labels1.to(device)
                
                    #Forward pass through the model and get the probabilities
                    ps = torch.exp(model(images1))
                    #Get classes and probabilities from ts
                    top_p, top_class = ps.topk(1, dim = 1)
                    #Calculate Loss 
                    loss = criterion(model(images1),labels1)
                
                    # Statistics
                    test_loss += loss.item()
                    equals = top_class == labels1.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
            print(f'Epoch No: {e+1}')
            print(f'Accuracy Validation: {accuracy.item()/len(validationloader)*100}%')
            print(f'Training Loss: {running_loss/len(trainloader)}')
            print(f'Validation Loss: {test_loss/len(validationloader)}')
    
    #Save the model
    model.class_to_idx = train_data.class_to_idx
    torch.save({'architecture':arch,
            'classifier': model.classifier,
            'state_dict': model.state_dict(),
            'class_to_idx': model.class_to_idx},
           save_name)
    
# Main Programm 
arch = args.arch
dat_dir = args.data_directory
hid_units = args.hidden_units
lr = args.learning_rate
epoch = args.epochs
dev = args.gpu
save_name = args.save_dir

train_model(dat_dir,arch,hid_units,lr, epoch, dev, save_name)

