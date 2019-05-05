# make some imports
import argparse
import os
from PIL import Image
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json

# Parsers
parser = argparse.ArgumentParser(description = 'Train Model')
parser.add_argument('--path', type = str)
parser.add_argument('--checkpoint', type = str)
parser.add_argument('--top_k', default = 5, type = int)
parser.add_argument('--category_names', default = 'cat_to_name.json', type = str)
parser.add_argument('--gpu', default = 'cpu', type = str)
args = parser.parse_args()


def predict_model(checkpoint,image_path, top_k,cat_name,device):
    # Check some inputs
    if os.path.exists(checkpoint) == False:
        print('Please type an existing checkpoint name')
        return
    if os.path.exists(image_path) == False:
        print("Can't find a picture under this path")
        return
    if (top_k <=0) | (top_k >102):
        print('Please enter a valid number of top categories 0 < cat <=102')
        return
              
    # Load pretrained model
    load_m = torch.load(checkpoint)
    
    if load_m['architecture'] == 'vgg16': 
        model = models.vgg16(pretrained=True)
        # Freeze the vgg parameters
        for param in model.parameters():
            param.requires_grad = False
        # Setup the model
        model.class_to_idx = load_m['class_to_idx']
        model.classifier = load_m['classifier']
        model.load_state_dict(load_m['state_dict'])
    else: 
        print('This model was trained on VGG16 Network!')
    
    # Predict class
    image = process_image(Image.open(image_path))
    model.to(device)
    
    with torch.no_grad():  
        model.eval()
        image = image.to(device)
        image.unsqueeze_(0)
        # make a forward pass
        ps = torch.exp(model(image))
        
        #Get classes and probabilities from ts
        top_p, top_class = ps.topk(top_k, dim = 1)
        
        # translate top classes and probabilities into numpy
        top_p = top_p.to('cpu')
        top_class = top_class.to('cpu')
        top_p = top_p.numpy()
        top_class = top_class.numpy()

    # Get top classes and probabilities
    pred_probability = top_p[0]
    pred_class = top_class[0]

    # Transform into "real" categories
    transform_cat = model.class_to_idx
    real_class = []

    for i in pred_class:
        for new, old in transform_cat.items():
            if old == i:
                real_class.append(new)
    # get flower names
    with open(cat_name, 'r') as f:
        cat_to_name = json.load(f)
    
    classes = []
    for i in real_class: 
        classes.append(cat_to_name[i])
    # Print results
    print('Flower: ',classes)
    print('Probability: ',pred_probability)
    
def process_image(image):
    # Process a PIL image for use in a PyTorch model
    normalize_pil = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    preprocess_pil = transforms.Compose([
                    transforms.Scale(255),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
    normalize_pil])
    return preprocess_pil(image)
    
    
# Main Programm 
checkpoint = args.checkpoint
top_k = args.top_k
cat_name = args.category_names
path = args.path
device = args.gpu

predict_model(checkpoint, path, top_k,cat_name,device)

