# Image-Classifier-using-Pytorch-CNN
Udacity "Data Scientist" ND Project aimed to classify flower images into 102 categories

# Project Overview
In this project I have implemented transfer learning in pytorch using VGG16 deep network to classify flower images into 102 categories. As a result I got 88% accuracy on the test set.

# Installations
In order to run this project following libraries are required: torch, pil, pandas, numpy, matplotlib

# Files
**Image Classifier Project.ipynb** - jupyter notebook containing all project steps from image loading, transformation upon training to test the solution. THe resulting model is stored in the file (not included into this repository due to large size >25 MB).

**train.py** - python Skript that performs training of the model and saves resulting model (can be executed from command line with set of input parameters, for details refer to a file)

**predict.py** - python Skript that uses defined NN Model (saved in a file) to predict a kind of a flower (can be executed from command line with set of input parameters, for details refer to a file)

# Acknowledgements
This project is a part of Udacity "Data Scientist" Nanodegree program
