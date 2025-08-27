# Dogs vs Cats Classification using SVM
# Project Overview

This project focuses on building a Support Vector Machine (SVM) classifier to distinguish between images of dogs and cats using the Kaggle Dogs vs Cats dataset.
Instead of deep learning models like CNNs, this project demonstrates how traditional machine learning models (like SVM) can be applied to image classification tasks with proper preprocessing and feature extraction.

The primary objective is to:

Preprocess images (resize, grayscale, flatten).

Extract features for classification.

Train an SVM classifier on the dataset.

Evaluate and predict new unseen images.

# Dataset Description

The dataset used is the Kaggle Dogs vs Cats dataset.

Training set: Contains labeled images of cats and dogs.

Testing set: Contains unlabeled images used for evaluation.

Labels:

0 → Cat

1 → Dog

👉 Dataset Link: Dogs vs Cats on Kaggle

# Workflow & Methodology

Dataset Loading

Load images from train/test directories.

Assign labels (cat = 0, dog = 1).

Preprocessing

Resize images to 64×64 pixels.

Convert to grayscale for simplicity.

Flatten images into 1D feature vectors.

Normalize pixel values (0–1).

# Model Training

Use Support Vector Machine (SVM) with RBF/Linear kernel.

Train on processed feature vectors.

# Evaluation

Test the model on unseen test data.

Measure accuracy, precision, recall, and F1-score.

# Prediction

Predict whether a given image is a cat or dog.

⚙️ Installation

Clone the repository:

git clone https://github.com/your-username/dogs-vs-cats-svm.git
cd dogs-vs-cats-svm


Install required dependencies:

pip install -r requirements.txt

# How to Run the Project

Prepare Dataset

Download dataset from Kaggle and place it in the project folder:

/dataset/train
/dataset/test


Run Training Script

python train.py


Run Prediction on a Sample Image

python predict.py --image sample.jpg

# Sample Prediction

Example input:

sample.jpg → (dog image)


Model output:

Prediction: Dog 🐶

# File Structure
dogs-vs-cats-svm/
│── dataset/
│   ├── train/
│   │   ├── cat.0.jpg
│   │   ├── dog.1.jpg
│   ├── test/
│── models/
│   ├── svm_model.pkl
│── train.py          # Script to train the SVM model
│── predict.py        # Script to make predictions on new images
│── requirements.txt  # Dependencies
│── README.md         # Project documentation

🚀 Future Enhancements

Apply CNNs for better performance.

Add data augmentation for robust training.

Deploy the model with Flask/Streamlit for user interaction.

⚡ This project highlights the usage of traditional ML with SVM in solving an image classification problem, serving as a stepping stone towards deep learning models.

# Developed By 

HARSHITHA PRASAD S G

GITHUB: harshithaprasadprasad

LINKEDIN: https://www.linkedin.com/in/harshitha-prasad-s-g-55a05a257
