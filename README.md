# cHEST-_cNN_CT-SCAN-
Data contain 3 chest cancer types which are Adenocarcinoma,Large cell carcinoma, Squamous cell carcinoma ,  : Ensample Model  and 1 folder for the normal cell  using Cnn, ResNet50 

## Getting Started:

     #Install with pip install:
     
          ## Core Libraries: ##

numpy as np: Fundamental library for numerical computations, array manipulation, and linear algebra operations.
pandas as pd: Powerful library for data analysis and manipulation, particularly for structured data in tables and DataFrames.
cv2: OpenCV library for real-time computer vision, image and video processing, and analysis.
keras: High-level API for building and training deep learning models, often used with TensorFlow as the backend.
tensorflow as tf: Open-source framework for machine learning, particularly for deep neural networks.
tensorflow_datasets as tfds: Library for accessing and loading ready-to-use datasets from TensorFlow.
Visualization:

matplotlib.pyplot as plt: Plotting library for creating various visualizations, including graphs, charts, and images.
Model Building and Layers:

keras.models: Provides classes for defining neural network architectures (Sequential and Model).
keras.layers: Offers a variety of layers for building neural networks, including:
Dense: Fully connected layers.
Dropout: Regularization technique to prevent overfitting.
Flatten: Transforms multidimensional input into a single vector.
Conv2D: Convolutional layers for image processing.
MaxPooling2D: Downsampling layers for feature extraction.
BatchNormalization: Normalizes layer inputs for faster training and stability.
Optimizers:

keras.optimizers: Contains optimization algorithms for training models, such as Stochastic Gradient Descent (SGD).
Pre-trained Models:

keras.applications: Offers pre-trained convolutional neural networks (CNNs) for image classification tasks:
MobileNet, VGG16, ResNet50
tensorflow.keras.applications.resnet50: Provides preprocessing functions specifically for ResNet50.
Callbacks:

keras.callbacks: Allows monitoring and managing model training, including:
ModelCheckpoint: Saves model weights at regular intervals.
EarlyStopping: Stops training if validation performance stops improving.
ReduceLROnPlateau: Reduces learning rate when progress plateaus.
Metrics:

sklearn.metrics: Provides functions for evaluating model performance:
precision_score, recall_score, accuracy_score
classification_report, confusion_matrix
Data Augmentation:

keras.preprocessing.image: Offers tools for image preprocessing and augmentation.
ImageDataGenerator: Generates batches of augmented images for training.
Data Splitting:

sklearn.model_selection: Contains functions for splitting data into training and testing sets.
train_test_split
Image Handling:

matplotlib.image as img: Provides functions for loading and displaying images.
File Operations:

shutil: Library for file management tasks, such as copying or moving files.

Example 1: Image Classification with ResNet50

# Image Classification with ResNet50,CNN, 

This repository contains a pre-trained ResNet50 model fine-tuned for image classification on a custom dataset.

## Getting Started

Install dependencies:
Bash
pip install -r requirements.txt
Use code with caution. Learn more
Download dataset:
Preprocess data:

python preprocess_data.py
Use code with caution. Learn more
Train model (optional):
Bash
python train.py
Use code with caution. Learn more
Make predictions:
Bash
python predict.py --image_path <path_to_image>
Use code with caution. Learn more
## Model Details

Architecture: ResNet50 with pre-trained ImageNet weights
Dataset: Chest CT-Scan images Dataset images in 5 classes
Training: Fine-tuned for 5 epochs with Adam optimizer
Evaluation: Achieved 92% accuracy on the test set
## Usage Examples

See predict.py for examples of making predictions on new images.
## Contributing

Fork the repository and create a pull request.
Follow the code style guidelines.
## License

MIT License
## Contact

For questions or issues, contact [
20221459892 محمد على محمد عبد العزيز
20221461977 احالم محمد مصطفى محمد
20221441500 آلاء عادل عبد الحميد عقيلى
210101068 نورهان محمد صالح الدين علي
20221464983 يارا حاتم ابراهيم] 
Example 2: image classification Analysis 



... (similar structure as above) ...

## Model Details
Training: Fine-tuned for 3 epochs with AdamW optimizer
Evaluation: Achieved 92% accuracy on the test set
