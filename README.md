# Rice Classification using Deep Learning
This project focuses on classifying rice grains using deep learning techniques. The goal is to develop a model that can accurately classify rice grain images into different classes. The project utilizes the PyTorch framework and implements a neural network architecture for training and evaluation.

## Dataset
The dataset used in this project consists of rice grain images. The image data is accompanied by a CSV file containing the corresponding labels for each image. The dataset is split into training and testing sets, with a ratio of 80:20. The images are preprocessed using various transformations, such as resizing and converting them to tensors.

Download
```
https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset/download?datasetVersionNumber=1
```

## Getting Started
- Clone the repository to your local machine.
- Ensure that you have the necessary dependencies installed. You can find the required libraries in the requirements.txt file.
- Prepare your dataset by providing the path to the CSV file (image_names.csv) and the folder containing the images (Allpics).
- Adjust the hyperparameters in the config.py file according to your preferences.
- Run the rain.py file to start training the model. The training progress will be displayed, showing the loss at each step.
- Once training is complete, the model will be saved as model.pth.
- Finally, the model's performance will be evaluated on the test set, and the accuracy will be displayed.


## Model Architecture
The neural network architecture used in this project is implemented in the NeuralNet.py file. It consists one hidden layer. The number of neurons in the input layer corresponds to the size of the input images, while the number of neurons in the output layer is equal to the number of distinct rice grain classes. The model uses the Adam optimizer and the cross-entropy loss function for training.

## Results
The trained model achieves an accuracy of 97% on the test set. This demonstrates its effectiveness in classifying rice grains accurately. Feel free to experiment with different hyperparameters or further enhance the model to achieve even better results.

Happy classifying!
