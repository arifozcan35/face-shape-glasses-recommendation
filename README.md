# Glasses Recommendation According to Face Shape

This project aims to develop an artificial intelligence-based system that allows users to easily identify their face shape and receive eyewear recommendations suitable for this face shape. 



![Ekran görüntüsü 2025-01-30 221751](https://github.com/user-attachments/assets/a84f96b6-2998-4f74-9ca6-08929e16d0a2)





## Technical Methods

- *Model Training:*
  - Using the EfficientNet-B4 (CNN architecture) model, training was performed in 5 different face shape classes. (Heart, Oblong, Oval, Round, Square.)
  - In training the model, the dataset contains 800 samples for each class. And the data set contains a total of 5000 samples.
  - As a result of the training, the model reached 84% accuracy.

- *Image Processing:*
  - Images are processed using the PyTorch and Torchvision libraries.
  - The images are subjected to OpenCV's Haar Cascade method for face detection.
  - The input data for face shape classification is provided in a size and normalised format suitable for the model.


---



## *Dataset*

  - Data set research on Kaggle and selecting a data set suitable for the project.
  - **[Dataset Link](https://www.kaggle.com/datasets/niten19/faceshape-dataset)**

