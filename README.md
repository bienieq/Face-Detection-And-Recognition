# Face-Detection-And-Recognition using OpenCV

Face Recognition project using OpenCV for Biometrics course at Wroclaw University of Science and Technology.

The goal of this project is to compare three algorithms used for face recognition: Eigenfaces, Fisherfaces, and Local Binary Pattern Histogram.

Each algorithm was tested against a test data set and two modified ones, with noise applied over the photos. The program allows to display the prediction on the identity of the subject and the "confidence" of that prediction, which is the Euclidean distance between the provided picture and the training data of the model. A lower value means bigger confidence. These values are also saved to a txt file, along with calculated efficiency (percentage of images with correct predictions, controlled by set threshold).

Face database used in the project: [Yalefaces](https://www.kaggle.com/datasets/olgabelitskaya/yale-face-database)

## Usage

Requirements: `cv2` and `numpy`

To use the program, run `python main.py <training-data-path-dir> <test-data-path-dir> <model-name> <output-filename>`, where `model-name` is one of the following algorithms: `[lbp|fish|pca]`.
