import cv2
import sys
import numpy as np

import file_utils as fu

# face detection on the image using haar classifier
def detect_face(img):
    tmp = img.copy()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(tmp, scaleFactor=1.2, minNeighbors=5);
    if (len(faces) == 0):
        return None, None
    (x, y, w, h) = faces[0]
    return tmp[y:y+w, x:x+h], faces[0]


# predict the subject on the image, add labels for the image
subjects = ["", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15"]

def predict_img(test_img):
    img = test_img.copy()
    face, rect = detect_face(img) 
    id_confidence = model.predict(img)
    label_text = subjects[id_confidence[0]] + " " + str(id_confidence[1])
    if rect is not None:
        fu.draw_info(img, rect, label_text)
    return img


if len(sys.argv) != 5:
    raise ValueError("Please provide correct arguments: training data path, test data path, model name (lbp, pca, fish) and output filename")
else:
    # argv1 - training data
    training_data, training_labels = fu.read_dataset_and_labels(sys.argv[1])
    # argv2 - test data
    test_data, test_labels = fu.read_dataset_and_labels(sys.argv[2])
    # argv3 - model name
    model_name = sys.argv[3]
    # argv4 - output filename
    filename = sys.argv[4]

# check for model 
if model_name == 'lbp':
    model = cv2.face.LBPHFaceRecognizer_create()
elif model_name == 'pca':
    model = cv2.face.EigenFaceRecognizer_create()
elif model_name == 'fish':
    model = cv2.face.FisherFaceRecognizer_create()
else:
    raise AttributeError()

# distance above this value is assumed incorrect
treshold = 8000

model.train(training_data, np.asarray(training_labels))

# display the results, save the predictions and efficiency to txt file
correct = 0
for i in range(len(test_data)):
    
    prediction = model.predict(test_data[i])
    text_line = str(prediction[0]) + ", " + str(prediction[1])
    fu.write2file(filename, text_line)
    if (prediction[0] == test_labels[i]):
        if(prediction[1] <= treshold):
            correct += 1
    
    predicted_img1 = predict_img(cv2.resize(test_data[i], (320, 243)))
    title_win = str(prediction[0]) + "prediciton" 
    cv2.imshow(title_win, cv2.resize(predicted_img1, (320, 243)))
    k = cv2.waitKey(0) 
    if k==27:
        break 
    cv2.destroyAllWindows()

fu.write2file(filename, str("efficiency: " + str(correct / len(test_data))))
