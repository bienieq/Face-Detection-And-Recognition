import os
import cv2
import re

def write2file(file_name, text):
    dir_path = os.path.dirname(os.path.abspath(__file__)) + "/"
    file = open(dir_path + file_name + '.txt', 'a')
    file.write(text + '\n')
    file.close()

def read_dataset_and_labels(directory_path):
    images = []
    labels = []
    for file in os.listdir(directory_path):
        if '.png' in file:
            if file is not None:
                image = cv2.imread(os.path.join(directory_path, file), 0)
                image = cv2.resize(image, (320, 243))
                images.append(image)
            label = re.findall('[0-9]+', file)
            labels.append(int(''.join(label)))
    return tuple([images, labels])

def draw_info(img, rect, text):
    (x, y, w, h) = rect
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
    text_x = x + int((w - text_width) / 2)
    text_y = y + h + text_height + 5  # offset for text
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
