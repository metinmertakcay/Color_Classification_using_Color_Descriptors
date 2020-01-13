# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:43:46 2019
@author: Metin Mert Akçay
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from scipy.stats import skew, kurtosis
from sklearn import metrics
from tqdm import tqdm
import numpy as np
import sys
import cv2
import os

BIN_SIZE = 16
TEST_PATH = 'test'
TRAIN_PATH = 'train'

""" 
    This function is used to read images.
    @param image_path: path of the image
    @return image: image
"""
def read_image(image_path):
    image = cv2.imread(image_path, cv2.COLOR_GRAY2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
    return image


"""
    This function is used to create histogram. After creation of histogram, histogram is 
        divided by total number of pixels and normalizing each histogram value between 0 and 1.
    @param image: image
    @return feature: normalized histogram values for each channel
"""
def normalized_color_histogram(image):
    row, column, channel = image.shape[:3]
    size = row * column
    
    feature = []
    for k in range(channel):
        histogram = np.squeeze(cv2.calcHist([image], [k], None, [BIN_SIZE], [0, 256]))
        histogram = histogram / size
        feature.extend(histogram)
    return feature


"""
    This function is used for find color moments.
    @param channel: channel (L, a, b)
    @return feature: color moment results ​​of the examined channel
"""
def moment(channel):
    feature = []    
    feature.append(np.mean(channel))
    feature.append(np.std(channel))
    feature.append(skew(channel))
    feature.append(kurtosis(channel))
    return feature


"""
    This function is used to create color moment features.
    @param image: image
    @return feature: calculated color moment values ​​for each channel
"""
def color_moment(image):
    row, column, channel = image.shape[:3]
    
    channel_list = []
    for i in range(channel):
        channel_list.append([])
    
    for i in range(row):
        for j in range(column):
            for k in range(channel):
                channel_list[k].append(image[i][j][k])
    
    feature = []
    for i in range(channel):
        feature.extend(moment(channel_list[i]))
    
    return feature
    

"""
    This function is used for apply normalization operation for moment feature
    @param data: All extracted features from images
    @param str_point: start point of moment features
    @param number_of_channel: number of channel in image
"""
def normalize_moment_feature(data, str_point, number_of_channel):
    # 4 : number of color moment feature
    end_point = str_point + number_of_channel * 4
    
    number_of_data = len(data)
    for i in range(str_point, end_point):
        min_val = sys.maxsize
        max_val = 0
        for j in range(number_of_data):
            if data[j][i] < min_val:
                min_val = data[j][i]
            if data[j][i] > max_val:
                max_val = data[j][i]
        
        # min - max normalization
        for j in range(number_of_data):
            data[j][i] = (data[j][i] - min_val) / (max_val - min_val)


""" code start """
if __name__ == '__main__':
    # find number of train images
    number_of_train_image_count = 0
    color_list = os.listdir(TRAIN_PATH)
    for index, color_name in enumerate(color_list):
        path = os.path.join(TRAIN_PATH, color_name)
        image_list = os.listdir(os.path.join(path))
        for image_name in image_list:
            number_of_train_image_count += 1
            
    # find number of test images
    number_of_test_image_count = 0
    color_list = os.listdir(TEST_PATH)
    for index, color_name in enumerate(color_list):
        path = os.path.join(TEST_PATH, color_name)
        image_list = os.listdir(os.path.join(path))
        for image_name in image_list:
            number_of_test_image_count += 1
    
    print('<----------TRAIN START ---------->')
    train_data = []
    train_label = []
    color_list = os.listdir(TRAIN_PATH)
    with tqdm(total=number_of_train_image_count) as pbar:
        for index, color_name in enumerate(color_list):
            path = os.path.join(TRAIN_PATH, color_name)
            image_list = os.listdir(os.path.join(path))
            for image_name in image_list:
                image = read_image(os.path.join(path, image_name))
                histogram_features = normalized_color_histogram(image)
                moment_features = color_moment(image)
                train_data.append(histogram_features + moment_features)
                train_label.append(index)
                pbar.update(1)
                print(' ', color_name, image_name)
    normalize_moment_feature(train_data, BIN_SIZE * image.shape[2], image.shape[2])
    
    model = KNeighborsClassifier(n_neighbors = 5)
    model.fit(train_data, train_label)
    
    print('<----------TEST START ---------->')
    test_data = []
    test_label = []
    color_list = os.listdir(TEST_PATH)
    with tqdm(total=number_of_test_image_count) as pbar:
        for index, color_name in enumerate(color_list):
            path = os.path.join(TEST_PATH, color_name)
            image_list = os.listdir(os.path.join(path))
            for image_name in image_list:
                image = read_image(os.path.join(path, image_name))
                histogram_features = normalized_color_histogram(image)
                moment_features = color_moment(image)
                test_data.append(histogram_features + moment_features)
                test_label.append(index)
                pbar.update(1)
                print(' ', color_name, image_name)
    normalize_moment_feature(test_data, BIN_SIZE * image.shape[2], image.shape[2])
    
    prediction = model.predict(test_data)
    print()
    print("Accuracy:", metrics.accuracy_score(test_label, prediction))
    print()
    print(confusion_matrix(test_label, prediction))