
# Color Classification
## HOW TO RUN PROGRAM?
- Go to directory where you download the project.
- Open console screen and enter “python main.py” command to run.

## REQUIREMENT
You can install necessary libraries using “pip install”.
- Sklearn
- Scipy
- OpenCv
- Numpy
- tqdm

## WHERE SHOULD IMAGES BE REPLACED?
The test images should be placed under the test folder and the train images should be placed under the train folder. You should put these folders in the same directory as the source code.

## DATASET
[Train images](/train)
[Test images](/test)

## PROJECT STEPS
> In order to do training, objects belonging to the colors which were desired to be classified from the web and kaggle were found.

> The size of each image has been reduced. This was done to speed up training and testing operation. The size of all images is set to 256 x 256.

>The images has been read in the L, u, v color space.

<p align="center">
	<img src="/images/table.JPG" alt="Number of images found in the train and test sets" width="400" height="120">
</p>

>Color histogram was calculated for each image. Histogram was performed by calcHist method in OpenCv. Histogram values ​​were normalized after creation of the histogram. By selecting the bin-size 16, the number of features has been reduced.

>After histogram extraction step, color moments were obtained in the images. Color moments are mean, standard deviation, skew and kurtosis. After finding of these values, normalization was performed using the min-max normalization [0 - 1].

>Knn is an algorithm that tries to find the closest images in the training set for the selected image from the test set. The closest “5” pictures were found and the most dominant class was assigned as the class for the test image.

>Feature extraction steps (color histogram and color moment) also was performed for test images.

> After training and feature extraction step, class label was found by using knn model for test images.

## RESULT & COMMENT
- Model accuracy (correct classified image) is 0.875.
- Model correctly predicted 70 of the 80 images.
- Model success and matrix are shown in the image below.
- Model fully recognizes the images containing black, blue, red and white colors in the created test folder.
- Model especially mixes yellow with purple and orange with red.
- To increase success, the number of images in the training set can be increased or a different classifier (SVM, Naive Bayes etc.) can be used. In addition, a different color space can be used or color correlogram feature can be extracted.
- In the images used in training and test set, the areas outside the objects are different color from the actual object. Therefore, these areas affect classification.

<p align="center">
	<img src="/images/confusion.JPG" alt="Confusion Matrix" width="300" height="300">
</p>