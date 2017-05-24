##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_examples.png
[image3]: ./output_images/sliding_windows.png
[image4]: ./output_images/sliding_window.png
[image5]: ./output_images/sliding_windows_test_image.png
[image6]: ./output_images/heatmap_all.png
[image7]: ./output_images/threshold.png
[image8]: ./output_images/labels.png
[image9]: ./output_images/result.png
[image11]: ./examples/bboxes_and_heat.png
[image12]: ./examples/labels_map.png
[image13]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the second code cell of the IPython notebook .  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters like` orientations`, `colorspace`,  `pixels_per_cell`, `cells_per_block`, `HOG_channel`. Finally I choose `orientations=11` and `pixels_per_cell=(16, 16)`,  `cells_per_block=(2, 2)`, `HOG_channel='ALL'`, `colorspace='YUV'` considering the SVM classifier's accuray and computation speed.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM feeding the HOG features only with default classifier parameters. At first I tried to combine the all three features (HOG, color, spatial) and feed to the classsifier, but it exhausted my old computer and the accuracy was not increased.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used the function `find_cars()` from the lesson materials.  This function performed the SVM classifier prediction on the HOG features for each window and returned a list of rectangle objects of the windows which were given a positive 'car' prediction by classifier.

I tried several configurations of window size and positions in order to match the images. The following image show the size and positons of search windows that used in video. There are 4 different sizes: 1x(blue), 1.5x(green), 2x(red) and 3.5x(black).

![alt text][image3]

Firstly I tried to search smaller scale window size (0.5x) and it returned lots of false positions. Additonally,  for the largest sacle (3.5x) the Y-startpoint should be lower than other scales. The overlapping of 75% in the Y-direction and X-direction could help to find more positive positions than the 50%.

The following image shows us the output results after run the function `find_car()` onto a test image. In the image there are two cars and several positive rectangle on them, but there is a rectangle on a oncoming car behind the guardrail, which we do NOT want. Normally the true positive positions are marked by several rectangles,  but on the contrary, the false positive positions marked by only one or two rectangles.

![alt text][image5]

Here I utilized the heatmap to differentiate them. The function `add_heat()` increases the value for each pixel of black image which locates in the range of the each rectangle. The areas which involve more overlapping rectangles are much 'hotter'.  

![alt text][image6]

In order to eliminate the false positive rectangle, a threshold (all pixels which don't exceed the threshold value were setted to 0, here the threshold value is 1) is applied.

![alt text][image7]

I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle. 

![alt text][image8]

The findal output image:

![alt text][image9]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I fed all the test images to the above pipeline and the results are displayed in the following images :

![alt text][image4]

In the above images there is no false positive rectangle and it identified two vehicles. In order to achieve this performance, I tried to different overlapping rates like 50% and 75%, window sizes and position. Additionally, a precise heatmap threshold need to be several times test. 

---
### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out_advanced.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

First time I generated the project  output video, I noticed that on the consecutive frames the vehicle just moved a little but the rectangle shifted a lot. It looked like the rectangles was always flickering. I created a class called `Rects_Store` to store the rectangle objects of previous frames. 

With help of the `Rects_Store`,  I could perform the heatmap/threshold/label steps not for the current frame, but for the combination of past 20 frames. The threshold for the heatmap is a key parameter and here was setted to `1+len(store_rect.rects)//2` since this value performed best.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The problem I faced in  implementation of this project was mainly how to predict  every window correctly.  When we extract more features to feed to the classifier, the accuracy of predictions could be increased and the classifier coulde be more robust. But in this way, the requirement of execution speed would be crucial. 

There is a potential reason to make the prediction fail -- the dataset for trainning the classifier is not big enough. In the real enviroment, there are a lot of vehicles with different colors, different sizes and different shapes. If the vehicles don't resemble those in the dataset, the classifier is probably to predict the false positive.

In future, I would improve the pipeline with a high accuracy classifier and more efficient window search method, such as:
- use convolutinal neural network instead of SVM classifier.
- divide the image into 4 parts from left to right. If the vehicle is drived on the left of the road, so the other vehicles mostly likely appear in the right 3 parts.
- calculate the vehicle's speed and predict the location in subsequent frames according to the location in the current frame. 

