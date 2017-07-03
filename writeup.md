#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[image9]: ./imgs/class_distribution_bar_chart.png "Class Distribution"
[image10]: ./imgs/30km_color.png "30km/hr RGB"
[image11]: ./imgs/30km_gray.png "30km/hr Gray"
[image12]: ./imgs/30km_sign.jpg "30km/hr Sign" 
[image13]: ./imgs/caution_sign.jpg "Caution Sign"
[image14]: ./imgs/pedestrian_sign.jpg "Pedestrian Sign"
[image15]: ./imgs/stop_sign.png "Stop Sign"
[image16]: ./imgs/yield_sign.jpg "Yield Sign"



## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 31367
* The size of the validation set is 7842
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image9]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale. Grayscaling reduced the number of features present in a model which helped in training the model faster.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image10]        ![alt text][image11]


As a last step, I normalized the image data because it reduces the distribution of pixel values to [-1, 1]. 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray scale image   					| 
| Convolution1     	    | 1x1 stride,padding: valid, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, padding:valid,outputs 14x14x6 	|
| Convolution2      	| 1x1 stride,padding: valid, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, padding:valid,outputs 5x5x16 	    |
| Fully connected1		| input: 400, output: 200        				|
| RELU					|												|
| Fully connected2		| input: 200, output: 84        				|
| RELU					|												|
| Fully connected3		| input: 84, output: 43        					|
| Softmax Cross Entropy	|         							|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I trained the model on the training set and after each epoch tested accuracy on validation set.
I am using Adam Optimizer for training. 
Following are the parameter values that I used for training 
Batch size: 128
Number of epochs: 15
Learning rate: 0.001

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 98.2%
* validation set accuracy of 97.2%
* test set accuracy of 88.2%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* Architecture
For training I chose LeNet architecture. LeNet architecture is a simple architecture to implement and gave a very good accuracy(>97). 
* Why did you believe it would be relevant to the traffic sign application?
LeNet performs very good for image classification which is relevant for traffic sign classification.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
Training, validation accuracy improved with each epoch. This demonstrated that the model is learning with each epoch. Final validation accuracy was ~ 97% which is a very good classification accuracy. 
After training the final test accuracy was ~ 88% which is a very good classification accuracy.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image12]{:width="200"}
![alt text][image13]{:width="200"} 
![alt text][image14]{:width="200"} 
![alt text][image15]{:width="200"} 
![alt text][image16]{:width="200"}

Second image was difficult to classifies because Pedestrian sign resembles a General Caution sign
####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Pedestrian     		| General Caution 								|
| 30 km/h				| 30 km/h									    |
| General Caution	    | General Caution					 			|
| Yield			        | Yield      							        |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ~88%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

For the first image, the model is relatively sure that this is a stop sign (probability of 0.99), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99999994         	| Stop sign   									| 
| 5.2332915e-08   		| No entry 										|
| 2.1297833e-08			| Keep right									|
| 1.2949982e-08	      	| Keep right ahead					 			|
| 3.2493966e-10		    | Priority road      							|


For the second image , classifier mades a mistake. It wrongly classifies Pedestrian sign as General Caution. Although, Pedestrian is the second most probable with .09 probability. 


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.91073626         	| General caution   							| 
| 0.08926335   			| Pedestrian 									|
| 3.3878004e-07			| Right-of-way									|
| 2.5473916e-08	      	| Narrow roads on right					 		|
| 2.4780041e-08		    | Roundabout mandatory      					|

Classifier correctly classifies the rest of signs with a very high probability. 
