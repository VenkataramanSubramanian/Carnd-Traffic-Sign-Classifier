Traffic Sign Recognition

Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:
	1.Load the data set 
	2.Explore, summarize and visualize the data set
	3.Design, train and test a model architecture
	4.Use the model to make predictions on new images
	5.Summarize the results

1.Load the data set

	The dataset is loaded as a pickle file and the data's as a columnn of a pandas dataframe with both fetarues and labels as numpy array.	
	
2.Data Set Summary & Exploration

	The size of the dataset are as follows

	Number of training examples = 34799
	Number of Validation examples = 4410
	Number of testing examples = 12630
	Image data shape = (32, 32)
	Number of classes = 43

3.Design and Test a Model Architecture

	The Image from the dataset is been preprocessed.
	
	I have converted the images into  grey scale as rgb is not required as colour is not an important factor in this classification, rather its the shape.
	so converting it to grey scale would mean it is easier to compute
	
	Then The data is normalized, to obtain zero mean and equal variance
	
	These are the pre-processing techniques used in the solution.
	
	My final model consisted of the following layers:

	| Layer         		|     Description	        					| 
	|:---------------------:|:---------------------------------------------:| 
	| Input         		| 32x32x1 RGB image   							| 
	| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
	| RELU					|												|
	| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|	
	| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16.  |
	| RELU					|												|
	| Max pooling	      	| 2x2 stride,  outputs 5x5x16					|
	| Flatten				| 5x5x16 to 400									|
	| Fully connected		| Input 400 ouput 200        					|
	| RELU					|         										|
	| Dropout				| Keep Prob - Placeholder						|
	| Fully connected		| Input 200 ouput 100        					|
	| RELU					|         										|
	| Dropout				| Keep Prob - Placeholder						|
	| Fully connected		| Input 100 ouput 43        					|
	| Softmax				| Activation

	The training data in each epoch is iterated for a batch of its date based on the batchsize and gets trained.
	After every epoch, the validation set is executed in the same process to predict its accuracy
	
	The model is trained using AdamOptimizer
	
	With Hyper parameters
	
	EPOCHS = 15
	BATCH_SIZE = 128
	learning rate = 0.005
	Keep_prob=0.7 for training and 1.0 for validation
	
	The training data in each epoch is iterated for a batch of its date based on the batchsize and gets trained.
	After every epoch, the validation set is executed in the same process to predict its accuracy
	
	The model is trained and validated with an accuracy of 0.943
	
	Then, Finnaly the model is tested with an accuracy of 0.932


	My final model results were:
		validation set accuracy of 0.94 
		test set accuracy of 0.924

	The architecture that was choosen is LeNet-5. It is a commonly known architecture.
	This architecture implements CNN which is very efficent with less memory and time consuming.
	The Final accuracy of both validation and testing proves that it is a best fit for this traffic sign classification probelm
 
4. Test a Model on New Images

	New images were taken from web to under go preprocess and predict the data.
	
	Then they ouput were predicted.

	Here are the results of the prediction:

	| Image			        |     Prediction	        					| 
	|:---------------------:|:---------------------------------------------:| 
	| Turn left ahead	    | Turn left ahead  								| 
	| No Passing     		| No passing 									|
	| Road work				| Speed limit (30km/h)							|
	| stop	      			| Stop							 				|
	| Yield					| Yield      									|


	The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set.
	
	The Model's softmax prediction are fairly close 
	
	For image 1 the correct prediction is 0.16 where as 0.1, so since there are 43 images in classfication it is fairly accurate
	There rest of the images accuracy also suggest that it has predicted it well.
	

5.Summarizing

	This model is meeting the requirements for the project submission 


	
