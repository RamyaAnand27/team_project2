## Project Name: *Financial Document Classification*

### Team Members: *Ramya Anand, Sunny Duggal, Jamil Abou, Heidi Ye, Leslie wang*

The second project of Team 1 is on 'Image Analysis and Classification of Financial and Personal Identification Documents'. For any kind of banking applications like opening of a bank account or applying for a mortgage etc.. there is always many document verification involved. Most of the time these documents are uploaded into the system and verfied manually. If this verification can be performed by a model, it could improve the operational efficiency of the whole process. The main intent of this project is to classify what kind of document is scanned or uploaded into the system to determine if it is a bank statement/ check/ salary slip/ utility bill and so on. This is premilinary step towards the identification and verification of the document acheived through the training and predictive analysis of the CNN model.

Below are the steps carried out in this project in order to successfully accomplish the above intent,

[***Image Data PreProcessing***](https://github.com/RamyaAnand27/team_project2/blob/team_project_2/src/1_DataPreprocess.ipynb):
In this phase, we focused on preparing and organizing our image dataset for the financial document classification task:

•	Data Loading and Organization: We began by setting up our base directory and exploring its structure. The code lists the subdirectories (classes) and a sample of images within each class, providing a clear overview of our dataset's organization.

•	Custom Data Loading Function: This function - iterates through class directories, loads and resizes images to a uniform size (150x150 pixels), converts images to numpy arrays and normalizes pixel values to the range [0, 1], handles potential errors during image loading, ensuring robustness, and assigns numerical labels to each class.

•	Data Splitting: Using the train_test_split function from scikit-learn, dataset was divided into training and testing sets, with 80% for training and 20% for testing to evaluate the model's performance on unseen data.

•	Data Reshaping and Visualization: We reshaped our label arrays and printed the shapes of our training and testing sets to confirm their dimensions. We also visualized a random sample of images from our training set, displaying them along with their corresponding class labels.

•	One-Hot Encoding: To prepare the labels for multi-class classification, one-hot encoding was applied using Keras' to_categorical function. This step is crucial for using categorical crossentropy loss in our models.

•	Validation Set Creation: We further split our training data to create a validation set, using a 80-20 split. This validation set will be used to tune our model's hyperparameters and prevent overfitting.

•	Data Export: Finally, we exported our preprocessed data (training, validation, and test sets) using pickle. This step allows us to easily load our prepared data in subsequent notebooks without repeating the preprocessing steps.

Through these preprocessing steps, we've transformed our raw image data into a format that's optimized for training deep learning models. The resulting datasets are well-structured, normalized, and split appropriately for training, validation, and testing purposes.

[***Building Baseline Model***](https://github.com/RamyaAnand27/team_project2/blob/team_project_2/src/2_BaselineModel.ipynb): 
This baseline Convolutional Neural Network (CNN) model is designed for image classification tasks. It begins with a Conv2D layer that applies 32 filters of size 3x3 to the input images, which are expected to be of size 150x150 pixels with 3 color channels (RGB). The ReLU activation function introduces non-linearity to the model, helping it learn complex patterns. A MaxPooling2D layer follows, reducing the spatial dimensions of the feature maps and thus, the computational load. The model then flattens the feature maps into a one-dimensional vector, which is passed through a fully connected (Dense) layer with 64 neurons and ReLU activation, providing the model with the capacity to learn from the extracted features. Finally, a softmax output layer is used, making this model suitable for classification tasks.

    _Optimizer_: The Adam optimizer is chosen with a learning rate of 0.001. Adam is a popular choice for training deep learning models as it adapts the learning rate for each parameter, combining the benefits of both AdaGrad and RMSprop.

    _Loss Function_: The loss function is set to categorical_crossentropy, which is appropriate for multi-class classification problems where the output layer has a softmax activation function, and the labels are one-hot encoded. This function measures the discrepancy between the predicted class probabilities and the true labels.

    _Metrics_: The accuracy metric is specified to monitor the proportion of correctly classified instances during training and validation. It provides a straightforward way to assess the model's performance.

Overall, this configuration is suited for a classification task with multiple classes, ensuring that the model optimizes its weights to minimize the classification error. The baseline model showcases a perfect accuracy of 100% which is extremely good. The F1 score (0.76) also looks good which denotes the model is not overfitting and baselined in right direction. But, the validation accuracy is only around 70%. So, there is still room for improvement which can be improved by advancing the model with more layers or fine tuning of the hyper parameters. 