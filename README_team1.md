## Project Name: *Financial Document Classification*

### Team Members: *Ramya Anand, Sunny Duggal, Jamil Abou, Heidi Ye, Leslie Wang*

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
This baseline Convolutional Neural Network (CNN) model is designed for image classification tasks. The architecture consists of the following layers,

•	It begins with a Conv2D layer that applies 32 filters of size 3x3 to the input images, which are expected to be of size 150x150 pixels with 3 color channels (RGB). The ReLU activation function introduces non-linearity to the model, helping it learn complex patterns. 

•	A MaxPooling2D layer follows, reducing the spatial dimensions of the feature maps and thus, the computational load. 

•	The model then flattens the feature maps into a one-dimensional vector, which is passed through a fully connected (Dense) layer with 64 neurons and ReLU activation, providing the model with the capacity to learn from the extracted features. 

•	Finally, a softmax output layer is used, making this model suitable for classification tasks.

•	Optimizer: The Adam optimizer is chosen with a learning rate of 0.001. Adam is a popular choice for training deep learning models as it adapts the learning rate for each parameter, combining the benefits of both AdaGrad and RMSprop.

•	Loss Function: The loss function is set to categorical_crossentropy, which is appropriate for multi-class classification problems where the output layer has a softmax activation function, and the labels are one-hot encoded. This function measures the discrepancy between the predicted class probabilities and the true labels.

•	Metrics: The accuracy metric is specified to monitor the proportion of correctly classified instances during training and validation. It provides a straightforward way to assess the model's performance.

  - **Architecture:** Standard CNN with Conv2D, MaxPooling2D, Flatten and Dense layer.

  - **Performance:** The baseline model showcases a perfect accuracy of 100% which is extremely good. The F1 score (0.76) also looks good which denotes the model is not overfitting and baselined in right direction. But, the validation accuracy is only around 70%. So, there is still room for improvement which can be improved by advancing the model with more layers or fine tuning of the hyper parameters. 

Overall, this configuration is suited for a classification task with multiple classes, ensuring that the model optimizes its weights to minimize the classification error. 

[***Building Advanced Model***](https://github.com/RamyaAnand27/team_project2/blob/team_project_2/src/3_AdvancedModel.ipynb): 
The advanced model is built using Keras, incorporating multiple convolutional, max pooling, and dropout layers for feature extraction and regularization. Batch normalization is applied after each convolutional layer to stabilize and accelerate the training process.
CNN Architecture
The model architecture consists of the following components:
Convolutional Layers: Four convolutional blocks with increasing filter sizes (32, 64, 128, and 256). Each block includes:
A Conv2D layer with ReLU activation.
A Batch Normalization layer to stabilize and accelerate training.
A MaxPooling layer to reduce the spatial dimensions.
A Dropout layer to prevent overfitting.
Fully Connected Layers: After flattening the feature maps from the convolutional layers:
A Dense layer with 512 neurons and ReLU activation.
A Dropout layer to further reduce overfitting.
An output Dense layer with 4 neurons and softmax activation to output the classification probabilities.
Model Compilation
The model is compiled using the Adam optimizer, with categorical crossentropy as the loss function, and accuracy as the evaluation metric.

Model Training
The model is trained for 20 epochs with a batch size of 8. The training process includes monitoring validation accuracy to gauge the model’s generalization ability. During training, the model's weights are updated to minimize the loss function and maximize accuracy on both the training and validation datasets.

Performance Evaluation
Post-training, the model's performance is evaluated using several metrics:
Accuracy: The accuracy is tracked during training and validation.
F1 Score: The F1 score is calculated on the validation set to measure the balance between precision and recall.
These metrics provide insights into the model’s effectiveness in correctly classifying images.
Results
The training and validation accuracy are plotted over the epochs to visualize the model's learning process. The final F1 score is also computed to quantify the model's performance on the validation set.

[***Cross Validation***](https://github.com/RamyaAnand27/team_project2/blob/team_project_2/src/Cross_validation.ipynb):
To apply K-Fold Cross-Validation to the best models perviously provided, we would need to split your data into k folds (e.g., 5 or 10) and then train and validate each model on k-1 folds while testing on the remaining fold. We repeat this process k times, each time using a different fold as the validation set. Finally, we average the performance metrics across all folds and do data visualization.
Step 1: Import Required Libraries
Step 2: Define a Function to Perform K-Fold Cross-Validation
Step 3: Perform K-Fold Cross-Validation for Model
Step 4: Evaluate and Make Models Data Visulization
1. plot_accuracies_per_fold(accuracies): This function creates a bar chart where each bar represents the accuracy achieved on the validation set for each fold.
If the accuracies across the folds are consistent, it suggests that the model is stable and performs reliably across different subsets of data.
If there are significant variations between folds, this could indicate that the model is sensitive to the specific data it’s trained on, which might be a sign of overfitting or that the data is not uniformly distributed.
2. plot_loss_per_epoch(histories): This function creates a line plot showing the validation loss per epoch for each fold. This helps in visualizing how well the model converges during training.
A steady decrease in validation loss across epochs is a good sign, indicating that the model is learning effectively.
If the validation loss fluctuates or increases in later epochs, it may suggest overfitting, where the model starts to perform well on the training data but poorly on the validation data.
Comparing the loss curves between folds can also help identify if one particular fold is performing worse than others, which might indicate issues with the data or the need for further model tuning.
3. plot_combined_accuracy_per_fold(histories): This function plots both the training and validation accuracy across epochs for each fold. It helps in understanding how the model is learning with each fold.
If the training accuracy improves while validation accuracy plateaus or decreases, this is another sign of overfitting.
If both training and validation accuracy improve consistently and are close to each other, it suggests the model is generalizing well.
Large gaps between training and validation accuracy may indicate that the model is memorizing the training data (overfitting) rather than learning patterns that generalize to new data.

Cross Validation Overall performance: The model exhibited strong performance across all folds, with validation accuracy ranging from 77.55% to 100%. Fold 1 showed a slightly lower validation accuracy compared to the other folds, which might suggest some variability in the data or model's sensitivity to certain data subsets. Folds 2 and 3 demonstrated perfect or near-perfect performance, indicating that the model is highly effective for the given task.

[***Optimized Model with Hyperparameter Tuning and Data Augmentation***](https://github.com/RamyaAnand27/team_project2/blob/team_project_2/src/Hyperparameter_Tuning_Image_Gen.ipynb):
In this phase, we focused on enhancing the baseline and advanced models by introducing several techniques aimed at improving classification accuracy and model robustness. The key improvements included:

•	Additional Convolutional Layers: We enhanced the model architecture by adding extra Conv2D layers, each followed by ReLU activation and MaxPooling layers. This allowed the model to capture more complex features and patterns from the images.

•	Dropout Layers: To combat overfitting and enhance generalization, we incorporated Dropout layers after the fully connected (Dense) layers. This technique randomly drops a fraction of the neurons during training, reducing the model's tendency to become too tailored to the training data.

•	Hyperparameter Tuning: We employed Keras Tuner to optimize key hyperparameters, including learning rate, batch size, and the number of neurons in the dense layers. This optimization process enabled us to identify the best configuration, which yielded the highest accuracy on the validation set.

•	Data Augmentation: To further improve generalization, we utilized ImageDataGenerator for real-time data augmentation during training. This approach generated additional training examples through random transformations such as rotation, zoom, and horizontal flips, enhancing the model's robustness.
Model Summaries

- **Model 1: Tuned Model**

  - **Architecture:** Standard CNN with Conv2D, MaxPooling2D, and Dense layers.

  - **Key Features:** Configurable filters, kernel sizes, and optimizer choices (Adam or SGD).

  - **Performance:** This model, optimized using Keras Tuner, included additional convolutional layers but did not feature Dropout. It achieved a validation accuracy of approximately 80.65% and a test accuracy of 71.43%. It demonstrated balanced performance with a precision-recall F1 score around 0.71.

- **Model 2: Tuned Model with Dropout**

  - **Architecture:** Enhanced CNN with additional Dropout layers for regularization.

  - **Key Features:** Fixed hyperparameters, dropout layers to reduce overfitting.

  - **Performance:** Achieved the best balance between accuracy and generalization. It demonstrated improvements over the baseline with a training accuracy of 95.11%, a validation accuracy of 77.42%, and a test accuracy of 81.82%. The F1 score showed a balanced performance across different document types.

- **Model 3: Augmented Model with Data Augmentation**

  - **Architecture:** Similar to Model 2 but incorporates data augmentation techniques.

  - **Key Features:** Includes transformations such as rotation, shifts, shear, zoom, and horizontal flips.

  - **Performance:** Data augmentation did not result in significant improvements in model performance compared to Model 2.


The model with Dropout layers (Model 2) proved to be the most effective, outperforming both the baseline and the augmented models. Despite the application of data augmentation, the dropout-enhanced model achieved the most balanced and accurate results.


## Links to the video recordings

Jamil Abou: https://drive.google.com/file/d/1-C8eq_qkqb8bdXia3GfHLfiBGivz90Rj/view?usp=drive_link
