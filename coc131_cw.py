import numpy as np
import os
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from scipy import stats
from sklearn.manifold import LocallyLinearEmbedding

# Please write the optimal hyperparameter values you obtain in the global variable 'optimal_hyperparm' below. This
# variable should contain the values when I look at your submission. I should not have to run your code to populate this
# variable.
optimal_hyperparam = {}

class COC131:
    def q1(self, filename=None):
        """
        This function should be used to load the data. 
        To speed-up processing in later steps, lower resolution of the image to 32*32. 
        The folder names in the root directory of the dataset are the class names. 
        After loading the dataset, you should save it into an instance variable self.x (for samples) and self.y (for labels). 
        Both self.x and self.y should be numpy arrays of dtype float.

        :param filename: this is the name of an actual random image in the dataset. You don't need this to load the
        dataset. This is used by me for testing your implementation.
        :return res1: a one-dimensional numpy array containing the flattened low-resolution image in file 'filename'.
        Flatten the image in the row major order. The dtype for the array should be float.
        :return res2: a string containing the class name for the image in file 'filename'. This string should be same as
        one of the folder names in the originally shared dataset.
        """

        data_path = './EuroSAT_RGB'
        data_x = []
        data_y = []

        # Process the folders
        for className in sorted(os.listdir(data_path)): # Sorting ensures that the behaviour of processing is predictable and repeatable
            classFolder = os.path.join(data_path, className)
            if not os.path.isdir(classFolder):
                continue
            
            # Loop over all images in that classes folder
            for imageFile in sorted(os.listdir(classFolder)):
                imagePath = os.path.join(classFolder, imageFile)

                # Processing of images
                try:
                    image = Image.open(imagePath).convert('RGB') # Convert to RGB
                    imageResize = image.resize((32, 32)) # Resize to 32x32
                    imageArray = np.array(imageResize).astype(float).flatten() # Convert to numpy array

                    data_x.append(imageArray)
                    data_y.append(className)

                except Exception as e:
                    print(f"Error loading image {imagePath}: {e}")

        self.x = np.array(data_x, dtype=float)
        self.y = np.array(data_y)

        # Checks for filename in the class
        res1 = np.zeros(1)
        res2 = ''

        if filename is not None:
            for className in sorted(os.listdir(data_path)):
                classFolder = os.path.join(data_path, className)
                if not os.path.isdir(classFolder):
                    continue

                tempPath = os.path.join(classFolder, filename)

                # Process the image if it is found
                if os.path.exists(tempPath):
                    image = Image.open(tempPath).convert('RGB') # Convert to RGB
                    imageResize = image.resize((32, 32)) # Resize to 32x32
                    imageArray = np.array(imageResize).astype(float).flatten() # Convert to numpy array

                    res1 = imageArray
                    res2 = className
                    break

            else:
                print(f"Filename {filename} not found in dataset")


        return res1, res2

    def q2(self, inp):
        """
        This function should compute the standardized data from a given 'inp' data. The function should work for a
        dataset with any number of features.

        :param inp: an array from which the standardized data is to be computed.
        :return res1: a numpy array containing the standardized data with standard deviation of 2.5. The array should
        have the same dimensions as the original data
        :return res2: sklearn object used for standardization.
        """

        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(inp)
        res1 = np.array(standardized_data * 2.5)  # Adjusting standard deviation to 2.5
        res2 = scaler  # Returning the scaler object for potential reuse

        return res1, res2

    def q3(self, test_size=None, pre_split_data=None, hyperparam=None):
        """
        This function should build a MLP Classifier using the dataset loaded in function 'q1' and evaluate model
        performance. You can assume that the function 'q1' has been called prior to calling this function. This function
        should support hyperparameter optimizations.

        :param test_size: the proportion of the dataset that should be reserved for testing. This should be a fraction
        between 0 and 1.
        :param pre_split_data: Can be used to provide data already split into training and testing.
        :param hyperparam: hyperparameter values to be tested during hyperparameter optimization.
        :return: The function should return 1 model object and 3 numpy arrays which contain the loss, training accuracy
        and testing accuracy after each training iteration for the best model you found.
        """

        # Load dataset
        X, y = self.x, self.y  
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
        print(f"self.x: {self.x}, self.y: {self.y}")
        

        # Ensure y contains valid labels
        print("Unique labels in y:", np.unique(y))

        # Split dataset if not provided
        if pre_split_data:
            X_train, X_test, y_train, y_test = pre_split_data
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

        # Normalize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")    

        # Default hyperparameters (adjusted for better convergence)
        default_hyperparam = {
            'hidden_layer_sizes': (128, 64),  # More layers for better learning
            'activation': 'relu',
            'solver': 'adam',
            'learning_rate_init': 0.005,  # Increased learning rate
            'max_iter': 500,  # More iterations
            'early_stopping': True,  # Stops training if validation loss stops improving
            'n_iter_no_change': 20  # Number of iterations without improvement before stopping
        }
        
        # Use provided hyperparameters if any
        if hyperparam:
            default_hyperparam.update(hyperparam)

        # Train MLP Classifier
        model = MLPClassifier(**default_hyperparam, random_state=42, verbose=True)
        model.fit(X_train, y_train)

        # Extract training loss curve
        loss_curve = np.array(model.loss_curve_)

        # Compute training and testing accuracy
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))

        # Convert to NumPy arrays
        res1 = model
        res2 = loss_curve
        res3 = np.array([train_acc])
        res4 = np.array([test_acc])

        # Print accuracy for debugging
        print("Final Training Accuracy:", train_acc)
        print("Final Testing Accuracy:", test_acc)

        return res1, res2, res3, res4

    def q4(self):
        """
        This function should study the impact of alpha on the performance and parameters of the model. For each value of
        alpha in the list below, train a separate MLPClassifier from scratch. Other hyperparameters for the model can
        be set to the best values you found in 'q3'. You can assume that the function 'q1' has been called
        prior to calling this function.

        :return: A dictionary with alpha values, training accuracies, and testing accuracies for visualization.
        """

        alpha_values = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 50, 100]
        
        # Initialize lists to store results
        train_accuracies = []
        test_accuracies = []
        
        # Load dataset (assuming q1 has already been called and self.x, self.y are set)
        X, y = self.x, self.y
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
        
        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Default hyperparameters from q3
        default_hyperparam = {
            'hidden_layer_sizes': (128, 64),
            'activation': 'relu',
            'solver': 'adam',
            'learning_rate_init': 0.005,
            'max_iter': 500,
            'early_stopping': True,
            'n_iter_no_change': 20
        }

        # Iterate through alpha values
        for alpha in alpha_values:
            # Update hyperparameters with the current alpha value
            hyperparam = default_hyperparam.copy()
            hyperparam['alpha'] = alpha
            
            # Train MLP Classifier
            model = MLPClassifier(**hyperparam, random_state=42, verbose=False)
            
            # Split data into training and testing (80% train, 20% test)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            model.fit(X_train, y_train)
            
            # Compute accuracies
            train_acc = accuracy_score(y_train, model.predict(X_train))
            test_acc = accuracy_score(y_test, model.predict(X_test))
            
            # Append accuracies to lists
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
        
        # Store results in a dictionary for easy access
        res = {
            'alpha_values': alpha_values,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies
        }
        
        return res

    def q5(self):
        """
        This function should perform hypothesis testing to study the impact of using CV with and without Stratification
        on the performance of MLPClassifier. Set other model hyperparameters to the best values obtained in the previous
        questions. Use 5-fold cross validation for this question. You can assume that the function 'q1' has been called
        prior to calling this function.

        :return: The function should return 4 items - the final testing accuracy for both methods of CV, p-value of the
        test and a string representing the result of hypothesis testing. The string can have only two possible values -
        'Splitting method impacted performance' and 'Splitting method had no effect'.
        """

        # Load the dataset
        X, y = self.x, self.y
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

        # Check if the data is valid
        print("X shape:", X.shape)
        print("y shape:", y.shape)

        # Define the model with the best hyperparameters from previous questions
        model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
                            learning_rate_init=0.005, max_iter=500, early_stopping=True, 
                            n_iter_no_change=20, random_state=42)

        # 5-Fold Cross-Validation without Stratification (using KFold)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores_no_strat = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

        # Check the scores from KFold cross-validation
        print("Scores without Stratification:", scores_no_strat)

        # 5-Fold Cross-Validation with Stratification (using StratifiedKFold)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores_strat = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

        # Check the scores from StratifiedKFold cross-validation
        print("Scores with Stratification:", scores_strat)

        # Compute p-value using a t-test
        t_stat, p_value = stats.ttest_ind(scores_no_strat, scores_strat)

        # Check if the p-value is significant (commonly using alpha = 0.05)
        if p_value < 0.05:
            result_string = 'Splitting method impacted performance'
        else:
            result_string = 'Splitting method had no effect'

        # Final testing accuracies (mean scores for both methods)
        final_accuracy_no_strat = np.mean(scores_no_strat)
        final_accuracy_strat = np.mean(scores_strat)

        res1 = final_accuracy_no_strat
        res2 = final_accuracy_strat
        res3 = p_value
        res4 = result_string

        return res1, res2, res3, res4

    def q6(self):
        """
        This function should perform unsupervised learning using LocallyLinearEmbedding in Sklearn. You can assume that
        the function 'q1' has been called prior to calling this function.

        :return: The function should return the data you visualize.
        """

        # Use self.x for the input data
        X = self.x  # Assuming self.x contains the flattened 32x32 images from q1

        # Apply Locally Linear Embedding (LLE)
        lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2, random_state=42)
        transformed_data = lle.fit_transform(X)

        

        # Return the transformed data for potential further use
        return transformed_data
