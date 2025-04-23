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
from itertools import product

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
        Build and evaluate an MLP Classifier, optionally performing hyperparameter optimization.

        :param test_size: float, fraction of data to use for testing (ignored if pre_split_data is used)
        :param pre_split_data: tuple of (X_train, X_test, y_train, y_test)
        :param hyperparam: dict of hyperparameter lists for grid search (e.g. {'hidden_layer_sizes': [(64,), (128, 64)]})
        :return: best model, loss curve (np.array), training accuracy (np.array), test accuracy (np.array)
        """

        # Load dataset
        X, _ = self.q2(self.x) 
        y = self.y
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

        if pre_split_data:
            X_train, X_test, y_train, y_test = pre_split_data
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Default training setup
        default_hyperparam = {
            'solver': 'sgd',
            'max_iter': 1,
            'warm_start': True,
        }

        param_grid = hyperparam if hyperparam else {
            'hidden_layer_sizes': [(64,), (128, 64)],
            'learning_rate_init': [0.001],
            'activation': ['relu']
        }

        classes = np.unique(y)
        n_epochs = 5

        # Track the best model
        best_model = None
        best_test_acc = 0
        best_loss_curve = None
        best_train_curve = None
        best_test_curve = None

        # Grid search over hyperparameters
        keys, values = zip(*param_grid.items())
        for combo in product(*values):
            current_params = dict(zip(keys, combo))
            current_params.update(default_hyperparam)

            model = MLPClassifier(**current_params, random_state=42)

            loss_curve = []
            train_acc_curve = []
            test_acc_curve = []

            for epoch in range(n_epochs):
                model.partial_fit(X_train, y_train, classes=classes)

                # Track performance per epoch
                if hasattr(model, "loss_"):
                    loss_curve.append(model.loss_)
                train_acc = accuracy_score(y_train, model.predict(X_train))
                test_acc = accuracy_score(y_test, model.predict(X_test))
                train_acc_curve.append(train_acc)
                test_acc_curve.append(test_acc)

            final_test_acc = test_acc_curve[-1]
            if final_test_acc > best_test_acc:
                best_test_acc = final_test_acc
                best_model = model
                best_loss_curve = np.array(loss_curve)
                best_train_curve = np.array(train_acc_curve)
                best_test_curve = np.array(test_acc_curve)


        return best_model, best_loss_curve, best_train_curve, best_test_curve

    def q4(self, test_size=None, pre_split_data=None, hyperparam=None):
        """
        This function should study the impact of alpha on the performance and parameters of the model. For each value of
        alpha in the list below, train a separate MLPClassifier from scratch. Other hyperparameters for the model can
        be set to the best values you found in 'q3'. You can assume that the function 'q1' has been called prior to
        calling this function.

        :return: A dictionary with alpha values, training accuracies, and testing accuracies for visualization.
        """
        
        # Load dataset
        X, _ = self.q2(self.x) 
        y = self.y
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

        if pre_split_data:
            X_train, X_test, y_train, y_test = pre_split_data
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Default training setup
        default_hyperparam = {
            'solver': 'sgd',
            'max_iter': 1,
            'warm_start': True,
            'hidden_layer_sizes': (128, 64),
            'learning_rate_init': 0.001,
            'activation': 'relu'
        }

        param_grid = hyperparam if hyperparam else {
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
        }

        classes = np.unique(y)
        n_epochs = 5

        results = {}  # Initialize the dictionary to store results

        # Grid search over hyperparameters (alphas)
        keys, values = zip(*param_grid.items())
        for combo in product(*values):
            current_params = dict(zip(keys, combo))
            current_params.update(default_hyperparam)

            model = MLPClassifier(**current_params, random_state=42)

            loss_curve = []
            train_acc_curve = []
            test_acc_curve = []

            for epoch in range(n_epochs):
                model.partial_fit(X_train, y_train, classes=classes)

                # Track performance per epoch
                if hasattr(model, "loss_"):
                    loss_curve.append(model.loss_)
                train_acc = accuracy_score(y_train, model.predict(X_train))
                test_acc = accuracy_score(y_test, model.predict(X_test))
                train_acc_curve.append(train_acc)
                test_acc_curve.append(test_acc)

            final_test_acc = test_acc_curve[-1]
            alpha = combo[0]  # Get the current alpha value from the product

            # Store results in a dictionary
            results[alpha] = {
                'loss_curve': np.array(loss_curve),
                'train_acc_curve': np.array(train_acc_curve),
                'test_acc_curve': np.array(test_acc_curve),
            }

        return results

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

        X = self.x

        lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2, random_state=42)
        data = lle.fit_transform(X)

        return data
