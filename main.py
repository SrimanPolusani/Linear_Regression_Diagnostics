# Import Statements
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


class Linear_Regression_Diagnoser:
    def __init__(self):
        # Loading the csv data using numpy module
        self.data = np.loadtxt("data_w3_ex1.csv", delimiter=',')  # data.shape = (50, 2)
        self.x_train, self.y_train, \
        self.x_cv, self.y_cv, \
        self.x_test, self.y_test = self.split_data()
        # Initialize lists containing the lists, models, and scalers
        self.train_mses = []
        self.cv_mses = []
        self.degrees = []

    def data_cleaner(self):
        # Separating features(x) and targets(y)
        x = self.data[:, 0]  # x.shape = (50,)
        y = self.data[:, 1]  # x.shape = y.shape

        # Changing 1-D arrays into 2D which helps in the code moving forward
        x = np.expand_dims(x, axis=1)  # x.shape = (50, 1)
        y = np.expand_dims(y, axis=1)  # x.shape = y.shape
        return x, y

    def split_data(self):
        x, y = self.data_cleaner()
        # Creating x_ and y_ temporary variables to split x and y into three sets
        # [training(train) = (60%), cross-verification(cv) = (20%) and testing(test) = (20%)]
        x_train, x_, y_train, y_ = train_test_split(x, y, test_size=.40, random_state=1)
        x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=.50, random_state=1)

        # Deleting temporary variables
        del x_, y_

        print(
            "shapes: x_train ={}, y_train = {}, x_cv = {}, y_cv = {}, x_test = {}, y_test = {}".format(
                x_train.shape, y_train.shape, x_cv.shape, y_cv.shape, x_test.shape, y_test.shape
            )
        )  # (30, 1), (30, 1), (10, 1), (10, 1), (10, 1), (10, 1) respectively

        return x_train, y_train, x_cv, y_cv, x_test, y_test

    def build_and_run(self):
        models = []
        scalers = []
        for degree in range(1, 11):
            self.degrees.append(degree)
            # Adding polynomial features
            poly = PolynomialFeatures(degree, include_bias=False)
            x_train_mapped = poly.fit_transform(self.x_train)

            # Feature scaling
            scaler_poly = StandardScaler()
            x_train_mapped_scaled = scaler_poly.fit_transform(x_train_mapped)
            scalers.append(scaler_poly)

            # Training the model
            model = LinearRegression()
            model.fit(x_train_mapped_scaled, self.y_train)
            models.append(model)

            # Compute the MSE
            yhat = model.predict(x_train_mapped_scaled)
            train_mse = mean_squared_error(self.y_train, yhat) / 2
            self.train_mses.append(train_mse)

            # Add polynomial features and feature scaling for cross-verification data
            poly = PolynomialFeatures(degree, include_bias=False)
            x_cv_mapped = poly.fit_transform(self.x_cv)
            x_cv_mapped_scaled = scaler_poly.transform(x_cv_mapped)

            # compute the MSE of cross-verification
            yhat_cv = model.predict(x_cv_mapped_scaled)
            cv_mse = mean_squared_error(self.y_cv, yhat_cv) / 2
            self.cv_mses.append(cv_mse)

            # print results
            print("{}".format(degree))
            print("Train MSE, degree {} polynomial = {}".format(degree, train_mse))
            print("CV MSE, degree {} polynomial = {}".format(degree, cv_mse))

    def visualize_results(self):
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(8, 5))
        # Plot 1 and 2
        ax.plot(self.degrees, self.train_mses, label='Train MSE', marker='o', linestyle='-')
        ax.plot(self.degrees, self.cv_mses, label='CV MSE', marker='s', linestyle='--')
        ax.set_title("Training data MSE vs Cross-verification data MSE")  # Setting Title
        ax.set_xlabel("Degrees")  # Setting Label
        ax.set_ylabel("MSE")
        ax.legend()  # Adding Legend
        plt.show()  # Show the plot

    @staticmethod
    def mse_from_scratch(targets, predictions):
        # Calculating MSE train from the scratch
        squared_error = 0
        for i in range(len(predictions)):
            yhat_minus_y = (predictions[i] - targets[i]) ** 2
            squared_error += yhat_minus_y
        mse = squared_error / (2 * len(predictions))
        return mse


# Instance of Linear_Regression_Diagnoser Object
diagnoster = Linear_Regression_Diagnoser()
diagnoster.build_and_run()
diagnoster.visualize_results()