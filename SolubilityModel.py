import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt  
from sklearn.metrics import mean_squared_error, r2_score

def diffModels(y_pred,y_real):
    # Plot the points using matplotlib.
    plt.plot(range(len(y_pred)),y_pred, label = "predictie") 
    plt.plot(range(len(y_pred)),y_pred, label = "real") 
    plt.show()  
    # Calculate error.
    lr_train_mse = mean_squared_error(y_real, y_pred)
    lr_train_r2 = r2_score(y_real, y_pred)
    # We output the results:
    print("Train mean square error:" + str(lr_train_mse))
    print("Train root  error:" + str(lr_train_r2))

# Defining main function
def main():
    dataSet = pd.read_csv('solubility.csv')
    X = dataSet.drop(['logS'], axis=1)
    y = dataSet.logS
    # Split the data into train and test dataSet.
    # This setup sets the testing data to 20% of the overall data and the training data to be 80% of the overall data set.
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=105)
    # Train using linear regression.
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    # Here we make predicitons using the linear model.
    y_lr_train_pred = lr.predict(X_train)
    y_lr_test_pred = lr.predict(X_test)
    diffModels(y_lr_train_pred,y_train)
    diffModels(y_lr_test_pred,y_test)


    return 0


# Using the special variable 
# __name__
if __name__=="__main__":
    main()
