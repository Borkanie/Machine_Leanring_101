from re import I
from sklearn import datasets
import matplotlib.pyplot as plt  

# Defining main function
def main():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    # making subplots objects
    fig, ax = plt.subplots( len(X[0][:]),1)
    # Plot the points using matplotlib 
    for i in range(len(X[0][:])):
        column = []
        for j in range(len(X)):
            column.append(X[j][i])
        ax[i].plot(column, y,'*') 
    plt.show()  
    return 0


# Using the special variable 
# __name__
if __name__=="__main__":
    main()

