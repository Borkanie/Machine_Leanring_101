import matplotlib
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt  

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])

plt.title( "Inputs") 

# Plot the points using matplotlib 
plt.plot(x, y) 

model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print(f"coefficient of determination: {r_sq}")
print(f"intercept: {model.intercept_}")
print(f"slope: {model.coef_}")

plt.title("Outputs") 

newY=[value*model.coef_ for value in x]

# Plot the points using matplotlib 
plt.plot(x, newY) 
plt.legend(['Original','Regression'])
plt.show() 
