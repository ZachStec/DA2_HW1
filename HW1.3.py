import numpy as np
from sklearn.linear_model import LinearRegression

#1.3a -------------------------------------------------------------------------------

def MAE(true_labels, pred_labels):

    #Convert input labels into arrays
    true_array = np.array(true_labels)
    pred_array = np.array(pred_labels)

    #Calculate Total Error by finding absolute value of the difference of every datapoint
    #and summing them up.
    total_error = np.sum(np.abs(true_array - pred_array))

    #Return the Function Value
    return total_error

#1.3b -------------------------------------------------------------------------------

#Values from 1.3b
x_values = [-2.69, -2.25, -1.76, -1.25, -0.36, 0.06, 0.30, 1.25, 2.36, 2.38]
y_values = [-0.44, -0.78, -0.98, -0.95, -0.35, 0.06, 0.30, 0.95, 0.71, 0.69]
y_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

MAE_val = MAE(y_values,y_pred)
print(f"The MAE value of the 1.3b function is {MAE_val:.2f}.")

#1.3c -------------------------------------------------------------------------------

def predict(x):

    #Reshape x_values array for Linear Regression
    x_values = np.array([-2.69, -2.25, -1.76, -1.25, -0.36, 0.06, 0.30, 1.25, 2.36, 2.38]).reshape(-1, 1)

    #Use Linear Regerssion from the previous data values
    model = LinearRegression()
    model.fit(x_values, y_values)

    #Obtain slope (m) and intercept (b) from created model
    m = model.coef_[0]
    b = model.intercept_

    #Use the slope, intercept, and inputted x value to predict a y value
    y = (m*x) + b
    return y

#Redefine X Values
x_values = [-2.69, -2.25, -1.76, -1.25, -0.36, 0.06, 0.30, 1.25, 2.36, 2.38]

#Calculate new predictions using the predict(x) function
new_y_pred = [predict(x) for x in x_values]

#Calculate the error with new prediction values
new_MAE_val = MAE(y_values,new_y_pred)
print(f"The MAE value of the 1.3c function is {new_MAE_val:.2f}.")