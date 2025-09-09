# ESI4611 | HW1 | 1.2
# Zach Stec

#1.2a -------------------------------------------------------------------------------

#Import the provided .csv
import pandas as pd
df = pd.read_csv('quartet.csv')

MasterStats = [] #Intialize a Display Vector for Results

#Loop through the 4 columns of the .csv, recording mean, variance, and correlation
for i in range(1,5):
    #Identify the Columns
    x_col = f'x{i}'
    y_col = f'y{i}'

    #Calculate Stats
    stats = {
        'Set #': i,
        'Mean (x)': df[x_col].mean(),
        'Mean (y)': df[y_col].mean(),
        'Variance (x)': df[x_col].var(),
        'Variance (y)': df[y_col].var(),
        'Correlation': df[x_col].corr(df[y_col])
    }

    #Combine all into the Display Vector
    MasterStats.append(stats)

#Convert Display Vector into Dataframe
Master_df = pd.DataFrame(MasterStats)
Master_df = Master_df.set_index('Set #')

#Display Results (2 decimal places as stated in question)
print(Master_df.round(2))

#1.2b -------------------------------------------------------------------------------

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

for i in range(1,5):
    #Identify the Columns (again)
    x_col = f'x{i}'
    y_col = f'y{i}'

    #Put all values in the selected columns into arrays
    X = df[[x_col]].values
    y = df[y_col].values

    #Create a Linear Regression Model and fit the created arrays
    model = LinearRegression()
    model.fit(X,y)

    #Obtain results from LR (Slope m and Intercept b)
    m = model.coef_[0]
    b = model.intercept_

    #Create "predictions" array, finding the y values from the regression line
    predictions = model.predict(X)

    #Then use "predictions" to find the average of the squared errors of the line at each data point
    MSE = mean_squared_error(y, predictions)

    #Finally, square root this value to obtain the RMSE for the set
    RMSE = np.sqrt(MSE)

    #Report each Set's Linear Regression Line and RMSE
    print(f"--- Set {i} ---")
    print(f"Equation: y = {m:.2f}x + {b:.2f}")
    print(f"Loss (RMSE): {RMSE:.2f}\n")

#1.2c -------------------------------------------------------------------------------

#See HW1_ZStec.pdf

#1.2d -------------------------------------------------------------------------------

#See HW1_ZStec.pdf