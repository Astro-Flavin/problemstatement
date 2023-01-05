import numpy as np
import pandas as pd
from sklearn import linear_model as lm

def predict(time):
    data = pd.read_csv("trainingdata.txt", header=None, names=["x", "y"])
    x = data["x"]
    y = data["y"]
        # Notice that max value to y is 8.0
    x = np.array(x[y[y < 8.0].index], float)
    y = np.array( y[y < 8.0], float)
    model = lm.LinearRegression()
    model.fit(x.reshape(-1,1), y)
    return model.predict(np.array([time]).reshape(-1,1))[0]

if __name__ == '__main__':
    timeCharged = float(input().strip())
        # from visualization i noticed that laptop fully charged at 4.0 hours
    if 4.0 - timeCharged <= 0.0:
        print(8.0)
    else:
        print('{:.2f}'.format(round(predict(timeCharged),2)))
