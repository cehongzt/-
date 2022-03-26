# @Time : 2022/1/22 11:56
# @Author : hongzt
# @File : xgbt
import pandas as pd
import numpy as np
import warnings

from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt
train= pd.read_csv('F:\\database\\cfst\\cacfsttraint.csv')
dem=train[["D (mm)","t (mm)","Le (mm)","fy (MPa)","fc (MPa)"]].values
object=train[["N Test (kN)"]].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_trian,y_test=train_test_split(dem,object,test_size=0.3,shuffle=True)
from sklearn.metrics import mean_squared_error
import xgboost as xgbt
from sklearn.model_selection import KFold
kf=KFold(n_splits=10)
rmse=[]

for train_indices,test_indices in kf.split(dem):
    X_train,X_test=dem[train_indices],dem[test_indices]
    Y_train, Y_test = object[train_indices], object[test_indices]
    xgbtRLR=xgbt.XGBRFRegressor()
    xgbtRLR.fit(X_train,Y_train)
    y_pred = xgbtRLR.predict(X_test)
    RSM=np.sqrt(mean_squared_error(Y_test,(y_pred)))
    rmse.append(RSM)
print(rmse)
print(f'average rmse:{np.mean(rmse)}')
xgbtRLR.fit(dem,object)
test= pd.read_csv('F:\\database\\cfst\\test.csv')
test_pred=xgbtRLR.predict(test.values)
result_df=pd.DataFrame(columns=["n"])
result_df["n"]=test_pred
result_df.to_csv("xbt_base.csv",index=None,header=True)

result_df["n"].plot(figsize=(16,8))
x=max(object)

y=max(test_pred)

plt.show()
