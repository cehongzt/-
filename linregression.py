# @Time : 2022/1/21 19:13
# @Author : hongzt
# @File : line回归
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt
train= pd.read_csv('F:\\database\\cfst\\cacfsttraint.csv')
test= pd.read_csv('F:\\database\\cfst\\test.csv')
#train.info()数据查看
#数据描述z=train.describe()
dem=train[["D (mm)","t (mm)","Le (mm)","fy (MPa)","fc (MPa)"]].values
object=train[["N Test (kN)"]].values

#划分数据集测试集
from sklearn.model_selection import train_test_split
x_train,x_test,y_trian,y_test=train_test_split(dem,object,test_size=0.3,shuffle=True)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
'''
LR=LinearRegression()
LR.fit(x_train,y_trian)
y_pred=LR.predict(x_test)



print(f'root mean square eoore:{np.sqrt(mean_squared_error(np.log(y_test),np.log(y_pred)))}')
'''
from sklearn.model_selection import KFold
kf=KFold(n_splits=5)
rmse=[]
for train_indices,test_indices in kf.split(dem):
    X_train,X_test=dem[train_indices],dem[test_indices]
    Y_train, Y_test = object[train_indices], object[test_indices]
    LR=LinearRegression(normalize=True)
    LR.fit(X_train,Y_train)
    y_pred = LR.predict(X_test)
    RSM=np.sqrt(mean_squared_error(Y_test,abs(y_pred)))
    rmse.append(RSM)

print(rmse)
print(f'average rmse:{np.mean(rmse)}')
