​
1.1加载数据
sklearn的实现使用了numpy的arrays，所以需要使用numpy来读入数据

# -*- coding: utf-8 -*-

import numpy as np

import urllib.request

#直接从网络上读取数据,该数据有767个样本，每个样本有9列，最后一列为标签列

url="http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"

rawdata=urllib.request.urlopen(url)

#利用numpy.loadtxt从网络地址上读数据

dataset=np.loadtxt(rawdata,delimiter=',')#也可以先将数据下载到本地，然后从本地路径上下载数据,只要将rawdata换成本地路径就可以了

X=dataset[:,0:8]

y=dataset[:,8]


train= pd.read_csv('E:\\database\\cfst\\lincacfst.csv')
dem=train[["D (mm)","t (mm)","Le (mm)","fy (MPa)","fc (MPa)"]].values
object=train[["N Test (kN)"]].values

1.2特征工程


1.2.1标准化与区间放缩法

无量纲化使不同规格的数据转换到同一规格。常见的无量纲化方法有标准化和区间缩放法。标准化的前提是特征值服从正态分布，标准化后，其转换成标准正态分布。区间缩放法利用了边界值信息，将特征的取值区间缩放到某个特点的范围，例如[0, 1]等。

标准化

from sklearn.preprocessing import StandardScaler

 

#标准化，返回值为标准化后的数据

StandardScaler().fit_transform(iris.data)

区间放缩法

from sklearn.preprocessing import MinMaxScaler

 

#区间缩放，返回值为缩放到[0, 1]区间的数据

MinMaxScaler().fit_transform(iris.data)

1.2.2 归一化与正则化区别

　简单来说，标准化是依照特征矩阵的列处理数据，其通过求z-score的方法，将样本的特征值转换到同一量纲下。正则化是依照特征矩阵的行处理数据，其目的在于样本向量在点乘运算或其他核函数计算相似性时，拥有统一的标准，也就是说都转化为“单位向量”。

1.3训练集与验证集分类
from sklearn.model_selection import train_test_split
x_train,x_test,y_trian,y_test=train_test_split(dem,object,test_size=0.3,shuffle=True)

1.4模型训练
RLR=RandomForestRegressor()#引入模型
RLR.fit(X_train,Y_train)#模型训练

In [4]: from sklearn import linear_model

In [5]: linear_reg = linear_model.LinearRegression()

from sklearn import tree

tree_reg = tree.DecisionTreeRegressor()

In [7]: from sklearn import svm

In [8]: svr = svm.SVR()

In [11]: from sklearn import neighbors

In [12]: knn = neighbors.KNeighborsRegressor()

In [14]: from sklearn import ensemble

In [16]: rf =ensemble.RandomForestRegressor(n_estimators=20)#这里使用20个决策树

In [18]: ada = ensemble.AdaBoostRegressor(n_estimators=50)

In [20]: gbrt = ensemble.GradientBoostingRegressor(n_estimators=100)

lgbmRLR=lgb.LGBMRegressor()

1.5验证集预测
y_pred = RLR.predict(X_test)

1.6模型评估
from sklearn.metrics import mean_squared_error

RSM=np.sqrt(mean_squared_error(Y_test,((y_pred))))

包：sklearn.metrics

包含评分方法、性能度量、成对度量和距离计算

评估的方式可通过传入y_true,y_pred给metrics的相关接口，得到计算结果；也可以在交叉验证的时候指标评估标准。

分类结果度量

输入大多为y_true,y_pred

具体评价指标如下：

**metrics.confusion_matrix(y_true, y_pred[, …])**分类混淆矩阵

accuracy_score: 分类准确度

正确分类的样本数与检测样本总数(S)的比值

分类准确率这一衡量分类器的标准比较容易理解，但是它不能告诉你响应值的潜在分布，并且它也不能告诉你分类器犯错的类型。

metrics.precision_score(y_true, y_pred[, …])

**metrics.average_precision_score(y_true, y_score) **

metrics.recall_score(y_true, y_pred[, …])

metrics.f1_score(y_true, y_pred[, labels, …])

precision_recall_fscore_support: 计算精确度、召回率、f、支持度

metrics.roc_curve(y_true, y_score[, …])

metrics.auc

metrics.roc_auc_score(y_true, y_score[, …])

metrics.precision_recall_curve(y_true, …)

metrics.brier_score_loss(y_true, y_prob[, …])

jaccard_similarity_score: 计算jaccard相似度

杰卡德相似系数：两个集合A和B的交集元素在A，B的并集中所占的比例，称为两个集合的杰卡德相似系数，用符号J(A,B)表示。​

metrics.matthews_corrcoef(y_true, y_pred[, …])

metrics.cohen_kappa_score(y1, y2[, labels, …])

metrics.fbeta_score(y_true, y_pred, beta[, …])

hamming_loss: 计算汉明损失

zero_one_loss: 计算0-1损失

hinge_loss: 计算hinge损失

可参考机器学习中的损失函数 （着重比较：hinge loss vs softmax loss）

log_loss: 计算log损失

更多，可参考sklearn中的损失函数

classification_report: 分类报告

回归结果度量

mean_absolute_error: 平均绝对误差

mean_squared_erro 均方误差

metrics.mean_squared_log_error(y_true, y_pred):

metrics.median_absolute_error(y_true, y_pred):

explained_variance_score: 可解释方差的回归评分，与决定系数相比，这里是1-残差平方和/y的方差，而决定系数是1-残差的方差/比上y 的方差。总体上两者的趋势是一致的，但有细微差别，结果上一般决定系数的更接近1.

metrics.r2_score(y_true, y_pred[, …]): 也就通过说的决定系数r 2 r^2r

2

相关计算公式可参考：scikit-learn中拟合结果的评价指标

多标签的度量

当有多列标签时，实际上还没生成多列标签处理过，不知道这样处理有什么优势

coverage_error: 涵盖误差

label_ranking_average_precision_score: 基于排名的平均精度

metrics.label_ranking_loss(y_true, y_score)

聚类结果度量

adjusted_mutual_info_score: 调整的互信息得分

silhouette_score: 所有样本轮廓系数的平均值

silhouette_sample: 所有样本轮廓系数

官网上有更多指标

1.7模型参数调整
1.7.1交叉验证

包：sklearn.cross_validation

http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection

KFold(): 交叉验证迭代器，接收元素个数、fold数、是否清洗

LeaveOneOut(): 交叉验证迭代器

LeavePOut(): 交叉验证迭代器

LeaveOneLableOut()： 交叉验证迭代器

LeavePLableOut(): 交叉验证迭代器

交叉验证是利用训练集本身来测试模型精度，思路是将训练集分成n份，然后按一定比例取其中的几份（大头）作为真正的训练集来训练模型，其中的几份（少数）不参与训练来验证模型精度。然后循环这个过程，得到多组测试精度。所以在交叉验证器里就涉及到抽取测试集的比例问题。

LeaveOneOut()就是从n份(全校样本)里抽1份作为测试集，LeavePOut()就是从n份(全校样本)里抽p份作为测试集。

LeaveOneOut(n) 相当于 KFold(n_folds=n) ，当n等于全体样本数时（KFold的n_folds参数指的是将全体样本分成几份，每次拿其中的1份或几份做为测试集，而且它多次操作，会让测试覆盖整个数据集）；相当于LeavePOut(n, p=1)(LeavePOut中p参数指p个样本)。三者均会重复，让测试集覆盖全体样本。

而LeaveOneLableOut()和LeavePLableOut()与上面的区别在于，这里可以直接加入第三方的标签作为数据集的区分，也就是第三方标签已经把数据分成n份了。们的数据是一些季度的数据。那么很自然的一个想法就是把1,2,3个季度的数据当做训练集，第4个季度的数据当做测试集。这个时候只要输入每个样本对应的季度Label，就可以实现这样的功能。

from sklearn.model_selection import KFold
kf=KFold(n_splits=30)
rmse=[]




for train_indices,test_indices in kf.split(dem):
    X_train,X_test=dem[train_indices],dem[test_indices]
    Y_train, Y_test = object[train_indices], object[test_indices]
    lgbmRLR=lgb.LGBMRegressor()
    lgbmRLR.fit(X_train,Y_train)
    y_pred = lgbmRLR.predict(X_test)
    RSM=np.sqrt(mean_squared_error(np.log(Y_test),np.log((y_pred))))
    rmse.append(RSM)

1.7.2网格搜索

包：sklearn.grid_search

网络搜索是用来寻找最佳参数的。

GridSearchCV: 搜索指定参数网格中的最佳参数

ParameterGrid: 参数网络

ParameterSampler: 用给定分布生成参数的生成器

RandomizedSearchCV: 超参的随机搜索

通过best_estimator_get_params() 方法获取最佳参数

1.8模型保存与调用
[code=python] import joblib joblib.dump(filename='tiaocanLgbm.model2',value=dlgbnmodel) predietmodel= joblib.load('tiaocanLgbm.model1') [/code]
