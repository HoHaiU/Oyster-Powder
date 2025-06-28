import openpyxl
import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as grid_spec
import torch.optim as optim
import torch.nn as nn
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import warnings
from torch import tensor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
warnings.filterwarnings('ignore')
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score 
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.utils import resample
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression,LogisticRegression
from collections import defaultdict

import xgboost as xgb
from sklearn import model_selection, ensemble
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import  LabelEncoder
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
import time
import shap
data= pd.read_excel('D:\牡蛎流变代码输入数据\训练集牡蛎剪切速率——剪切应力曲线.xlsx')
data.head()
data.isna().sum()
data.info()
cols=data.columns
new_cols=['Shear rate','Shear stress','Cement varieties','Cement content','CP content','NS particle size','NS content','Water content','Total content']

for i in range(len(new_cols)):
  data.rename(columns={cols[i]:new_cols[i]},inplace=True)
sns.heatmap(data.corr(),annot=True)
plt.show()
from sklearn.model_selection import train_test_split
data['w/c']=data.iloc[:,7]/data.iloc[:,8]
data.info()
data.drop([data.columns[3],data.columns[7]],axis=1,inplace=True)
X,y=data.iloc[:,[0,2,3,4,5,6,7]].values, data.iloc[:,1].values
sc=StandardScaler()


X_train=sc.fit_transform(X)

X_train=torch.Tensor(X_train)
y_train=torch.Tensor(y)
data1=pd.read_excel('D:\牡蛎流变代码输入数据\测试集牡蛎剪切应力——剪切应力曲线.xlsx')
data1['w/c']=data1.iloc[:,7]/data1.iloc[:,8]
data1.info()
data1.drop([data1.columns[3],data1.columns[7]],axis=1,inplace=True)
X,y=data1.iloc[:,[0,2,3,4,5,6,7]].values, data1.iloc[:,1].values

X=sc.transform(X)
Xtest=torch.Tensor(X)
ytest=torch.Tensor(y)
def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5)).mean()
    return rmse
def mape(y_true, y_pred):
   y_true, y_pred = np.array(y_true), np.array(y_pred)
   return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate(y, predictions):
    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    r_squared = r2_score(y, predictions)
    return mae, mse, r_squared
models = pd.DataFrame(columns=["Model", "MAE", "MSE", "r2 Score", "RMSE (Cross Validated)"])
xgb = XGBRegressor(booster = 'gbtree', learning_rate = 0.01, max_depth = 12, n_estimators = 600,silent=1,random_state =7,colsample_bytree=0.85)

xgb.fit(X_train, y_train)
xgb.score(X_train, y_train)
print(xgb.score(X_train, y_train))
predictions = xgb.predict(Xtest)
mae, mse, r2 = evaluate(ytest, predictions)
rmse = rmse_cv(xgb)
predictions1=np.array(predictions,dtype=np.int64)
ytest=np.array(y,dtype=np.int64)
RMSE = np.sqrt(np.mean((predictions1 - ytest)**2))
y_true=ytest
y_pred=predictions1
mape_value = mape(y_true, y_pred)
print("MAPE:", mape_value)
print("MAE:", mae)
print("MSE:", mse)
print("r2 Score:", r2)
print("RMSE:", RMSE)
shap.initjs()
Xtest_df = pd.DataFrame(Xtest.numpy())
X_sampled = Xtest_df.sample(1000, random_state=7)
#print(X_sampled)
# explain the model's predictions using SHAP values
# (same syntax works for LightGBM, CatBoost, and scikit-learn models)
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_sampled)
#print(shap_values)
# visualize the first prediction's explanation
#shap.force_plot(explainer.expected_value, shap_values[0,:], X_sampled.iloc[0,:])
# visualize the training set predictions
#shap.force_plot(explainer.expected_value, shap_values, X_train)
# summarize the effects of all the featurescolors = ["#9bb7d4", "#0f4c81"]
          
#shap_values1 = np.array(shap_values)
#shap_values1=torch.Tensor(shap_values1) 
#shap_values1 = pd.DataFrame(shap_values1.numpy())

#shap.dependence_plot(X_sampled.columns[5], shap_values, X_sampled, interaction_index=X_sampled.columns[5],alpha=1.0,show=False)
#plt.title("Cement varieties dependence plot",loc='left',fontfamily='serif',fontsize=15)
#plt.ylabel("SHAP value for the 'Cement varieties' feature")
#plt.show()
shap.summary_plot(shap_values, X_sampled)
shap.summary_plot(shap_values, X_sampled, plot_type="bar")
x=data1.iloc[:,0].values
y=torch.Tensor(predictions)
print(y)
plt.scatter(x,y)
plt.ylabel('Shear stress')
plt.xlabel('Shear rate')
plt.title('Shear rate-Shear stress')
plt.show()
with open("D:\牡蛎python代码\output_XGBoost剪切应力.xslx", "w") as f:
   for value in y:
       f.write(str(value.item()) + "\n")
