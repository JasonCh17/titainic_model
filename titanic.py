import pandas #ipython notebook
titanic = pandas.read_csv("titanic_train.csv")
#一：数据预处理
#缺少数值填充
# print (titanic.describe()) #看看是否有缺失的值
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median()) #使用均值titanic["Age"].median()来填充
#缺失字符填充
# print (titanic["Embarked"].value_counts())
titanic["Embarked"] = titanic["Embarked"].fillna('S') #使用最多字符S来填充

#字符转化为数值
# print titanic["Sex"].unique() #查看Sex有哪些可能性
# Replace all the occurences of male with the number 0.
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0 #修改loc(”male“行，"Sex"列)=0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

# print titanic["Embarked"].unique()
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2







#二：线性回归
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

# 特征选择
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# 模型
alg = LinearRegression()
kf = KFold(titanic.shape[0], n_folds=3, random_state=1) #.shape返回(m,n),.shape[0]指的是m，即多少个样本

predictions = [] #来放预测数据
for train, test in kf:
    train_predictors = (titanic[predictors].iloc[train,:])
    train_target = titanic["Survived"].iloc[train]
    alg.fit(train_predictors, train_target)
    test_predictions = alg.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions)

#评估
import numpy as np
predictions = np.concatenate(predictions, axis=0)
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0
accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
print (accuracy)




三：随机森林（优）
from sklearn.model_selection import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

#特征选择
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
#建立模型
alg = RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=4, min_samples_leaf=2)
#模型评估
kf = KFold(titanic.shape[0], n_folds=10, random_state=1) #指定KFold参数
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)

print(scores.mean())