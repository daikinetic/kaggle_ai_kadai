import pandas as pd
df = pd.read_csv('train.csv')
df.head(3)

df.isnull().sum()

L_onehot=["Education","City","Gender","EverBenched","PaymentTier"]
from sklearn.preprocessing import OneHotEncoder

encoder=OneHotEncoder(sparse=False)
encoder.fit(df[L_onehot].values)
encoded=encoder.transform(df[L_onehot].values)
df[encoder.get_feature_names(L_onehot)]=encoded

df["JoiningYear"]=df["JoiningYear"]-df["JoiningYear"].min()
df.head()

df_y=df["LeaveOrNot"]
df_x=df.drop(L_onehot+["ID","Gender_Male","EverBenched_No","LeaveOrNot"],axis=1)
y=df_y.values
x=df_x.values

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y)
Scaler=StandardScaler()
Scaler.fit(x_train)
x_train_scaled=Scaler.transform(x_train)
x_test_scaled=Scaler.transform(x_test)

model=SVC()
model.fit(x_train_scaled,y_train)

print("train: ",model.score(x_train_scaled,y_train))
print("test: ", model.score(x_test_scaled,y_test))

testdf=pd.read_csv('test.csv')
testdf.head()

encoded=encoder.transform(testdf[L_onehot].values)
testdf[encoder.get_feature_names(L_onehot)]=encoded # get_feature_names_out should be used for current version of Scikit-learn
testdf["JoiningYear"]=testdf["JoiningYear"]-testdf["JoiningYear"].min()
testdf_x=testdf.drop(L_onehot+["ID","Gender_Male","EverBenched_No"],axis=1)
xx=testdf_x.values

xx_scaled=Scaler.transform(xx)

yy=model.predict(xx_scaled)

Ans=pd.DataFrame(yy,index=testdf["ID"],columns=["Pred"])

Ans.to_csv("submission.csv")
