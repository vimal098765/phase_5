import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

a=pd.read_csv("./dataset/covid_19_india.csv")#load the dataset
print(a.head())
a.dropna(inplace=True)
a.drop_duplicates(inplace=True)
a['Date']=pd.to_datetime(a['Date'])
a['Time']=pd.to_datetime(a['Time'],format='%I:%M %p')
a['Cured']=pd.to_numeric(a['Cured'])
a['Deaths']=pd.to_numeric(a['Deaths'])
a['Confirmed']=pd.to_numeric(a['Confirmed'])
a=a.rename(columns={'State/UnionTerritory': 'State'})
a=a[['Date','State','Cured','Deaths','Confirmed']]
X=a[['Cured', 'Deaths']]
y=a['Confirmed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model=LinearRegression()
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
mse=mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
a.to_csv("./dataset/new.csv",index=False)