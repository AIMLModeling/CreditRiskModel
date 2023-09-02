import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
credit_risk= pd.read_csv("UCI_credit_card.csv")
print(credit_risk.head())
df= credit_risk.copy()
# As we seen Column ID has no meaning here so, we will remove it
df.drop(["ID"], axis=1, inplace= True) #axis=1 -- column removal and inplcae= True --means change in the original data
# Lets check the statistics of data
print(df.describe())
# checking for missing values
print(df.isnull().sum())
df['EDUCATION'].replace({0:1,1:1,2:2,3:3,4:4,5:1,6:1}, inplace=True)
print(df.EDUCATION.value_counts())
df['MARRIAGE'].replace({0:1,1:1,2:2,3:3}, inplace=True)

print(df['MARRIAGE'].value_counts())
# Lets visualize the target column "default.payment.next.month"
plt.figure(figsize=(6,6))
ax = sns.countplot(df['default.payment.next.month'])
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')
plt.xticks([0,1], labels=["Not Deafaulted", "Defaulted"])
plt.title("Target Distribution")
plt.show()

# with age column
plt.figure(figsize=(6,6))
sns.distplot(df['AGE'], kde=True)
plt.xticks(rotation=0)
plt.ylabel('Count')
plt.title("Age distribution")
plt.show()
# With sex columns 
#Gender (1=male, 2=female)
plt.figure(figsize=(6,6))
ax = sns.countplot('SEX',hue='default.payment.next.month',data=df)
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')
plt.xticks([0,1], labels=["Male", "Female"])
#plt.xlabel({1:'Male',2:'Feamle'})
plt.title("Gender Distribution")
plt.show()
# With EDUCATION columns 
# (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
plt.figure(figsize=(10,6))
ax = sns.countplot('EDUCATION', hue='default.payment.next.month',data=df)
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')
plt.xticks([0,1,2,3],labels=["graduate school", "university",'high school','others'])
plt.title("Education Distribution")
plt.show()
# With MARRIAGE columns 
#
plt.figure(figsize=(10,6))
ax = sns.countplot('MARRIAGE',hue='default.payment.next.month',data=df)
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')
plt.xticks([0,1,2],labels=["Married", "single",'others'])
plt.title("MARRIAGE Distribution")
plt.show()
# With column 'limit_bal'
sns.distplot(df.LIMIT_BAL, kde=True)
plt.subplots(figsize=(20,10))
plt.subplot(231)
plt.scatter(x=df.PAY_AMT1, y=df.BILL_AMT2, c='r', s=1)
plt.xlabel('PAY_AMT1')
plt.ylabel('BILL_AMT2')

plt.subplot(232)
plt.scatter(x=df.PAY_AMT2, y=df.BILL_AMT3, c='g', s=1)
plt.xlabel('PAY_AMT2')
plt.ylabel('BILL_AMT3')
plt.title('Payment structure vs Bill amount in the last 6 months', fontsize=15)

plt.subplot(233)
plt.scatter(x=df.PAY_AMT3, y=df.BILL_AMT4, c='b', s=1)
plt.xlabel('PAY_AMT3')
plt.ylabel('BILL_AMT4')

plt.subplot(234)
plt.scatter(x=df.PAY_AMT4, y=df.BILL_AMT5, c='y', s=1)
plt.xlabel('PAY_AMT4')
plt.ylabel('BILL_AMT5')

plt.subplot(235)
plt.scatter(x=df.PAY_AMT5, y=df.BILL_AMT6, c='black', s=1)
plt.xlabel('PAY_AMT5')
plt.ylabel('BILL_AMT6')
plt.show()

# Independnet features
X = df.drop(['default.payment.next.month'], axis=1)
# Dependent feature
y = df['default.payment.next.month']
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
X= scaler.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.20,random_state=42)
from imblearn.over_sampling import SMOTE
from collections import Counter

# summarize class distribution
print("Before oversampling: ",Counter(y_train))
# define oversampling strategy
SMOTE= SMOTE()
# fit and apply the transform 
X_train,y_train= SMOTE.fit_resample(X_train,y_train)
# summarize class distribution
print("After oversampling: ",Counter(y_train))
