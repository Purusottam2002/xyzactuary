import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error

car14 = pd.read_csv("cardata14.csv")
car18 = pd.read_csv("cardata18.csv")

#############################        merge two dataset      ##################################

car = pd.concat([car14,car18],ignore_index=True)

car = pd.DataFrame(car)

print(car.shape)

car.drop_duplicates(inplace=True)
print(car.shape)

print(car.columns)
claim_paid = car[['CLAIM_PAID']]
# claim_paid.to_csv("claim_paid.csv", index=False)

print(car.info())

print(car.isnull().sum())

car['INSR_BEGIN'] = pd.to_datetime(car['INSR_BEGIN'], format="%d-%b-%y")
car['INSR_END']   = pd.to_datetime(car['INSR_END'], format="%d-%b-%y")
car['policy_duration'] = (car['INSR_END'] - car['INSR_BEGIN']).dt.days
car.drop(['INSR_BEGIN', 'INSR_END','CLAIM_PAID','EFFECTIVE_YR','OBJECT_ID'],axis=1, inplace=True)

premium_missing = car[car['PREMIUM'].isna()]
# premium_missing.to_csv("premium_missing.csv", index=False)
car.dropna(subset=['PREMIUM'], inplace=True)


numeric_cols = car.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(12, 8))
sns.heatmap(car[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap (Numeric Features)")
plt.show()



numeric_cols = ['INSURED_VALUE', 'CCM_TON', 'PROD_YEAR', 'PREMIUM','SEATS_NUM','CARRYING_CAPACITY']  


car_log = car.copy()
for col in ['INSURED_VALUE','CCM_TON','PREMIUM','SEATS_NUM','CARRYING_CAPACITY']:
    car_log[col] = np.log1p(car_log[col])

plt.figure(figsize=(12, 8))
sns.heatmap(car_log[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap (Log-Transformed Features)")
plt.show()



############################     BOXPLOT ALL NUMERIC DATA       ########################################

numeric_cols = car.select_dtypes(include=[np.number]).columns

plt.figure(figsize=(15, len(numeric_cols)*3))

for i, col in enumerate(numeric_cols, 1):
    plt.subplot(len(numeric_cols), 1, i)
    sns.boxplot(x=car[col])
    plt.title(f"Boxplot of {col}")

plt.tight_layout()
plt.show()


#############################           train n test        ##########################

X = car.drop(columns = ['PREMIUM'])
y = car['PREMIUM']


X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.8, random_state=42)

top_makes = X_train['MAKE'].value_counts().nlargest(50).index
subset = X_train[X_train['MAKE'].isin(top_makes)]

plt.figure(figsize=(14, 6))
sns.boxplot(data=subset, x='MAKE', y=y_train.loc[subset.index])
plt.xticks(rotation=45)
plt.title("Premium Distribution by Vehicle Make (Training Set)")
plt.tight_layout()
plt.show()




# ============================= TRAIN PHASE =================================================



def hierarchical_impute(row, maps):
    for m in maps:
        if isinstance(m, pd.Series) and m.index.names is not None:
            key = tuple(row[k] for k in m.index.names)
            val = m.get(key)
        else:
            val = m 
        if pd.notna(val):
            return val
    return np.nan




ccm_maps = [
    X_train.groupby(['TYPE_VEHICLE', 'USAGE', 'MAKE'])['CCM_TON'].median(),
    X_train.groupby(['TYPE_VEHICLE', 'USAGE'])['CCM_TON'].median(),
    X_train.groupby(['TYPE_VEHICLE'])['CCM_TON'].median(),
    X_train['CARRYING_CAPACITY'].median()
]

X_train['CCM_TON'] = X_train.apply(
    lambda row: row['CCM_TON'] if pd.notna(row['CCM_TON']) else hierarchical_impute(row, ccm_maps),
    axis=1
)


seat_maps = [
    X_train.groupby(['TYPE_VEHICLE', 'USAGE', 'MAKE'])['SEATS_NUM'].median(),
    X_train.groupby(['TYPE_VEHICLE', 'USAGE'])['SEATS_NUM'].median(),
    X_train.groupby(['TYPE_VEHICLE'])['SEATS_NUM'].median()
]

X_train['SEATS_NUM'] = X_train.apply(
    lambda row: row['SEATS_NUM'] if pd.notna(row['SEATS_NUM']) else hierarchical_impute(row, seat_maps),
    axis=1
)

ccp_maps = [
    X_train.groupby(['TYPE_VEHICLE', 'USAGE', 'MAKE'])['CARRYING_CAPACITY'].median(),
    X_train.groupby(['TYPE_VEHICLE', 'USAGE'])['CARRYING_CAPACITY'].median(),
    X_train.groupby(['TYPE_VEHICLE'])['CARRYING_CAPACITY'].median(),
    X_train['CARRYING_CAPACITY'].median()
]

X_train['CARRYING_CAPACITY'] = X_train.apply(
    lambda row: row['CARRYING_CAPACITY'] if pd.notna(row['CARRYING_CAPACITY']) else hierarchical_impute(row, ccp_maps),
    axis=1
)




prod_maps = [
    X_train.groupby(['TYPE_VEHICLE', 'USAGE', 'MAKE'])['PROD_YEAR'].median(),
    X_train.groupby(['TYPE_VEHICLE', 'USAGE'])['PROD_YEAR'].median(),
    X_train.groupby(['TYPE_VEHICLE'])['PROD_YEAR'].median()
]

X_train['PROD_YEAR'] = X_train.apply(
    lambda row: row['PROD_YEAR'] if pd.notna(row['PROD_YEAR']) else hierarchical_impute(row, prod_maps),
    axis=1
)

X_train['MAKE'] = X_train['MAKE'].where(X_train['MAKE'].isin(top_makes), 'OTHER')
X_train['CCM_TON'] = np.log1p(X_train['CCM_TON'])
X_train['INSURED_VALUE'] = np.log1p(X_train['INSURED_VALUE'])

X_train_subset = X_train[['CARRYING_CAPACITY', 'MAKE', 'TYPE_VEHICLE', 'USAGE']].copy()
X_train_subset['Source'] = 'X_train'

# One-hot encode
X_train = pd.get_dummies(X_train, columns=['SEX','TYPE_VEHICLE','USAGE','INSR_TYPE','MAKE'], drop_first=True)

# =============================== TEST PHASE ==========================



X_test['CCM_TON'] = X_test.apply(
    lambda row: row['CCM_TON'] if pd.notna(row['CCM_TON']) else hierarchical_impute(row, ccm_maps),
    axis=1
)

X_test['SEATS_NUM'] = X_test.apply(
    lambda row: row['SEATS_NUM'] if pd.notna(row['SEATS_NUM']) else hierarchical_impute(row, seat_maps),
    axis=1
)

X_test['CARRYING_CAPACITY'] = X_test.apply(
    lambda row: row['CARRYING_CAPACITY'] if pd.notna(row['CARRYING_CAPACITY']) else hierarchical_impute(row, ccp_maps),
    axis=1
)


X_test['PROD_YEAR'] = X_test.apply(
    lambda row: row['PROD_YEAR'] if pd.notna(row['PROD_YEAR']) else hierarchical_impute(row, prod_maps),
    axis=1
)


X_test['MAKE'] = X_test['MAKE'].where(X_test['MAKE'].isin(top_makes), 'OTHER')
X_test['CCM_TON'] = np.log1p(X_test['CCM_TON'])
X_test['INSURED_VALUE'] = np.log1p(X_test['INSURED_VALUE'])

X_test_subset = X_test[['CARRYING_CAPACITY', 'MAKE', 'TYPE_VEHICLE', 'USAGE']].copy()
X_test_subset['Source'] = 'X_test'

# one-hot encoding with training
X_test = pd.get_dummies(X_test, columns=['SEX','TYPE_VEHICLE','USAGE','INSR_TYPE','MAKE'], drop_first=True)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)



############# MODEL TIME          #########################
y_train_log = np.log1p(y_train)
model = LinearRegression()
model.fit(X_train,y_train_log)
y_pred_log = model.predict(X_train)
print("Intercept (β0):", model.intercept_)
print("Coefficients (β1, β2, ...):", model.coef_)
print("Feature Names:", X_train.columns.tolist())
print("R²:", r2_score(y_train_log, y_pred_log))
print("Train RMSE:", root_mean_squared_error((y_train_log), (y_pred_log)))

y_test_log = np.log1p(y_test)
y_pred_log_test = model.predict(X_test)
print("Train predictions (log): min =", y_pred_log.min(), "max =", y_pred_log.max())
print("Test predictions (log): min =", y_pred_log_test.min(), "max =", y_pred_log_test.max())
print("Test R²:", r2_score(y_test_log, y_pred_log_test))
print("Test RMSE:", root_mean_squared_error((y_test_log),(y_pred_log_test)))

residuals = y_train_log - y_pred_log
sns.histplot(residuals, bins=50, kde=True)
plt.title("Residuals (Log Scale)")
plt.show()


plt.scatter((y_test_log), (y_pred_log_test), alpha=0.3)
plt.xlabel("Actual Premium (logₑ)")
plt.ylabel("Predicted Premium (logₑ)")
plt.title("Actual vs Predicted (Log Scale)")
plt.plot([min(y_test_log), max(y_test_log)], [min(y_test_log), max(y_test_log)], color='red', linestyle='--')
plt.show()

#######################      ############################




pred_car = pd.read_csv("premium_missing.csv")


X = pred_car.drop(columns = ['PREMIUM'])
Y = pred_car['PREMIUM']

print(pred_car.info())

print(pred_car.isnull().sum())

df_co = pd.concat([X_test_subset, X_train_subset])
df_co.to_csv('car_capa.csv', index=False)

def hierarchical_impute_p(row, maps):
    for m in maps:
        if isinstance(m, pd.Series) and m.index.names is not None:
            key = tuple(row[k] for k in m.index.names)
            val = m.get(key)
        else:
            val = m 
        if pd.notna(val):
            return val
    return np.nan

Xc = pd.read_csv("car_capa.csv")

ccpp_maps = [
    Xc.groupby(['TYPE_VEHICLE', 'USAGE', 'MAKE'])['CARRYING_CAPACITY'].median(),
    Xc.groupby(['TYPE_VEHICLE', 'USAGE'])['CARRYING_CAPACITY'].median(),
    Xc.groupby(['TYPE_VEHICLE'])['CARRYING_CAPACITY'].median(),
    Xc['CARRYING_CAPACITY'].median()
]

X['CARRYING_CAPACITY'] = X.apply(
    lambda row: row['CARRYING_CAPACITY'] if pd.notna(row['CARRYING_CAPACITY']) else hierarchical_impute_p(row, ccpp_maps),
    axis=1
)

X['CCM_TON'] = np.log1p(X['CCM_TON'])
X['INSURED_VALUE'] = np.log1p(X['INSURED_VALUE'])


categorical_cols = ['SEX','TYPE_VEHICLE','USAGE','INSR_TYPE','MAKE']
categorical_cols = [c for c in categorical_cols if c in X_train.columns and c in X.columns]

for col in categorical_cols:
    if col == 'MAKE':
        X[col] = X[col].where(X[col].isin(top_makes), 'OTHER')
    else:
        X[col] = X[col].where(X[col].isin(X_train[col].unique()), 'OTHER')

X = pd.get_dummies(X, columns=['SEX','TYPE_VEHICLE','USAGE','INSR_TYPE','MAKE'], drop_first=True)

X = X.reindex(columns=X_train.columns, fill_value=0)


print(X.isnull().sum())

predict_premium = model.predict(X)

print(predict_premium)