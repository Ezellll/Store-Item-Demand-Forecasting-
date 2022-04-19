# Store Item Demand Forecasting

############################################################
# İş Problemi
#############################################################

# Bir mağaza zinciri, 10 farklı mağazası ve 50 farklı ürünü için 3 aylık bir talep tahmini istemektedir.

########################
# Veri Seti Hikayesi
#########################

# Bu veri seti farklı zaman serisi tekniklerini denemek için sunulmuştur.
# Bir mağaza zincirinin 5 yıllık verilerinde 10 farklı mağazası ve 50 farklı ürünün bilgileri yer almaktadır.

# 4 Değişken 958023 Gözlem

# date ---> Satış verilerinin tarihi (Tatil efekti veya mağaza kapanışı yoktur.)
# store ----> Mağaza ID’si(Her birmağazaiçineşsiznumara)
# Items ---> Ürün ID’si(Her bir ürün için eşsiz numara )
# Sales ---> Satılan ürün sayıları (Belirli bir tarihte belirli bir mağazadan satılan ürünlerin sayısı.)


##########################################
# Görev :Aşağıdaki zaman serisi ve makine öğrenmesi tekniklerini kullanarak ilgili mağaza
# zinciri için 3 aylık bir talep tahmin modeli oluşturunuz.
#########################################


# •RandomNoise
# •Lag/ShiftedFeatures
# •Rolling MeanFeatures
# •Exponentially Weighted MeanFeatures
# •Custom Cost Function(SMAPE)
# •LightGBM ile Model Validation


import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


########################
# Loading the data
########################

train = pd.read_csv('Dataset/train.csv', parse_dates=['date'])
test = pd.read_csv('Dataset/test.csv', parse_dates=['date'])

# Test ve Train seti birleştirrilerek Data Preprocessing işlemleri ve değişken üretme işlemleri
# iki veri içinde yapılmış olur.
df = pd.concat([train, test], sort=False)

#####################################################
# Keşifci Veri Analizi (EDA)
#####################################################

df["date"].min(), df["date"].max()
train["date"].max()
test["date"].min(),test["date"].max()
# (Timestamp('2018-01-01 00:00:00'), Timestamp('2018-03-31 00:00:00'))

check_df(df)


df[["store"]].nunique()
df[["item"]].nunique()
df.groupby(["store"])["item"].nunique()
df.groupby(["store", "item"]).agg({"sales": ["sum"]})
df.groupby(["store", "item"]).agg({"sales": ["sum", "mean", "median", "std"]})


#####################################################
# FEATURE ENGINEERING
#####################################################

# Makine öğrenmesi ile  zaman serisi yapmak için
# Bir seri kendisnden önceki değerlerden etkilenir
# gecikmeler serinin level'ı verir
# Trend, mevsimsellik Bunların hepsini yeni değişkenlerle temsil ediyor olmamız
# gerekli

def create_date_features(df):
    df['month'] = df.date.dt.month
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    df['week_of_year'] = df.date.dt.weekofyear
    df['day_of_week'] = df.date.dt.dayofweek
    df['year'] = df.date.dt.year
    df["is_wknd"] = df.date.dt.weekday // 4
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)
    return df

df = create_date_features(df)

df.groupby(["store", "item", "month"]).agg({"sales": ["sum", "mean", "median", "std"]})


########################
# 1-) Random Noise
########################

# Üretecek olduğumuz gecikme featureları bağımlı değişken üzerinden üretilkecek
# aşırı öğrenmenin önüne geçmek için random noise oluşturulur
def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))


########################
# 2-) Lag/Shifted Features
########################
# Geçmiş dönem satış sayılarına göre yeni değişken türeticez

# Sıralı olması gerekli
df.sort_values(by=['store', 'item', 'date'], axis=0, inplace=True)

# geçmiş gerçek değerler
def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

# İlgilendiğimiz 3 aylık satış olduğu için 3 ay önceki ve katlarının değerlerine baktık
df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])

check_df(df)



###########################
# Rolling MeanFeatures
#############################


def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe

# En anlamlıları getirdik
# tahmin yapacağımız periyodun 1 yıl öncesinin ortalaması
df = roll_mean_features(df, [365, 546])# İş döngüsüne göre yapılacak


#####################################
# Exponentially Weighted MeanFeatures
######################################

def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            # Dinamik isimlendirme yapabilmek için
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]

df = ewm_features(df, alphas, lags)
check_df(df)

########################
# One-Hot Encoding
########################

df = pd.get_dummies(df, columns=['store', 'item', 'day_of_week', 'month'])

check_df(df)

#############################################
# Converting sales to log(1+sales)
#############################################

# iterasyon sürceği varsayımıyla bağımlı değişken standartlaştırılabilir
# Train süresinin daha hızlı olması
df['sales'] = np.log1p(df["sales"].values)

check_df(df)

############################
# CustomCostFunction(SMAPE)
#############################


def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val

# lgbm ile smape çağırma
def lgbm_smape(preds, train_data):
    labels = train_data.get_label()# gerçek değerler
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False
# SMAPE: Symmetric mean absolute percentage error (adjusted MAPE) --> ne kadar düşük
# o kadar iyi.


##################################
# LightGBM ile Model Validation
##################################

#######################################
# Time-Based Validation Sets
#######################################
train["date"].max()
train = df.loc[(df["date"] < "2017-01-01"), :]
val = df.loc[(df["date"] >= "2017-01-01") & (df["date"] < "2017-04-01"), :]

cols = [col for col in train.columns if col not in ['date', 'id', "sales", "year"]]# year ---> veri setinde birkaç yıl olduğu için gereksiz kayda değer bir veri değil

Y_train = train['sales']
X_train = train[cols]

Y_val = val['sales']
X_val = val[cols]

Y_train.shape, X_train.shape, Y_val.shape, X_val.shape

"""
from lightgbm import LGBMRegressor
lgbm_model = LGBMRegressor(random_state=17)
lgbm_model.get_params()
lgbm_params = {"learning_rate": [0.01,0.02, 0.1],
               "n_estimators": [1000,10000],
               "colsample_bytree": [0.5, 0.7, 0.8,1],
               "num_leaves":[5,10,20],
               "max_depth":[4,5]}
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X_train, Y_train)
"""

# Optimize edilmiş hiperparametre değerleri
lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 10000,
              'early_stopping_rounds': 200,
              'nthread': -1}

lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape,
                  verbose_eval=100)

y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(Y_val))



#################################
# Feature Importance
#################################

def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))
    return feat_imp

plot_lgb_importances(model, num=30, plot=True)

feat_imp = plot_lgb_importances(model, num=200)

importance_zero = feat_imp[feat_imp["gain"] == 0]["feature"].values

imp_feats = [col for col in cols if col not in importance_zero]
len(imp_feats)


########################
# Final Model
########################

train = df.loc[~df.sales.isna()]
Y_train = train['sales']
X_train = train[cols]


test = df.loc[df.sales.isna()]
X_test = test[cols]

lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}

lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)


test_preds = final_model.predict(X_test, num_iteration=model.best_iteration)


########################
# Submission File
########################

test.head()

submission_df = test.loc[:, ["id", "sales"]]
submission_df['sales'] = np.expm1(test_preds)

submission_df['id'] = submission_df.id.astype(int)

submission_df.to_csv("submission_demand.csv", index=False)

# Kaggle sonuçları
# Private Score: 12.89467
# Public Score : 14.25484


