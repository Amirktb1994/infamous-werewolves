import numpy as np
import pandas as pd

def preprocess(df):
    df.rename(columns = {'Load [MWh]':'load', 'Time [s]':'time', 'City':'city'}, inplace = True)
    df.time = pd.to_datetime(df.time)
    
    return df


def encode(df, col, max_val):
    df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_val)
    df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_val)

    return df

def add_time_features(df, drop=True):
    df['hour'] = df.time.dt.hour
    df['day_name'] = df.time.dt.day_name()
    df['weekday'] = df.time.dt.weekday
    df['day'] = df.time.dt.day
    df['month'] = df.time.dt.month
    df['year'] = df.time.dt.year
    df['dayofyear'] = df.time.dt.dayofyear
    
    encode(df, 'hour', 23)
    encode(df, 'day', 31)
    encode(df, 'month', 12)
    encode(df, 'dayofyear', 365)
    
    if drop:
        df.drop(columns = ['time'], inplace = True)

    return df

    
def add_ts_features(df, return_as_list=False):
    CITIES = df.city.unique()
    ROLLING_WINDOW = 24

    df["load_sq"] = df["load"] ** 2

    df_tmp_list = []
    
    for city in CITIES:
        
        df_tmp = df.loc[df.city == city,:]

        df_tmp_len = df_tmp.shape[0]
        
        df_tmp.loc[:,"load_t-1"] = df_tmp.loc[:,"load"].shift(1)
        df_tmp.loc[:,"load_t-2"] = df_tmp.loc[:,"load"].shift(2)
        df_tmp.loc[:,"load_rmean"] = df_tmp.loc[:,"load"].rolling(ROLLING_WINDOW).mean()
        df_tmp.loc[:,"load_rstd"] = df_tmp.loc[:,"load"].rolling(ROLLING_WINDOW).std()
        df_tmp.loc[:,"load_rmin"] = df_tmp.loc[:,"load"].rolling(ROLLING_WINDOW).min()
        df_tmp.loc[:,"load_rmax"] = df_tmp.loc[:,"load"].rolling(ROLLING_WINDOW).max()
        
        df_tmp.fillna(method="backfill", inplace=True)

        assert df_tmp_len == df_tmp.shape[0]
        assert df_tmp.isnull().values.any() == False
        
        df_tmp_list.append(df_tmp)
    
    if return_as_list:
        return df_tmp_list
    else:
        return pd.concat(df_tmp_list)