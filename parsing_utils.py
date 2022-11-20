import pandas as pd
import numpy as np


def make_stat_agg(df):
    
    
    date_cols = [col for col in df.columns if '2021-' in col]

    agg_features = ['month_argmax','month_mean','month_5','month_7','max_mean_diff','month_max_mean_diff','argmin',
                    'mean','month_8','var','std','month_var','month_std','month_max','max','month_4','argmax','month_argmin','month_6']

    dates_4 = [col for col in date_cols if "-04-" in col]
    dates_5 = [col for col in date_cols if "-05-" in col]
    dates_6 = [col for col in date_cols if "-06-" in col]
    dates_7 = [col for col in date_cols if "-07-" in col]
    dates_8 = [col for col in date_cols if "-08-" in col]
    
    
    df["mean"] = df[date_cols].values.mean(axis=1)
    df["min"] = df[date_cols].values.min(axis=1)
    df["max"] = df[date_cols].values.max(axis=1)
    df["var"] = df[date_cols].values.var(axis=1)
    df["std"] = df[date_cols].values.std(axis=1)
    df["argmax"] = np.argmax(df[date_cols].values, axis=1)
    df["argmin"] = np.argmin(df[date_cols].values, axis=1)
    df["max_mean_diff"] = df["max"] - df["mean"]
    
    month_sum = np.array([df[dates_4].values.sum(axis=1), df[dates_5].values.sum(axis=1),
                          df[dates_6].values.sum(axis=1), df[dates_7].values.sum(axis=1),
                          df[dates_8].values.sum(axis=1)]).T
    
    df["month_mean"] = month_sum.mean(axis=1)
    df["month_min"] = month_sum.min(axis=1)
    df["month_max"] = month_sum.max(axis=1)
    df["month_var"] = month_sum.var(axis=1)
    df["month_std"] = month_sum.std(axis=1)
    df["month_argmax"] = np.argmax(month_sum, axis=1)
    df["month_argmin"] = np.argmin(month_sum, axis=1)
    
    df["month_max_mean_diff"] = df["month_max"] - df["month_mean"]
    
    month_sum = pd.DataFrame(month_sum, columns=["month_4", "month_5", "month_6", "month_7", "month_8"])
    df = pd.concat([df, month_sum], axis=1)
    
    return df, agg_features


################################

def near_feild_fs(df):
    
    # Парсим статистики на основе числоа ближайщих соседенй в определенном диапазоне
    
    X_array = df['x'].values
    Y_array = df['y'].values
    #################### sum ###################
    
    f = lambda x : np.sum(abs(X_array - x) <= 0.05)
    df['x_n_near_05'] = df['x'].apply(f)
    f = lambda x : np.sum(abs(X_array - x) <= 0.01)
    df['x_n_near_01'] = df['x'].apply(f)
    
    df['x_n_near_01_on_05'] = df['x_n_near_01']/df['x_n_near_05'] 
    
    f = lambda x : np.sum(abs(Y_array - x) <= 0.01)
    df['y_n_near_01'] = df['y'].apply(f)
    f = lambda x : np.sum(abs(Y_array - x) <= 0.05)
    df['y_n_near_05'] = df['y'].apply(f)
    
    df['y_n_near_01_on_05'] = df['y_n_near_01']/df['y_n_near_05'] 
    
#     #################### mean ###################
    
    f = lambda x : np.mean(abs(X_array - x) <= 0.05)
    df['x_n_near_05_mean'] = df['x'].apply(f)

    f = lambda x : np.mean(abs(X_array - x) <= 0.01)
    df['x_n_near_01_mean'] = df['x'].apply(f)

    f = lambda x : np.mean(abs(Y_array - x) <= 0.01)
    df['y_n_near_01_mean'] = df['y'].apply(f)

    f = lambda x : np.mean(abs(Y_array - x) <= 0.05)
    df['y_n_near_05_mean'] = df['y'].apply(f)
    
    
    
def get_county_type(county):
    if "район" in county:
        return "район"
    if "округ" in county:
        return "округ"
    return "другое"
    
def get_state_type(state):
    if "область" in state:
        return "область"
    if "край" in state:
        return "край"
    return "другое"

def get_municipality_type(municipality):
    if "поселение" in municipality:
        return "поселение"
    if "сельсовет" in municipality:
        return "сельсовет"
    return "другое"

def get_adress(df):
    df['municipality'] = df["location"].apply(lambda x: x['address'].get('municipality', 'nan'))
    df['municipality_type'] = df["municipality"].apply(lambda county: get_municipality_type(county))
    df['county'] = df["location"].apply(lambda x: x['address'].get('county', 'nan'))
    df['county_type'] = df["county"].apply(lambda county: get_county_type(county))
    df['state'] = df["location"].apply(lambda x: x['address'].get('state', 'nan'))
    df['state_type'] = df["state"].apply(lambda state: get_state_type(state))
    df['ISO3166-2-lvl4'] = df["location"].apply(lambda x: x['address'].get('ISO3166-2-lvl4', 'nan'))
    df['region'] = df["location"].apply(lambda x: x['address'].get('region', 'nan'))
    df['country_code'] = df["location"].apply(lambda x: x['address'].get('country_code', 'nan'))
    
    
    #################################
    
from meteostat import Stations, Point, Daily
    
    
def generate_weather_fs(df):
    
    weather_cols = ["prcp", "snow", "wdir", "wspd", "wpgt", "tavg", "tmin", "tmax",  "pres", "tsun"]
    
    for date_col in tqdm.tqdm(date_cols):
        weather_dict = {wc: [] for wc in weather_cols}
        for i, row in df.iterrows():
            date = date_col.split('-')
            y,m,d = int(date[0]), int(date[1]), int(date[2])
            start = datetime(y, m, d)
            end = datetime(y, m, d)
            
            location = Point(row.x, row.y)
            
            stations = Stations()
            stations = stations.nearby(row.x, row.y)
            station = stations.fetch(10)

            data = Daily(station, start, end)
            data = data.fetch()
            data.fillna(0., inplace=True)
            
            # print(row.x, row.y, date, data)
            for wc in weather_cols:
                try:
                    weather_dict[wc].append(np.median(data[wc].values))
                except Exception as e:
                    print(e, data)
                    weather_dict[wc].append(0.)
                
        for wc in weather_cols:
            df[f"{date_col}_{wc}"] = weather_dict[wc]
            
            
