import os
import pickle
from features import *

pd.options.mode.chained_assignment = None

with open('model_dict.pickle' , 'rb') as f:
    model = pickle.load(f)
    
data_dir = '/hkfs/work/workspace/scratch/bh6321-energy_challenge/data/'
save_dir = 'forecasts/'

# dataloader
test_file = os.path.join(data_dir, 'test.csv')
valid_file = os.path.join(data_dir, 'valid.csv')

data_file = test_file if os.path.exists(test_file) else valid_file

df = pd.read_csv(data_file)
df = preprocess(df)
df = add_time_features(df, drop = True)
df.drop(columns='day_name', inplace=True)
df_list = add_ts_features(df, return_as_list=True)

forecast_dict = dict()
for df_tmp in df_list:
    city = df_tmp.city.unique()[0]
    print(city)
    df_tmp.drop(columns='city', inplace=True)
    pred = model[city].predict(df_tmp)
    forecast_dict[city] = pred
    
y_pred = np.array([])
for f in forecast_dict.values():
    f = f[:-24*7]
    for i in range(8424):
        y_pred = np.concatenate([y_pred, f[i : i+168]])
        
submission = pd.DataFrame(np.reshape(y_pred, (len(y_pred)//(24*7), 24*7)))

# save to csv
result_path = os.path.join(save_dir, 'forecasts.csv')
submission.to_csv(result_path, header=False, index=False)

print(f"Done! The result is saved in {result_path}")