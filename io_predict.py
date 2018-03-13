import numpy as np
import pandas as pd
import pickle
import datetime

def data_read(filename):
    return pd.read_csv(filename)


def generate_features(bid, year, month, roi_cpa, data):
    """
    :type bid: int
    :type year: int
    :type month: int
    :type data: pandas dataframe
    :rtype: list[floats]
    """
    feature_set = data[(data.year <= year) & (data.month <= month) & (data.bid == bid)]
    feature_set = feature_set.sort_values(by=['year', 'month'])
    feature_set = feature_set.tail(5)
    fuelx_conv_cnt = feature_set['fuelx_conv_cnt'].mean()

    sw_conv_cnt = feature_set['sw_conv_cnt'].mean()
    fuelx_sw_cr = feature_set['fuelx_sw_cr'].mean()
    sw_pgv_cnt = feature_set['sw_pgv_cnt'].mean()
    new_user_rate = feature_set['new_user_rate'].mean()
    total_conv = feature_set['total_conv'].mean()
    total_clk = feature_set['total_clk'].mean()
    total_imp = feature_set['total_imp'].mean()
    total_ctr = feature_set['total_ctr'].mean()
    total_cr = feature_set['total_cr'].mean()
    roi_cpa = feature_set['roi_cpa'].mean()
    is_roi = feature_set['is_roi'].mean()
    feature_list = []
    feature_list.append(year)
    feature_list.append(month)
    feature_list.append(fuelx_conv_cnt)
    feature_list.append(sw_conv_cnt)
    feature_list.append(fuelx_sw_cr)
    feature_list.append(sw_pgv_cnt)
    feature_list.append(new_user_rate)
    feature_list.append(total_conv)
    feature_list.append(total_clk)
    feature_list.append(total_imp)
    feature_list.append(total_ctr)
    feature_list.append(total_cr)
    feature_list.append(roi_cpa)
    feature_list.append(is_roi)
    with open('scaled_obj.pkl', 'rb') as f:
        sc = pickle.load(f)
    feature_list = sc.transform(np.array(feature_list).reshape(1,-1))
    
    return feature_list

def model_load(file):
    with open(file, 'rb') as f:
        model = pickle.load(f)
    return model

if __name__ == '__main__':
    file1 = 'transformed_dataset.csv'
    file2 = 'budget_calculator_model.pkl'
    df = data_read(file1)
    date=datetime.datetime.now()
    year = date.year
    month = date.month
    bid = int(input('give a bid here:',))
    roi_cpa = int(input('give a goal here:',))
    feat = generate_features(bid, year, month, roi_cpa, df)
    model = model_load(file2)
    io_predict = model.predict(feat)
    print(io_predict)
