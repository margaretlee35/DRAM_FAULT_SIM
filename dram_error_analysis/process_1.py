import pandas as pd
import numpy as np
import pickle
import dask.dataframe as dd
import os

from utils_new import groupby_machine_informations, make_decision, generate_min_max, normalize_err_time, split_transient_and_permanent, extract_feature_vector

def load_csv(filename):
    '''Load csv and return a dataframe.'''
    df = dd.read_csv(filename, dtype={'sid': 'str', 'memoryid': 'int64', 'rankid': 'int64', 'bankid': 'int64', 'row': 'int64', 'col': 'int64', 'error_type': 'int64', 'error_time': 'str'})
    return df

if __name__ == '__main__':
    trouble_tickets = pd.read_csv('trouble_tickets.csv')
    trouble_tickets.to_pickle('trouble_tickets.pkl')
    df = load_csv('mcelog.part.csv')
    df = normalize_err_time(df)

    if os.path.exists('data_df.pkl'):
        with open('data_df.pkl', 'rb') as f:
            df = pickle.load(f)
    else:
        df = df.compute()
        with open('data_df.pkl', 'wb') as f:
            pickle.dump(df, f)

    if os.path.exists('transient_df.pkl') and os.path.exists('might_permanent_df.pkl'):
        transient_phy_res = pickle.load(open('transient_phy_res.pkl', 'rb'))
        permanent_phy_res = pickle.load(open('permanent_phy_res.pkl', 'rb'))
    else:       
        df = df.sort_values(by=['sid', 'memoryid', 'rankid', 'bankid', 'row', 'col','error_time']).reset_index(drop=True)
        df = generate_min_max(df).reset_index(drop=True)
        df = groupby_machine_informations(df)
        
        transient_df, might_permanent_df = split_transient_and_permanent(df)

        # make feature vectors for transient and might permanent errors
        ifeature_vector_df = extract_feature_vector(transient_df)
        pfeature_vector_df = extract_feature_vector(might_permanent_df)

        # make decisions based on the feature vectors
        transient_log_res, transient_phy_res = make_decision(ifeature_vector_df)
        permanent_log_res, permanent_phy_res = make_decision(pfeature_vector_df)


    category_phy = pd.DataFrame(columns=['sid', 'memoryid', 'rankid', 'bankid', 'category', 'permanency'])
    for key in permanent_phy_res.keys():
        permanent_phy_res
        if permanent_phy_res[key].empty:
            continue
        category_phy = pd.concat([category_phy, pd.DataFrame({'sid': permanent_phy_res[key]['sid'], 'memoryid': permanent_phy_res[key]['memoryid'], 'category': key, 'permanency': 'permanent'})])
    for key in transient_phy_res.keys():
        if transient_phy_res[key].empty:
            continue
        category_phy = pd.concat([category_phy, pd.DataFrame({'sid': transient_phy_res[key]['sid'], 'memoryid': transient_phy_res[key]['memoryid'], 'category': key, 'permanency': 'transient'})])
    category_phy.reset_index(drop=True, inplace=True)
    with open('category.pkl', 'wb') as f:
        pickle.dump(category_phy, f)
    '''
    for cat in category_phy['category'].unique():
        cnt = (category_phy['category'] == cat).sum()
        print(f"{cat}")
    '''

    category_log = pd.DataFrame(columns=['sid', 'memoryid', 'rankid', 'bankid', 'category', 'permanency'])
    for key in permanent_log_res.keys():
        if permanent_log_res[key].empty:
            continue
        category_log = pd.concat([category_log, pd.DataFrame({'sid': permanent_log_res[key]['sid'], 'memoryid': permanent_log_res[key]['memoryid'], 'category': key, 'permanency': 'permanent'})])
    for key in transient_log_res.keys():
        if transient_log_res[key].empty:
            continue
        category_log = pd.concat([category_log, pd.DataFrame({'sid': transient_log_res[key]['sid'], 'memoryid': transient_log_res[key]['memoryid'], 'category': key, 'permanency': 'transient'})])
    category_log.reset_index(drop=True, inplace=True)
    category_log.to_csv('category_log.csv', index=False)
