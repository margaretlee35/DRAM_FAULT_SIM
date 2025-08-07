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
    df = load_csv('mcelog.csv')
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
        df = generate_min_max(df, drop_hour=True).reset_index(drop=True)
        df = groupby_machine_informations(df)
        print('Unique server IDs:', df['sid'].nunique())
        
        transient_df, might_permanent_df = split_transient_and_permanent(df)

        # make feature vectors for transient and might permanent errors
        ifeature_vector_df = extract_feature_vector(transient_df)
        pfeature_vector_df = extract_feature_vector(might_permanent_df)

        # make decisions based on the feature vectors
        transient_log_res, transient_phy_res = make_decision(ifeature_vector_df, msocket=False)
        permanent_log_res, permanent_phy_res = make_decision(pfeature_vector_df, msocket=False)
    '''
    category_phy_per = pd.DataFrame(columns=['sid', 'memoryid', 'rankid', 'bankid', 'category', 'permanency', 'DRAM_model'])
    category_phy_trs = pd.DataFrame(columns=['sid', 'memoryid', 'rankid', 'bankid', 'category', 'permanency', 'DRAM_model'])
    category_phy = pd.DataFrame(columns=['sid', 'memoryid', 'rankid', 'bankid', 'category', 'permanency', 'DRAM_model'])
    for key in permanent_phy_res.keys():
        if permanent_phy_res[key].empty:
            continue
        category_phy_per = pd.concat([category_phy_per, pd.DataFrame({'sid': permanent_phy_res[key]['sid'], 'memoryid': permanent_phy_res[key]['memoryid'], 'category': key, 'permanency': 'permanent', 'DRAM_model': permanent_phy_res[key]['DRAM_model']})])
        category_phy = pd.concat([category_phy, pd.DataFrame({'sid': permanent_phy_res[key]['sid'], 'memoryid': permanent_phy_res[key]['memoryid'], 'category': key, 'permanency': 'permanent', 'DRAM_model': permanent_phy_res[key]['DRAM_model']})])
    for key in transient_phy_res.keys():
        if transient_phy_res[key].empty:
            continue
        category_phy_trs = pd.concat([category_phy_trs, pd.DataFrame({'sid': transient_phy_res[key]['sid'], 'memoryid': transient_phy_res[key]['memoryid'], 'category': key, 'permanency': 'transient', 'DRAM_model': transient_phy_res[key]['DRAM_model']})])
        category_phy = pd.concat([category_phy, pd.DataFrame({'sid': transient_phy_res[key]['sid'], 'memoryid': transient_phy_res[key]['memoryid'], 'category': key, 'permanency': 'permanent', 'DRAM_model': transient_phy_res[key]['DRAM_model']})])
    category_phy_per.reset_index(drop=True, inplace=True)
    category_phy_trs.reset_index(drop=True, inplace=True)
    
    category_phy.reset_index(drop=True, inplace=True)
    with open('category.pkl', 'wb') as f:
        pickle.dump(category_phy, f)
    '''

    category_log_per = pd.DataFrame(columns=['sid', 'memoryid', 'rankid', 'bankid', 'category', 'permanency', 'DRAM_model'])
    category_log_trs = pd.DataFrame(columns=['sid', 'memoryid', 'rankid', 'bankid', 'category', 'permanency', 'DRAM_model'])
    for key in permanent_log_res.keys():
        if permanent_log_res[key].empty:
            continue
        category_log_per = pd.concat([category_log_per, pd.DataFrame({'sid': permanent_log_res[key]['sid'], 'memoryid': permanent_log_res[key]['memoryid'], 'category': key, 'permanency': 'permanent', 'DRAM_model': permanent_log_res[key]['DRAM_model']})])
    for key in transient_log_res.keys():
        if transient_log_res[key].empty:
            continue
        category_log_trs = pd.concat([category_log_trs, pd.DataFrame({'sid': transient_log_res[key]['sid'], 'memoryid': transient_log_res[key]['memoryid'], 'category': key, 'permanency': 'transient', 'DRAM_model': transient_log_res[key]['DRAM_model']})])
    category_log_per.reset_index(drop=True, inplace=True)
    category_log_trs.reset_index(drop=True, inplace=True)
    category_log_per.to_csv('category_log_per.csv', index=False)
    category_log_trs.to_csv('category_log_trs.csv', index=False)
    
    '''
    table_phy_trs = pd.pivot_table(category_phy_trs, index='category', columns='DRAM_model', aggfunc='size', fill_value=0  )
    table_phy_per = pd.pivot_table(category_phy_per, index='category', columns='DRAM_model', aggfunc='size', fill_value=0  )

    table_phy_trs.to_csv('table_phy_trs.csv')
    table_phy_per.to_csv('table_phy_per.csv')
    '''
    table_log_trs = pd.pivot_table(category_log_trs, index='category', columns='DRAM_model', aggfunc='size', fill_value=0  )
    table_log_per = pd.pivot_table(category_log_per, index='category', columns='DRAM_model', aggfunc='size', fill_value=0  )

    desired_index = ['multi_rank', 'multi_bank', 'single_bank', 'single_column', 'single_row', 'multiple_single_bit_failures']
    desired_columns = ['A1', 'B1', 'C1', 'A2', 'C2']

    table_log_trs = table_log_trs.reindex(index=desired_index, columns=desired_columns)
    table_log_per = table_log_per.reindex(index=desired_index, columns=desired_columns)

    table_log_trs.to_csv('table_log_trs.csv')
    table_log_per.to_csv('table_log_per.csv')
