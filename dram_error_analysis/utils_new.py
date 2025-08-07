import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt 
import itertools
import dask
import dask.dataframe as dd
import dask.diagnostics
import swifter

MAX_MAT_ROWS = 2**10

def load_csv(filename):
    '''Load csv and return a dataframe.'''
    df = dd.read_csv(filename, dtype={'sid': 'str', 'memoryid': 'uint8', 'rankid': 'uint8', 'bankid': 'uint8', 'row': 'uint32', 'col': 'uint16', 'error_type': 'uint8', 'error_time': 'str'})
    return df

def normalize_err_time(df):
    # choose the correct .apply() signature
    if isinstance(df, pd.DataFrame):
        # pandas: no meta arg
        date = df['error_time'].apply(lambda x: '-'.join(x.split('-')[:3]))
        date = date.apply(lambda x: x.replace('0001-01', '2019-10'))
        date = date.apply(lambda x: x.replace('0001-02', '2019-11'))
        date = date.apply(lambda x: x.replace('0001-03', '2019-12'))
        date = date.apply(lambda x: x.replace('0001-04', '2020-01'))
        date = date.apply(lambda x: x.replace('0001-05', '2020-02'))
        date = date.apply(lambda x: x.replace('0001-06', '2020-03'))
        date = date.apply(lambda x: x.replace('0001-07', '2020-04'))
        date = date.apply(lambda x: x.replace('0001-08', '2020-05'))

    else:
        # Dask: must supply meta
        date = df['error_time'].apply(lambda x: '-'.join(x.split('-')[:3]), meta=('error_time', 'str'))
        date = date.apply(lambda x: x.replace('0001-01', '2019-10'), meta=('error_time', 'str'))
        date = date.apply(lambda x: x.replace('0001-02', '2019-11'), meta=('error_time', 'str'))
        date = date.apply(lambda x: x.replace('0001-03', '2019-12'), meta=('error_time', 'str'))
        date = date.apply(lambda x: x.replace('0001-04', '2020-01'), meta=('error_time', 'str'))
        date = date.apply(lambda x: x.replace('0001-05', '2020-02'), meta=('error_time', 'str'))
        date = date.apply(lambda x: x.replace('0001-06', '2020-03'), meta=('error_time', 'str'))
        date = date.apply(lambda x: x.replace('0001-07', '2020-04'), meta=('error_time', 'str'))
        date = date.apply(lambda x: x.replace('0001-08', '2020-05'), meta=('error_time', 'str'))

    df['error_time'] = dd.to_datetime(date)
    return df

def strip_sid_prefix(df):
    return df.assign(sid=df['sid'].str.replace(r'^Server_', '', regex=True)
            .astype('uint16')
    )

def split_transient_and_permanent(df):
    # group by sid and memoryid, and calculate the min and max error_time
    min_value = df.groupby(['sid','memoryid'])['error_time_min'].apply(lambda x: x.min())
    max_value = df.groupby(['sid','memoryid'])['error_time_max'].apply(lambda x: x.max())
    diff_value = ((max_value - min_value) < pd.Timedelta(24,'h')).reset_index()
    diff_value.columns = ['sid','memoryid','diff_max_min']
    diff_df = df.merge(diff_value, on=['sid','memoryid'], how ='left')

    # classify the errors of sid & memoryid group into transient / might permanent
    transient_df = diff_df[diff_df['diff_max_min'] == True].reset_index(drop=True)
    potential_permanent_df = diff_df[diff_df['diff_max_min'] == False].reset_index(drop=True)
    transient_df = transient_df.drop(columns='diff_max_min')
    potential_permanent_df = potential_permanent_df.drop(columns='diff_max_min')

    return transient_df, potential_permanent_df

def extract_feature_vector(df, clustering=False):
    if clustering:
        feature_vector_df = df.groupby(['sid', 'memoryid']).agg({'rankid': list, 'bankid': list, 'row': list, 'col': list, 'error_time_min': list, 'error_time_max': list, 'DRAM_model': list, 'cluster': list})
    else:
        feature_vector_df = df.groupby(['sid', 'memoryid']).agg({'rankid': list, 'bankid': list, 'row': list, 'col': list, 'error_time_min': list, 'error_time_max': list, 'DRAM_model': list})
    feature_vector_df['DRAM_model'] = feature_vector_df['DRAM_model'].apply(lambda x: x[0])
    feature_vector_df = feature_vector_df.reset_index()
    feature_vector_df = feature_vector_df[(feature_vector_df['DRAM_model']!="B2") & (feature_vector_df['DRAM_model']!="B3")]

    return feature_vector_df


def generate_min_max(df, drop_hour=False):
    if drop_hour:
        df['error_time'] = df['error_time'].dt.floor('d')
    df = df.groupby(['sid','memoryid', 'rankid', 'bankid', 'row', 'col'])['error_time'].agg(['min', 'max', 'count'])
    df = df.rename(columns={'min': 'error_time_min', 'max': 'error_time_max'})    
    df = df.reset_index()
    return df

def line_analysis_col(df):
    ax = plt.gca()
    df.col.apply(np.sort).apply(lambda x: pd.DataFrame(x).rank(method='dense').values.flatten()).explode().reset_index().groupby('index').col.plot()
    for i in range (9):
        ax.axhline(y=i*16, color='k', linestyle='--',alpha=0.3)
    for i in range (5):
        ax.axhline(y=i*32, color='k', linestyle='--',alpha=0.3)
    for i in range(3):
        ax.axhline(y=i*64, color='k', linestyle='--',alpha=0.3) 


def line_analysis(df):
    ax = plt.gca()
    df.row.apply(lambda x: np.array(x)).apply(np.sort).explode().reset_index().groupby('index').row.plot()
    for i in range (33):
        ax.axhline(y=i*4*1024, color='k', linestyle='--',alpha=0.3)
    for i in range (17):
        ax.axhline(y=i*8*1024, color='k', linestyle='--',alpha=0.3)
    for i in range(9):
        ax.axhline(y=i*16*1024, color='k', linestyle='--',alpha=0.3) 


def make_tuple(a,b):
    return (a,b)

def custom_func(x,func):
    combinations = list(itertools.combinations(x, 2))
    return pd.DataFrame(combinations, columns=['value1', 'value2']).apply(lambda x: func(x['value1'],x['value2']), axis=1)


#using time intersection, if the there exist common time interval, then they are in the multi_socket failure             
def find_multisocket(df):
    multi_socket = df
    multi_socket.reset_index(inplace=True,drop=True)
    multi_socket['error_time_min'] = multi_socket['error_time_min'].swifter.apply(lambda x: min(x))
    multi_socket['error_time_max'] = multi_socket['error_time_max'].swifter.apply(lambda x: max(x))
    latest_start = multi_socket.groupby(['sid']).error_time_min.apply(lambda x: custom_func(x,max))
    earliest_end = multi_socket.groupby(['sid']).error_time_max.apply(lambda x: custom_func(x,min))
    intersections =  latest_start<=earliest_end
    iterations = multi_socket.groupby(['sid']).memoryid.apply(lambda x: custom_func(x,make_tuple))
    intersections = intersections.reset_index()
    iterations = iterations.reset_index()

    iterations = pd.merge(iterations,intersections, on=['sid','level_1'], how='inner')
    iterations = iterations[iterations[0]==True]
    iterations.reset_index(inplace=True,drop=True)
    # using iterations make sid, memoryid table. memoryid is tuple but need to be element by expand
    iterations = iterations[['sid','memoryid']].explode('memoryid').reset_index(drop=True)
    iterations = iterations[['sid','memoryid']].drop_duplicates()
    # keep multisocket only if sid and memoryid are in iterations
    multi_socket = multi_socket.join(iterations.set_index(['sid','memoryid']), on=['sid','memoryid'], how='inner')
    return multi_socket.reset_index(drop=True)

def decide_logical_fault_category(feature_vector_df, msocket=True, mrank=True):
    testdf = feature_vector_df.swifter.apply(lambda x: list(zip(x['row'], x['col'],x['rankid'],x['bankid'])), axis=1)
    testdf = testdf.swifter.apply(lambda x: len(set(x)))

    multiple_single_bit_failures = feature_vector_df[testdf <= 2]
    multi_bit_failures = feature_vector_df[testdf > 2]
    multi_bit_failures = multi_bit_failures.reset_index(drop=True)

    if msocket:
        try:
            multi_socket_multibits = multi_bit_failures.groupby(['sid']).agg({'memoryid': list})
            multi_socket_multibits = multi_socket_multibits[multi_socket_multibits['memoryid'].swifter.apply(lambda x: len(set(x))) > 1]
            multi_socket_multibits = multi_socket_multibits.reset_index()
            multi_socket_multibits = multi_socket_multibits[['sid']]
            multi_socket_multibits = multi_bit_failures.join(multi_socket_multibits.set_index('sid'), on='sid', how='inner')
            multi_socket_multibits = multi_socket_multibits.reset_index(drop=True)
            multi_socket = find_multisocket(multi_socket_multibits)

            # join two tables to get index of multi_socket
            multi_socket = multi_bit_failures.join(multi_socket.set_index(['sid','memoryid']), on=['sid','memoryid'], how='inner', lsuffix='', rsuffix='_right')
            # drop right suffix columns
            multi_socket = multi_socket.drop([col for col in multi_socket.columns if col.endswith('_right')], axis=1)

        except:
            multi_socket = pd.DataFrame(columns=multi_bit_failures.columns)
        
        multi_bit_failures = multi_bit_failures[~multi_bit_failures.index.isin(multi_socket.index)]
    else:
        multi_socket = pd.DataFrame(columns=multi_bit_failures.columns)

    if mrank:
        single_rank = multi_bit_failures[multi_bit_failures['rankid'].swifter.apply(lambda x: len(set(x))) == 1]
    else:
        single_rank = multi_bit_failures
    single_rank = single_rank.reset_index(drop=True)
    single_column = single_rank[mask_unique(single_rank, 'col', eq_count=1) & mask_unique(single_rank, 'bankid', eq_count=1)]
    single_row = single_rank[mask_unique(single_rank, 'row', eq_count=1) & mask_unique(single_rank, 'bankid', eq_count=1)]
    single_bank = single_rank[mask_unique(single_rank, 'col', min_gt_count=1) & mask_unique(single_rank, 'row', min_gt_count=1) & mask_unique(single_rank, 'bankid', eq_count=1)]
    multi_bank = single_rank[mask_unique(single_rank, 'bankid', min_gt_count=1)]

    # if feature_vector_df is dask dataframe, then compute it
    if isinstance(feature_vector_df, dd.DataFrame):
        with dask.diagnostics.ProgressBar():
            multiple_single_bit_failures, single_rank, single_column, single_row, single_bank, multi_bank, multi_bit_failures \
            = dask.compute(multiple_single_bit_failures, single_rank, single_column, single_row, single_bank, multi_bank, multi_bit_failures)
    
    #remove single_column, row, bank, multibank from single_rank
    single_rank = single_rank[~single_rank.index.isin(single_column.index)]
    single_rank = single_rank[~single_rank.index.isin(single_row.index)]
    single_rank = single_rank[~single_rank.index.isin(single_bank.index)]
    single_rank = single_rank[~single_rank.index.isin(multi_bank.index)]
    
    if mrank:
        single_rank = single_rank[~single_rank.index.isin(multi_socket.index)]
        multi_rank = multi_bit_failures[mask_unique(multi_bit_failures, 'rankid', min_gt_count=1)]
        multi_rank = multi_rank.reset_index(drop=True)
    else:
        multi_rank = pd.DataFrame(columns=multi_bit_failures.columns)

    # make first result using above dataframes need deep copy
    # compute entire dask dataframe
    
    logical_result = {'single_rank': single_rank.copy(), 'multi_rank': multi_rank.copy(), 'multi_bank': multi_bank.copy(), 'multi_socket': multi_socket.copy(), 'single_bank': single_bank.copy(), 'single_row': single_row.copy(), 'single_column': single_column.copy(), 'multiple_single_bit_failures': multiple_single_bit_failures.copy()}

    return logical_result

def safe_min(seq):
    return min(seq) if len(seq)>0 else 0

def safe_max(seq):
    return max(seq) if len(seq)>0 else 0

def count_unique(seq):
    return len(set(seq))

def span(seq):
    return max(seq) - min(seq)

def mask_unique(df, entry, *, eq_count= None, max_count=None, min_gt_count=None):
    """
    Return a boolean mask where:
      - max_equal: count_unique ≤ max_equal
      - min_gt:  count_unique > min_gt
    """
    s = df[entry].swifter.apply(count_unique)
    mask = pd.Series(True, index=df.index)
    if eq_count is not None:
        mask &= (s == eq_count)
    if max_count is not None:
        mask &= (s <= max_count)
    if min_gt_count is not None:
        mask &= (s > min_gt_count)
    return mask

def mask_span(df, entry, *, eq_span=None, max_span=None, min_span=None):
    """
    Return a boolean mask where:
      - span ≤ max_span
      - span ≥ min_span
    """
    s = df[entry].swifter.apply(span)
    mask = pd.Series(True, index=df.index)
    if eq_span is not None:
        mask &= (s == eq_span)
    if max_span is not None:
        mask &= (s <= max_span)
    if min_span is not None:
        mask &= (s >= min_span)
    return mask 

def split_on_mask(df, mask):
    """
    Return (df_true, df_false) where mask is a boolean Series
    """
    return df[mask], df[~mask]

def make_empty_fault_df() -> pd.DataFrame:
    physical_fault_cols = [
        'sid','memoryid','rankid','bankid',
        'row','col','error_time_min','error_time_max','DRAM_model'
    ]
    return pd.DataFrame(columns=physical_fault_cols)


def decide_physical_fault_category(logical_result):
    single_rank = logical_result['single_rank']
    multi_rank = logical_result['multi_rank']
    multi_bank = logical_result['multi_bank']
    multi_socket = logical_result['multi_socket']
    single_bank = logical_result['single_bank']
    single_column = logical_result['single_column']
    single_row = logical_result['single_row']
    multiple_single_bit = logical_result['multiple_single_bit_failures']

    #########################
    # classify single_row
    #########################
    # local wordline error: single row in 1 mat
    local_wordline = single_row


    #########################
    # classify single_column
    # -------> single_sense_amp / decoder_single_col / single_csl_column / not_clustered_single_column
    #########################
    # single sense amplifier error : single col in 2 vertically-adjacent mats
        # first, remove local bitline error
    mask_ssa = mask_unique(single_column, 'row', max_count=MAX_MAT_ROWS*2)
    upper_cluster = single_column['row'].swifter.apply(lambda x: np.array(x)[np.array(x) > min(x)+MAX_MAT_ROWS])
    mask_ssa &= ( upper_cluster.swifter.apply(lambda x: max(x)-min(x) if len(x)>0 else 0) <= MAX_MAT_ROWS )
    mask_ssa &= mask_span(single_column, 'row', max_span=2**16)
    single_sense_amp, not_clustered_single_column = split_on_mask(single_column, mask_ssa)

    # decoder single column : 
        # single column caused by column decoder
        # is there any error after 64k from the min error
    mask_dsc = mask_span(not_clustered_single_column, 'row', min_span=63*1024)
    decoder_single_col, not_clustered_single_column = split_on_mask(not_clustered_single_column, mask_dsc)
    
    # single_csl_column <==> Remapping logics in csl (subbank 16K granularity failure based on histogram of max - min)
    mask_scc = mask_span(not_clustered_single_column,'row', max_span=2**14+1024)
    mask_scc |= ((not_clustered_single_column.DRAM_model == ('A1'))|(not_clustered_single_column.DRAM_model == ('A2')))  \
            & mask_span(not_clustered_single_column, 'row', max_span=2**13+1024)
    single_csl_column, not_clustered_single_column = split_on_mask(not_clustered_single_column, mask_scc)
    '''
    single_csl_column =\
            single_csl_column[single_csl_column['row'].apply(lambda x: np.array(x)[np.array(x) > min(x)+2**11])\
            .apply(lambda x: np.array(x)[np.array(x) > min(x)+2**11] if x.any() else np.array([0]))\
                .apply(lambda x: np.array(x)[np.array(x) > min(x)+2**11] if x.any() else np.array([0]))\
                    .apply(lambda x: np.array(x)[np.array(x) > min(x)+2**11] if x.any() else np.array([0])).apply(len) <= 2]
    '''

    #############################
    # classify single_bank
    # -------> local_workdline_two_clusters, consequtive_rows, subarray_row_decoder, \
    #            subarray_row_decoder_two_clusters, lwl_sel, lwl_sel2, global_row_decoder_two_clusters, \
    #            decoder_multi_col, single_csl_bank, multi_csls
    ###########################
    # local wordline two clusters :
    try:
        mask_lwltc = mask_unique(single_bank, 'row', eq_count=2)
        mask_lwltc &= mask_unique(single_bank, 'col', min_gt_count=2)
        mask_lwltc &= mask_span(single_bank, 'row', eq_span=2**16)
        local_wordline_two_clusters, single_bank = split_on_mask(single_bank, mask_lwltc)

    except:
        local_wordline_two_clusters = make_empty_fault_df()

        # single bank, multi row
        # subarray row decoder, global row decoder, local word line select generator

    # consequtive rows:
        # consequtive_rows is part of single row error
    try:
        mask_cr = mask_span(single_bank, 'row', max_span=4)
        mask_cr &= mask_unique(single_bank, 'col', min_gt_count=2)
        consequtive_rows, single_bank = split_on_mask(single_bank, mask_cr)
    except:
        consequtive_rows = make_empty_fault_df()

    try:
        mask_srd = mask_span(single_bank, 'row', max_span=2**10)
        mask_srd &= mask_unique(single_bank, 'col', min_gt_count=2)
        subarray_row_decoder, ncsb = split_on_mask(single_bank, mask_srd)
    except:
        subarray_row_decoder = make_empty_fault_df()

    # ncsb stands for not_clustered_single_bank
    try:
        mask_srdtc = mask_unique(ncsb, 'col', min_gt_count=2)
        upper_cluster = ncsb['row'].swifter.apply(lambda x: np.array(x)[np.array(x) >= min(x)+2**10])
        mask_srdtc &= (upper_cluster.swifter.apply(safe_min) - ncsb['row'].swifter.apply(safe_min) >= 62*1024)
        subarray_row_decoder_two_clusters, ncsb = split_on_mask(ncsb, mask_srdtc)
    except:
        subarray_row_decoder_two_clusters = make_empty_fault_df()

    try:
        mask_ls = mask_unique(ncsb, 'row', min_gt_count=2)
        mask_ls &= mask_unique(ncsb, 'col', min_gt_count=2)

        # range from 0 to 64 and 1023 to 1023-64
        # 64 come from the fact that MWL is 64. when FX is wrong, it will be repeated, and 64 is the maximum for error region
        rng = [ i for i in range(0,64)] + [i for i in range(1024-1,1024-1-64,-1)]
        mask_ls &= ncsb['row'].swifter.apply(lambda x: sorted(set(x))).swifter.apply(lambda x: set(np.diff(x)%(2**10))).swifter.apply(lambda x: all([i in [j for j in rng] for i in x]))

        # Needs to span for 64k rows
        mask_ls &= mask_span(ncsb, 'row', min_span=2**16)
        lwl_sel, ncsb = split_on_mask(ncsb, mask_ls)
    except:
        lwl_sel = make_empty_fault_df()
    
    # subbank and subarray level repeat
    try:
        mask_ls2 = mask_unique(ncsb, 'row', min_gt_count=2)
        mask_ls2 &= mask_unique(ncsb, 'col', min_gt_count=2)

        # range from 0 to 1024 and 16k to 16k-1024
        rng = [ i for i in range(0,1024)] + [i for i in range(2**14-1,2**14-1-1024,-1)]
        mask_ls2 &= ncsb['row'].swifter.apply(lambda x: sorted(set(x))).swifter.apply(lambda x: set(np.diff(x)%(2**14))).swifter.apply(lambda x: all([i in [j for j in rng] for i in x]))
        
        # Needs to span for 64k rows
        mask_ls2 &= mask_span(ncsb, 'row', min_span=2**16)
        lwl_sel2, ncsb = split_on_mask(ncsb, mask_ls2)
    except:
        lwl_sel2 = make_empty_fault_df()

    try:
        mask_grdtc = mask_unique(ncsb, 'col', min_gt_count=2)
        upper_cluster = ncsb['row'].swifter.apply(lambda x: np.array(x)[np.array(x) >= min(x)+16*1024])
        lower_cluster = ncsb['row'].swifter.apply(lambda x: np.array(x)[np.array(x) < min(x)+16*1024])
        mask_grdtc &= ( upper_cluster.swifter.apply(safe_min) - lower_cluster.swifter.apply(safe_max) >= 62*1024 )
        #mask_grdtc &= ncsb['row'].swifter.apply(lambda x:min(np.array(x)[np.array(x) >= min(x)+16*1024])-x.swifter.apply(lambda x: np.array(x)[x>0]).swifter.apply(lambda x: x>=62*1024).swifter.apply(all)
        global_row_decoder_two_clusters, ncsb = split_on_mask(ncsb, mask_grdtc)
    except:
        global_row_decoder_two_clusters = make_empty_fault_df()

    # single bank, multi column
    try:
        upper_cluster = ncsb['row'].swifter.apply(lambda x: np.array(x)[np.array(x) >= min(x)+2**14])
        mask_dmc = (upper_cluster.swifter.apply(safe_min) - ncsb['row'].swifter.apply(safe_min) >= 63*1024)
        mask_dmc &= mask_unique(ncsb, 'col', eq_count=2)
        decoder_multi_col, ncsb = split_on_mask(ncsb, mask_dmc)
    except:
        decoder_multi_col = make_empty_fault_df()
    
    try:
        mask_scb = mask_span(ncsb, 'row', max_span=2**14+1024)
        mask_scb |= (((ncsb.DRAM_model == ('A1'))|(ncsb.DRAM_model == ('A2')))  \
            & mask_span(ncsb, 'row', max_span=2**13+1024 ))
        mask_scb &= mask_unique(ncsb, 'col', eq_count=2)
        single_csl_bank, ncsb = split_on_mask(ncsb, mask_scb)
    except:
        single_csl_bank = make_empty_fault_df()
    
    try:
        mask_mc = mask_span(ncsb, 'row', max_span=2**14+1024)
        mask_mc |= (((ncsb.DRAM_model == ('A1'))|(ncsb.DRAM_model == ('A2')))  \
            & mask_span(ncsb, 'row', max_span=2**13+1024 ))
        multi_csls, ncsb = split_on_mask(ncsb, mask_mc)
    except:
        multi_csls = make_empty_fault_df()


    ##########################
    # classify multi_bank
    # -------> bank_control, potentional_sense_amp, potential_csl_column
    ##########################

    #multi bank 
    # bank control, row_addr_mux

    # multi bank, but if it only affects at most 2K rows, then it is actually single_sense_amp
    try:
        mask_bc = mask_unique(multi_bank, 'col', max_count=2)
        mask_bc &= (multi_bank['row'].apply(len) >= 4)
        bank_control, not_clustered_multi_bank = split_on_mask(multi_bank, mask_bc)
    except:
        bank_control = make_empty_fault_df()
    try:
        potential_sense_amp = bank_control.copy()
        #potential_sense_amp = bank_control[bank_control.error_type != "read"]
        potential_positions = potential_sense_amp.groupby(['sid']).agg({'row':lambda x: sum(x.apply(len))})
        #potential_positions = potential_sense_amp.groupby(['sid']).agg({'row':lambda x: sum(x.apply(len))})
        potential_positions.rename(columns={'row':'row_num'}, inplace=True)
        potential_sense_amp = potential_sense_amp.reset_index()
        potential_sense_amp = potential_sense_amp.merge(potential_positions, on='sid', how='left')
        potential_sense_amp = potential_sense_amp.set_index('index')
        potential_csl_column = potential_sense_amp[potential_sense_amp.row_num > 2048]
        potential_sense_amp = potential_sense_amp[potential_sense_amp.row_num <=2048]
        # potential_sense_amp should not have duplicated sid if there is duplicated sid, then drop both
        # if it is multi module error, sum the counter of errors up and compare with 2048
        # potential_sense_amp = potential_sense_amp.groupby(['sid']).agg({'row':lambda x: sum(x.apply(len))}).reset_index()
        # potential_sense_amp = potential_sense_amp[~potential_sense_amp.sid.duplicated(keep=False)]



        bank_control = bank_control[~bank_control.index.isin(potential_sense_amp.index)]
        bank_control = bank_control[~bank_control.index.isin(potential_csl_column.index)]

    except Exception as e:
        potential_sense_amp = make_empty_fault_df()
        potential_csl_column = make_empty_fault_df()

    physiclal_result =  { 'multiple_single_bit_failures':multiple_single_bit,
    'local_wordline':local_wordline, 'single_sense_amp':single_sense_amp, 'decoder_single_col':decoder_single_col, 'single_csl_column':single_csl_column, 'not_clustered_single_column':not_clustered_single_column,
    'single_csl_bank':single_csl_bank, 'subarray_row_decoder':subarray_row_decoder,
    'decoder_multi_col':decoder_multi_col, 'multi_socket': multi_socket, 'multi_rank':multi_rank,
    'global_row_decoder_two_clusters':global_row_decoder_two_clusters, 'mutli_csls':multi_csls, 'lwl_sel2':lwl_sel2, 'lwl_sel':lwl_sel, 'bank_control':bank_control, 'not_clustered_single_bank':ncsb, 'not_clustered_multi_bank':not_clustered_multi_bank,
    "subarray_row_decoder_two_clusters":subarray_row_decoder_two_clusters, 'local_wordline_two_clusters':local_wordline_two_clusters,
    "consequtive_rows" : consequtive_rows,"potential_sense_amp":potential_sense_amp,"potential_csl_column":potential_csl_column}
    #retun all results
    return physiclal_result


def make_decision(feature_vector_df, msocket = True, mrank=True):
    logical_result = decide_logical_fault_category(feature_vector_df, msocket, mrank)
    physical_result = decide_physical_fault_category(logical_result)

    return logical_result, physical_result


def conv_mapping(df,column_name):
    '''
    Coarse grained convergence using this mapping
    map values to camparison
    To : From
    SWD : local_wordline, local_wordline_two_clusters, subarray_row_decoder, subarray_row_decoder_two_clusters
    BLSA: single_sense_amp
    Col_decoder: decoder_single_col, decoder multi_col
    CSL: single_csl_column, single_csl_bank, mutli_csls
    Row_decoder: global_row_decoder_two_clusters, lwl_sel, lwl_sel2 
    Bank_patterns: bank_control, row_addr_mux
    multi_bank: not_clustered_multi_bank
    '''
    mapping = {'SWD': ['local_wordline', 'local_wordline_two_clusters', 'subarray_row_decoder', 'subarray_row_decoder_two_clusters'],
                'BLSA': ['single_sense_amp'],
                'Col_decoder': ['decoder_single_col', 'decoder_multi_col'],
                'CSL': ['single_csl_column', 'single_csl_bank', 'mutli_csls'],
                'Row_decoder': [ 'global_row_decoder_two_clusters', 'lwl_sel', 'lwl_sel2'],
                'Bank_patterns': ['bank_control', 'row_addr_mux'],
                'multi_bank': ['not_clustered_multi_bank']}
    df = df.copy()    
    for key in mapping.keys():
        df.loc[df[column_name].isin(mapping[key]),column_name] = key
    return df

def conv_mapping_onlykey(key_name):
    mapping = {'SWD': ['local_wordline', 'local_wordline_two_clusters', 'subarray_row_decoder', 'subarray_row_decoder_two_clusters'],
                'BLSA': ['single_sense_amp'],
                'Col_decoder': ['decoder_single_col', 'decoder_multi_col'],
                'CSL': ['single_csl_column', 'single_csl_bank', 'mutli_csls'],
                'Row_decoder': ['global_row_decoder_two_clusters', 'lwl_sel', 'lwl_sel2'],
                'Bank_patterns': ['bank_control', 'row_addr_mux'],
                'multi_bank': ['not_clustered_multi_bank']}
    for key in mapping.keys():
        if key_name in mapping[key]:
            return key
    return key_name

def groupby_machine_informations(df):
    # first row is the name of the column
    machines=pd.read_csv("inventory.csv",header=0)

    #using sid, join the machines dataframe with the df dataframe
    df = df.merge(machines[['sid','DRAM_model','DIMM_number','server_manufacturer']], on='sid')   
    return df

#plot horizontal lines on the plot on 8k 16k 24k 32k 40k 48k ... 128k
def plot_hlines(ax):
    for i in range(17):
        ax.axhline(y=i*8*1024, color='k', linestyle='--',alpha=0.3)
    return ax

'''
Coarse grained convergence using this mapping
map values to camparison
To : From
SWD : local_wordline, local_wordline_two_clusters, subarray_row_decoder, subarray_row_decoder_two_clusters
BLSA: single_sense_amp
Col_decoder: decoder_single_col, decoder multi_col
CSL: single_csl_column, single_csl_bank, mutli_csls
Row_decoder: global_row_decoder_two_clusters, lwl_sel, lwl_sel2 
Bank_patterns: bank_control, row_addr_mux
multi_bank: not_clustered_multi_bank
'''
def merge_dictionary(dic):
    mapping = {'SWD': ['local_wordline', 'local_wordline_two_clusters', 'subarray_row_decoder', 'subarray_row_decoder_two_clusters'],
                'BLSA': ['single_sense_amp'],
                'Col_decoder': ['decoder_single_col', 'decoder_multi_col'],
                'CSL': ['single_csl_column', 'single_csl_bank', 'mutli_csls'],
                'Row_decoder': ['global_row_decoder_two_clusters', 'lwl_sel', 'lwl_sel2'],
                'Bank_patterns': ['bank_control', 'row_addr_mux'],
                'multi_bank': ['not_clustered_multi_bank']}
    
    for key in mapping.keys():
        if key not in dic.keys():
            dic[key] = pd.DataFrame()
        for value in mapping[key]:
            if value in dic.keys():
                dic[key] = pd.concat([dic[key],dic[value]])
                del dic[value]
    return dic


def format_results(results_dict, output_file):
    with open(output_file, 'w') as f:
        for category, df in results_dict.items():
            f.write(f"=== {category} ===\n")
            if df.empty:
                f.write("No entries.\n\n")
                continue

            # Group by sid and memoryid first
            for (sid, memoryid), sub_df in df.groupby(['sid', 'memoryid']):
                f.write(f"\nsid: {sid}\n")
                f.write(f"memoryid: {memoryid}\n")

                # Handle vectorized entries (list-type columns)
                expanded_rows = []
                for _, row in sub_df.iterrows():
                    rankid = row['rankid']
                    bankid = row['bankid']
                    row_addr = row['row']
                    col = row['col']
                    etmin = row['error_time_min']
                    etmax = row['error_time_max']
                    model = row['DRAM_model']

                    if isinstance(rankid, list):
                        for i in range(len(rankid)):
                            expanded_rows.append({
                                'rankid': rankid[i],
                                'bankid': bankid[i],
                                'row': row_addr[i],
                                'col': col[i],
                                'error_time_min': etmin[i],
                                'error_time_max': etmax[i],
                                'DRAM_model': model
                            })
                    else:
                        expanded_rows.append({
                            'rankid': rankid,
                            'bankid': bankid,
                            'row': row_addr,
                            'col': col,
                            'error_time_min': etmin,
                            'error_time_max': etmax,
                            'DRAM_model': model
                        })

                expanded_df = pd.DataFrame(expanded_rows)

                # Now group by (rankid, bankid)
                for (rankid, bankid), group in expanded_df.groupby(['rankid', 'bankid']):
                    f.write(f"\n  [rankid: {rankid}, bankid: {bankid}]\n")
                    f.write("    row    |   col   | error_time_min       | error_time_max       | DRAM_model\n")
                    f.write("    " + "-" * 80 + "\n")
                    for _, r in group.iterrows():
                        f.write(f"    {r['row']:>6} | {r['col']:>6} | {r['error_time_min']} | {r['error_time_max']} | {r['DRAM_model']}\n")
            f.write("\n")