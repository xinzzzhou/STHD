'''
This is writen by the author
'''
# data i/o
import os
# compute in parallel
from multiprocessing import Pool
# the usual
import numpy as np
import pandas as pd
import correlation.wt_deepgraph as dg

'''computing settings'''
n_features_ = int(862)
n_samples_ = int(17544)
step_size = 1e8
n_processes = 500
'''load data'''
root_path = './traffic/'
df_dataset = 'traffic.csv'
data='traffic'
df = pd.read_csv(root_path+df_dataset, header = None)
'''format data'''
df_x = df.values
df_x = df_x.astype(float)
df_x = df_x.T
df_x = (df_x - df_x.mean(axis=1, keepdims=True)) / df_x.std(axis=1, keepdims=True)
np.save(f'samples_{data}', df_x)
# connector function to compute pairwise pearson correlations
def corr(index_s, index_t):
    features_s = df_x[index_s]
    features_t = df_x[index_t]
    corr = np.einsum('ij,ij->i', features_s, features_t) / n_samples_
    return corr
'''define computing'''
# load samples as memory-map
df_x = np.load(f'samples_{data}.npy', mmap_mode='r')
# create node table that stores references to the mem-mapped samples
v = pd.DataFrame({'index': range(df_x.shape[0])})
# index array for parallelization
pos_array = np.array(np.linspace(0, n_features_*(n_features_-1)//2, n_processes), dtype=int)
# parallel computation
def create_ei(i):
    from_pos = pos_array[i]
    to_pos = pos_array[i+1]
    # initiate DeepGraph
    g = dg.DeepGraph(v)
    # create edges
    g.create_edges(connectors=corr, step_size=step_size,
                   from_pos=from_pos, to_pos=to_pos)
# store edge table
    g.e.to_pickle(f'tmp/correlations-{data}/{str(i).zfill(3)}.pickle')

'''computation'''
if __name__ == '__main__':
    os.makedirs(f'tmp/correlations-{data}', exist_ok=True)
    indices = np.arange(0, n_processes - 1)
    p = Pool()
    for _ in p.imap_unordered(create_ei, indices):
        pass
'''computing end, store correlation values'''
files = os.listdir(f'tmp/correlations-{data}/')
files.sort()
store = pd.HDFStore(f'e-{data}.h5', mode='w')
for f in files:
    et = pd.read_pickle(f'tmp/correlations-{data}/{f}')
    store.append('e', et, format='t', data_columns=True, index=False)
store.close()
'''transfer correlation table --> matrix'''
e = pd.read_hdf(f'e-{data}.h5')
max_index = max(e.index.get_level_values(1)) + 1
matrix = np.full((max_index, max_index), np.nan)
for (s, t), value in e.iterrows():
    matrix[s, t] = value
    matrix[t, s] = value
np.save(f'adjacency-correlation-{data}', matrix)
# for each row of tr, remove the value that the same as the index
tr = np.load(f'matrix_rank_{data}.npy')
results=[]
for i in range(862):
    tmp = []
    for j in range(862):
        if tr[i][j] != i:
            tmp.append(tr[i][j])
    results.append(tmp)
result_arr = np.array(results)
np.save(f'matrix_rank_{data}_final', result_arr)