'''
Author: Xin Zhou
Date: 2023-11-20
Description: This code is used to reconstruct the correlation matrix from the traffic dataset, referring to ALGNN'''

from collections import defaultdict
import numpy as np
import networkx as nx
import netrd
import matplotlib.pyplot as plt
import itertools as it
import pandas as pd

'''init settings'''
recons = {
    'CorrelationMatrix':            netrd.reconstruction.CorrelationMatrix()
}
root_path = './traffic/'
dataset = 'traffic.csv'
data='traffic'
'''load data'''
df = pd.read_csv(root_path+dataset, header = None)
df.columns = df.iloc[0,:]
df=df.iloc[1:,:]
df.set_index('date', inplace=True)
df = df.T
df = df.astype(float)
'''begin reconstruction'''
for ri, R1 in list(recons.items()):
    adj = None
    print( "**** starting - " + str(ri) + " ********")
    try:
        R1.fit(np.array(df))
        print( "**** Fit successful - " + str(ri) + " ********")
        if 'thresholded_matrix' in R1.results:
            adj = pd.DataFrame(R1.results['thresholded_matrix']).abs()
        elif 'adjacency_matrix' in R1.results:
            adj = pd.DataFrame(R1.results['adjacency_matrix']).abs()
        else:
            print(f'no thresholded_matrix in {ri}\n')
            print(f'alternative keys are {R1.results.keys()}\n')
            # record R1
            with open(root_path+dataset+'_'+str(ri)+'_reconstruct.txt', 'w') as file:
                file.write(f'{R1}\n')
    except Exception as e:
        print( "**** could not perform - " + str(ri) + " ********")
        print(e)
        print()
        continue 
'''save resutls'''
if adj is not None: 
        adj.replace([np.inf, np.nan], 0, inplace=True)
        adj.to_csv(root_path+dataset+'_'+str(ri)+'_reconstruct.csv')  