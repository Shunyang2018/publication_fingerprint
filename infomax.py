# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 12:07:30 2021

@author: Study
"""

import dgl
from collections import defaultdict
from dgl.nn.pytorch.glob import AvgPooling
from dgllife.model import load_pretrained
from dgllife.model.model_zoo import *
from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
import numpy as np
import pandas as pd
import pickle
from rdkit import Chem
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

def collate(gs):
    return dgl.batch(gs)

random_state=34

processor = VAE_model.ProcessSMILES()



# all drugs
with open('../alphafold/mol.smi') as f:
    tmp = f.readlines()
    f.close()

with open('./smiles_files/smiles_drugcomb_BY_cid_duplicated.pickle','rb') as f:
    b = pickle.load(f)

smi_list = []
for smi in tmp:
    smi_list.append( smi.replace('/n',''))
    
    
smiles = pd.Series(smi_list)


model = load_pretrained('gin_supervised_infomax')

graphs = []

for smi in smiles:
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        g = mol_to_bigraph(mol, add_self_loop=True,
                           node_featurizer=PretrainAtomFeaturizer(),
                           edge_featurizer=PretrainBondFeaturizer(),
                           canonical_atom_order=True)
        graphs.append(g)

    except:
        continue
    
    
data_loader = DataLoader(graphs, batch_size=256, collate_fn=collate, shuffle=False)

readout = AvgPooling()

mol_emb = []
for batch_id, bg in enumerate(data_loader):
    nfeats = [bg.ndata.pop('atomic_number').to('cpu'),
              bg.ndata.pop('chirality_type').to('cpu')]
    efeats = [bg.edata.pop('bond_type').to('cpu'),
              bg.edata.pop('bond_direction_type').to('cpu')]
    with torch.no_grad():
        node_repr = model(bg, nfeats, efeats)
    mol_emb.append(readout(bg, node_repr))
mol_emb = torch.cat(mol_emb, dim=0).detach().cpu().numpy()

df = pd.DataFrame(data=mol_emb)

df2 = pd.read_csv('../map4/mol_fp.csv').iloc[:,0:9]

df = df2.join(df)
d=300
b=d+9
# df = pd.read_csv('mol_fp.csv')
dfE5 = df.iloc[:,np.r_[2,9:b]]
dfG7 = df.iloc[:,np.r_[3,9:b]]
dfE9 = df.iloc[:,np.r_[4,9:b]]
dfG6 = df.iloc[:,np.r_[5,9:b]]
dfA1 = df.iloc[:,np.r_[6,9:b]]
dfA2 = df.iloc[:,np.r_[7,9:b]]
dfA5 = df.iloc[:,np.r_[8,9:b]]


dfG7.rename(columns={'G7':'reaction'}, 
                                    inplace=True)
dfE5.rename(columns={'E5':'reaction'}, 
                                    inplace=True)
dfE9.rename(columns={'E9':'reaction'}, 
                                    inplace=True)
dfG6.rename(columns={'G6':'reaction'}, 
                                    inplace=True)
dfA1.rename(columns={'2A1':'reaction'}, 
                                    inplace=True)
dfA2.rename(columns={'2A2':'reaction'}, 
                                    inplace=True)
dfA5.rename(columns={'2A5':'reaction'}, 
                                    inplace=True)

dfG7.reset_index(drop=True, inplace=True)
dfE5.reset_index(drop=True, inplace=True)
dfE9.reset_index(drop=True, inplace=True)
dfG6.reset_index(drop=True, inplace=True)
dfA1.reset_index(drop=True, inplace=True)
dfA2.reset_index(drop=True, inplace=True)
dfA5.reset_index(drop=True, inplace=True)

dfG7.index = dfE5.index
dfE9.index = dfE5.index
dfG6.index = dfE5.index
dfA1.index = dfE5.index
dfA2.index = dfE5.index
dfA5.index = dfE5.index



df_append = pd.concat([dfE5,
                       dfG7,dfE9,dfG6,dfA1,dfA2,dfA5],axis=0)

y = df_append.iloc[:,0].to_numpy().astype('int')
X = df_append.iloc[:,1:].to_numpy().astype('float32')


df_append.to_csv('./repeat_reaction_fp_infomax.csv')


    