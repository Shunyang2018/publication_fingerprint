# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 12:07:30 2021

@author: Study
"""

import pickle
import pandas as pd
import numpy as np
from imp import reload
import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from mymodule import VAE_model
from collections import defaultdict

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

#%%
m = torch.load('./model_files/VAE256_model_best.pth.tar')
encoder = VAE_model.MolEncoder(i=210, o=256, c=len(m['charset']))
encoder.load_state_dict(m['encoder'])
encoder.eval()

temp = []
for i in smiles:#b
# for i in b:
    i = i.replace('\n','')
  
    temp.append(processor.one_hot_encode(STRING=i, 
                                         charset=m['charset'], 
                                         size_of_encoding=len(m['charset']), 
                                         max_len_smiles=210, 
                                         dtype=np.uint8))

temp = torch.from_numpy(np.array(temp, dtype=np.uint8))
vae = TensorDataset(temp, torch.zeros(temp.size()[0]))
v_loader = DataLoader(vae, batch_size=1, shuffle=False)

vae_fps256 = np.zeros((len(smiles), 256+1), dtype=np.float) # +1 is because we also need the ID coming from the smiles file

with torch.no_grad():
    for (ind, sm) in  enumerate(v_loader):
        x_var = Variable(sm[0].type(torch.FloatTensor))
        vae_fps256[ind] = np.hstack(( encoder(x_var).detach().numpy().reshape(-1)))
        
vae_fps256 = pd.DataFrame(data=vae_fps256)

drugs_name = '/tf/notebooks/code_for_pub/smiles_files/drugcomb_drugs_export_OCT2020.csv'
drugs = pd.read_csv(drugs_name, names=['dname','id', 'smiles', 'cid'], header=0) # oct2020 version

mapping = defaultdict(list) 
for i in drugs.itertuples(): # map cid to id
    mapping[i.cid] = i.id
    
vae_fps256.iloc[:,0] = vae_fps256.iloc[:,0].astype(int)
vae_fps256.rename(columns={0:'cid'}, inplace=True)
vae_fps256['cid'] = vae_fps256['cid'].map(mapping)
vae_fps256.rename(columns={'cid':'id'}, inplace=True)
vae_fps256.set_index(keys='id', drop=True, inplace=True)

vae_fps256.head()



#%%


import pickle
import pandas as pd
import numpy as np
from imp import reload
import h5py
import shutil
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from itertools import chain
from sklearn.model_selection import KFold
from progiter import ProgIter
from sklearn.model_selection import train_test_split
from mymodule import VAE_model

random_state=34
reload(VAE_model)
processor = VAE_model.ProcessSMILES(smiles_chembl=smiles, smiles_dc=smiles)
df = processor.one_hot_encoded(processor.smiles, max_len_smiles=220)



batch_size = 1800
patience = 3
n_epochs = 10
charset = processor.charset
dtype=torch.cuda.FloatTensor

#best = torch.load('/tf/notebooks/code_for_pub/_logs_as_python_files/vae_training_logs/model_best.pth.tar')
encoder = VAE_model.MolEncoder(i=220, o=256, c=len(charset))
#encoder.load_state_dict(best['encoder'])
encoder.apply(VAE_model.initialize_weights)
decoder = VAE_model.MolDecoder(i=256, o=220, c=len(charset))
decoder.apply(VAE_model.initialize_weights)
#decoder.load_state_dict(best['decoder'])
encoder.cuda()
decoder.cuda()
best_loss = 1e6
#best_loss = best['avg_val_loss']

optimizer = torch.optim.Adam(chain(encoder.parameters(), decoder.parameters()), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                       factor=0.2,
                                                       patience=patience,
                                                       mode='min', 
                                                       min_lr=1e-6)

kf = KFold(n_splits=5, shuffle=True, random_state=random_state)        
for ind,(train_index, test_index) in enumerate(kf.split(df)):
    
    # make loader train
    dt = torch.from_numpy(df[train_index])
    train = TensorDataset(dt, torch.zeros(dt.size()[0]))
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)

    # make loader test
    dte = torch.from_numpy(df[test_index])
    test = TensorDataset(dte, torch.zeros(dte.size()[0]))
    val_loader = DataLoader(test, batch_size=batch_size, shuffle=True)

    
    # for other folds we re-use previous best model's weights
    # if ind != 0:
    #     encoder = VAE_model.MolEncoder(i=220, o=256, c=len(charset))
    #     best = torch.load('./model_files/VAE256_model_best.pth.tar')
    #     encoder.load_state_dict(best['encoder'])
    #     decoder = VAE_model.MolDecoder(i=220, o=256, c=len(charset))
    #     decoder.load_state_dict(best['decoder'])
    #     encoder.cuda()
    #     decoder.cuda()

    #     optimizer = torch.optim.Adam(chain(encoder.parameters(), decoder.parameters()), lr=1e-3)
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
    #                                                            factor=0.1,
    #                                                            patience=patience,
    #                                                            mode='min', 
    #                                                            min_lr=1e-6)
    
    
    
    # train for n epochs
    for epoch in range(n_epochs):
        print(f'outer fold:{ind}, epoch:{epoch}')
        VAE_model.train_model(train_loader, encoder, decoder, optimizer, dtype)
        avg_val_loss = VAE_model.validate_model(val_loader, encoder, decoder, dtype)

        scheduler.step(avg_val_loss)

        is_best = avg_val_loss < best_loss

        if is_best:
            best_loss = avg_val_loss
        VAE_model.save_checkpoint({
            'epoch': epoch,
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'charset': charset,
            'avg_val_loss': avg_val_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best, size=256,
            filename='./vae_training_logs')


























