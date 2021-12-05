# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 14:53:15 2021

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


from function import smi2infomax



# smi = '../alphafold/validation/smi_val.txt'
# df = smi2infomax(smi)

# df.to_csv('../alphafold/validation/infomax_val.csv')

df = pd.read_csv('../alphafold/validation/infomax_val.csv').iloc[:,1:]

#%% deepec generate 


import pkg_resources
from tensorflow.keras.models import load_model
from deepec import utils
from deepec import ec_prediction_dl
from deepec import ec_prediction_seq
from deepec import __version__
import os
import pandas as pd
import numpy as np
DeepEC3d_oh = pkg_resources.resource_filename('deepec', 'data/DeepEC_3d.h5')
DeepEC_enzyme_oh = pkg_resources.resource_filename('deepec', 'data/Binary_class.h5')


model = load_model(DeepEC_enzyme_oh)


fasta_file = './validation.fasta'
output_dir = './output'
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

## One-hot encoding
temp_file = '%s/SwissProt_input.csv'%(output_dir)
temp_fasta_file = '%s/temp_fasta.fa'%(output_dir)

ec_prediction_dl.preprocessing(fasta_file, temp_fasta_file)
ec_prediction_dl.run_one_hot_encoding(temp_fasta_file, temp_file)
temp_df = pd.read_csv(temp_file, index_col=0)





from sklearn.decomposition import PCA
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
from sklearn.preprocessing import StandardScaler
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px


layout = go.Layout(

    plot_bgcolor="#FFF",  # Sets background color to white
    xaxis=dict(
        title="PC1",
        linecolor="#BCCCDC",  # Sets color of X-axis line
        showgrid=False  # Removes X-axis grid lines
    ),
    yaxis=dict(
        title="PC2",  
        linecolor="#BCCCDC",  # Sets color of Y-axis line
        showgrid=False,  # Removes Y-axis grid lines    
    )
)


fig = go.Figure(layout=layout)

pca = PCA(n_components=2)
components = pca.fit_transform(temp_df)
name = temp_df.index.values.tolist()
x = components[:,0]
y = components[:,1]
points = list(set(zip(x, y)))
count=[len([x for x,y in zip(x,y) if x==p[0] and y==p[1]]) for p in points]
count = np.array(count).size

fig.add_trace(go.Scatter(x=x, y=y,
                mode='markers', text=name,name='embeding'))





#%%

import sklearn
print(sklearn.__version__)
# gradient boosting for classification in scikit-learn
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from matplotlib import pyplot
import pkg_resources
from tensorflow.keras.models import load_model
from deepec import utils
from deepec import ec_prediction_dl
from deepec import ec_prediction_seq
from deepec import __version__
import os
import pandas as pd
import numpy as np
import pickle



from function import get_feature_layer

os.chdir('../deepec')
DeepEC_enzyme_oh = pkg_resources.resource_filename('deepec', 'data/Binary_class.h5')
model = load_model(DeepEC_enzyme_oh)
X_col = np.asarray(temp_df).reshape(104, 1000, 21, 1)
test = get_feature_layer(model, X_col)


components = pca.fit_transform(test)

x = components[:,0]
y = components[:,1]
points = list(set(zip(x, y)))
count=[len([x for x,y in zip(x,y) if x==p[0] and y==p[1]]) for p in points]
count = np.array(count).size

fig.add_trace(go.Scatter(x=x, y=y,
                mode='markers', text=name,name='cnn'))


plot(fig)




