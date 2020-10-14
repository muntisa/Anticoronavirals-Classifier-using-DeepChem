#!/usr/bin/env python
# coding: utf-8

# # Antivirals QSAR using DeepChem for Coronaviruses

# From the Chembl database we extracted all compounds with reported interaction to CHEMBL5118 and CHEMBL3927. These references correspond to replicase polyprotein 1ab and SARS coronavirus 3C-like proteinase respectively. These are the only coronavirus related targets in the ChemBL database.
# 
# We used SMILES of drugs as inputs that are transformed into a specific format for the future classification (anti-viral or non-antiviral) that uses Graph Convolutional Nets (DeepChem package). Instead of designing a featurization ourselves, the method learns one automatically from the data similar with the Convolutional Neural Networks for images.
# 
# 
# The calculations used Google Colab GPU.

# ### Instalation of DeepChem with GPU on GColab

# In[3]:


get_ipython().run_line_magic('tensorflow_version', '1.x')
get_ipython().system('wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh')
get_ipython().system('chmod +x Miniconda3-latest-Linux-x86_64.sh')
get_ipython().system('bash ./Miniconda3-latest-Linux-x86_64.sh -b -f -p /usr/local')
get_ipython().system('conda install -y -c deepchem -c rdkit -c conda-forge -c omnia deepchem-gpu=2.3.0')
import sys
sys.path.append('/usr/local/lib/python3.7/site-packages/')


# ### Set parameters, file names, path
# 
# Set the seed for numpy and tensorflow:

# In[ ]:


import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)
import deepchem as dc


# Mount the folder with all files from Google Drive:

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import os
import pandas as pd

# define the path to the dataset for training
input_data=os.path.join("/content/drive/My Drive/MyProjects/DeepChem/antivirals/datasets",'antivirals_SMILES.csv')

# define the path to new prediction data
pred_data=os.path.join("/content/drive/My Drive/MyProjects/DeepChem/antivirals/datasets",'DB_SMILES4prediction.csv')


# ### Transform SMILES into Convolutional Graph format

# In[8]:


# define output name (Class)
tasks=['Class']

# define features
featurizer=dc.feat.ConvMolFeaturizer()

# load data and calculate the features for dataset
loader = dc.data.CSVLoader(tasks=tasks, smiles_field="smiles",featurizer=featurizer)
dataset=loader.featurize(input_data, data_dir='/content/drive/My Drive/MyProjects/DeepChem/antivirals/features_antivirals/')

# calculate the same features for new data to predict
loader2 = dc.data.CSVLoader(tasks=tasks, smiles_field="smiles",featurizer=featurizer)
dataset_pred=loader2.featurize(pred_data, data_dir='/content/drive/My Drive/MyProjects/DeepChem/antivirals/features_DBpredictions/')


# In[9]:


print('Full dataset samples : {}'.format(dataset.X.shape[0]))
print('External dataset samples : {}'.format(dataset_pred.X.shape[0]))


# In[10]:


# define a transformer for data using only training subset!
transformers = [
                dc.trans.NormalizationTransformer(transform_y=True, dataset=dataset)]

# apply transformation to all datasets including the external 
for transformer in transformers:
  dataset = transformer.transform(dataset)

for transformer in transformers:
  dataset_pred = transformer.transform(dataset_pred)


# Define the metrics for training as AUROC:

# In[ ]:


# define mean AUROC metrics for the classifier
metric = dc.metrics.Metric(
    dc.metrics.roc_auc_score, np.mean, mode="classification")

# define number of internal features
n_feat = 75

# define batch size during the training
batch_size = 32

# dropout 
n_dropout = 0.05


# ### Test with one split and different epochs

# In[65]:


from deepchem.models import GraphConvModel

# define a splitter 
splitter = dc.splits.SingletaskStratifiedSplitter() #ScaffoldSplitter

# split dataset into train, test subsets (80% - 20%)
train_dataset, test_dataset= splitter.train_test_split(dataset, 
                                                       seed=80,
                                                       frac_train=0.8, 
                                                       verbose=False)
print('Full dataset samples : {}'.format(dataset.X.shape[0]))
print('Train dataset samples : {}'.format(train_dataset.X.shape[0]))
print('Test dataset samples : {}'.format(test_dataset.X.shape[0]))

model = GraphConvModel(
    len(tasks), batch_size=batch_size, mode='classification',
    dropout=n_dropout,
    # model_dir='/content/drive/My Drive/MyProjects/DeepChem/antivirals/models/oneSplitMoreEpochs',
    random_seed=42) # same seed here!

# check the error for optimal number of epochs
num_epochs = 200

losses = []
auroc_train = []
auroc_test = []

for i in range(num_epochs):
  loss = model.fit(train_dataset, nb_epoch=1, deterministic=True)
  print("Epoch %d loss: %f" % (i, loss))
  losses.append(loss)

  # print statistics
  print("Evaluating model")
  train_scores = model.evaluate(train_dataset, [metric], transformers)
  print("Training ROC-AUC Score: %f" % train_scores["mean-roc_auc_score"])
  test_scores = model.evaluate(test_dataset, [metric], transformers)
  print("Test ROC-AUC Score: %f" % test_scores["mean-roc_auc_score"])

  auroc_train.append(train_scores["mean-roc_auc_score"])
  auroc_test.append(test_scores["mean-roc_auc_score"])


# In[66]:


# plot the errors
import matplotlib.pyplot as plot

plot.ylabel("Loss")
plot.xlabel("Epoch")
x = range(num_epochs)
y = losses
plot.scatter(x, y)
plot.show()


# In[67]:


# plot the auroc train
import matplotlib.pyplot as plot

plot.ylabel("AUROC train")
plot.xlabel("Epoch")
x = range(num_epochs)
y = auroc_train
plot.scatter(x, y)
plot.show()


# In[68]:


# plot the auroc test
import matplotlib.pyplot as plot

plot.ylabel("AUROC test")
plot.xlabel("Epoch")
x = range(num_epochs)
y = auroc_test
plot.scatter(x, y)
plot.show()


# In[70]:


np.max(auroc_test)


# In[71]:


np.mean(auroc_test)


# ### Classification for 10 random stratified splits

# In[ ]:


# define batch size during the training
batch_size = 32

# dropout 
n_dropout = 0.05

# number of epochs
n_epoch = 70


# In[74]:


from deepchem.models import GraphConvModel

scores_train=[]
scores_test =[]

# for each seed for external split TRAIN-TEST
for seed_ext in [10,20,30,40,50,60,70,80,90,100]:
  print("*** External split")
  print("> ext seed =", seed_ext)

  # define a splitter 
  splitter = dc.splits.SingletaskStratifiedSplitter()

  # split dataset into train, test subsets (80% - 20%)
  train_dataset, test_dataset= splitter.train_test_split(dataset, 
                                                         seed=seed_ext,
                                                         frac_train=0.8, 
                                                         verbose=False)
  print('Full dataset samples : {}'.format(dataset.X.shape[0]))
  print('Train dataset samples : {}'.format(train_dataset.X.shape[0]))
  print('Test dataset samples : {}'.format(test_dataset.X.shape[0]))

  model = GraphConvModel(
      len(tasks), batch_size=batch_size, mode='classification',
      dropout=n_dropout,
      random_seed=42) # same seed here!
    
  # Fit model using train_data
  model.fit(train_dataset, nb_epoch=n_epoch, deterministic=True) # 5 for testing

  # evaluating the model for train-test
  train_scores = model.evaluate(train_dataset, [metric], transformers)

  scores_train.append(train_scores["mean-roc_auc_score"])
  
  # evaluating test scores
  test_scores = model.evaluate(test_dataset, [metric], transformers)
  scores_test.append(test_scores["mean-roc_auc_score"])


# In[75]:


scores_train


# In[76]:


scores_test


# In[77]:


np.mean(scores_train), np.mean(scores_test)


# Test scores:

# In[78]:


import matplotlib.pyplot as plt
fig1, ax1 = plt.subplots()
ax1.set_title('Basic Plot')
ax1.boxplot(scores_test)


# ### Get the final model to make prediction

# In[ ]:


# define batch size during the training
batch_size = 32

# dropout 
n_dropout = 0.05

# number of epochs
n_epoch = 70


# In[53]:


# define a splitter 
splitter = dc.splits.SingletaskStratifiedSplitter() #ScaffoldSplitter

# split dataset into train, test subsets (80% - 20%)
train_dataset, test_dataset= splitter.train_test_split(dataset, 
                                                      seed=80,
                                                      frac_train=0.8, 
                                                      verbose=False)
print('Full dataset samples : {}'.format(dataset.X.shape[0]))
print('Train dataset samples : {}'.format(train_dataset.X.shape[0]))
print('Test dataset samples : {}'.format(test_dataset.X.shape[0]))


# In[54]:


# define the model
from deepchem.models import GraphConvModel

model = GraphConvModel(
    len(tasks), batch_size=batch_size, mode='classification',
    dropout=n_dropout,
    model_dir='/content/drive/My Drive/MyProjects/DeepChem/antivirals/models_antivirals/', # output folder for models!
    random_seed=42) # same seed here!


# In[55]:


# Fit trained model
model.fit(train_dataset, nb_epoch=n_epoch, deterministic=True)


# In[56]:


# print statistics
print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
print("Training ROC-AUC Score: %f" % train_scores["mean-roc_auc_score"])
test_scores = model.evaluate(test_dataset, [metric], transformers)
print("Test ROC-AUC Score: %f" % test_scores["mean-roc_auc_score"])


# In[22]:


test_scores


# ### Make prediction with this model

# In[ ]:


predictions = model.predict(dataset_pred)


# In[58]:


import pandas as pd

df = pd.read_csv(pred_data)
df


# In[59]:


# create a dataframe with the predictions
df_preds = pd.DataFrame (columns = ['smiles','ProbClass1'])
df_preds['smiles'] = list(dataset_pred.ids)
df_preds['ProbClass1'] = list(predictions[:,0,1]) # second column = % class 1
df_preds


# In[60]:


merged_inner = pd.merge(left=df, right=df_preds, left_on='smiles', right_on='smiles')
merged_inner


# In[61]:


merged_inner = merged_inner.sort_values(by='ProbClass1', ascending=False)
merged_inner


# In[ ]:


# save the predictions on disk
merged_inner.to_csv('/content/drive/My Drive/MyProjects/DeepChem/antivirals/antivirals_predictions.csv', index=False)

