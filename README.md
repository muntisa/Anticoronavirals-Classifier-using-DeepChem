# Anticoronavirals-Classifier-using-DeepChem
Classifier and predictions for anti-coronavirals drug repurposing

- [antivirals_DeepChem.ipynb](antivirals_DeepChem.ipynb) - the main Jupyter/python notebook: reading datasets, calculating descriptors, build of the best classifier and predictions for drug repurposing.
- [antivirals_SMILES.csv from datasets](datasets/DB_SMILES4prediction.csv) - main dataset to calculate descriptores using SMILES of antiviral compounds; to build the best QSAR model
- [antivirals_SMILES.csv from datasets](datasets/antivirals_SMILES.csv) - external dataset used to calculate descriptors for DrugBank molecules to predict antiviral activity for drug repurposing
- [antivirals_predictions.csv](antivirals_predictions.csv) - the predictions with the best model for drug repurposing


Note:

If you have problem with DeepChem installation, you should use:

```%tensorflow_version 1.x
!wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
!chmod +x Miniconda3-latest-Linux-x86_64.sh
!bash ./Miniconda3-latest-Linux-x86_64.sh -b -f -p /usr/local
!conda install -y -c deepchem -c rdkit -c conda-forge -c omnia deepchem-gpu=2.3.0 python=3.7
import sys
sys.path.append('/usr/local/lib/python3.7/site-packages/')```
