# Anticoronavirals-Classifier-using-DeepChem
Classifier and predictions for anti-coronavirals drug repurposing

- [antivirals_DeepChem.ipynb](antivirals_DeepChem.ipynb) - the main Jupyter/python notebook: reading datasets, calculating descriptors, build of the best classifier and predictions for drug repurposing.
- [antivirals_SMILES.csv from datasets](datasets/DB_SMILES4prediction.csv) - main dataset to calculate descriptores using SMILES of antiviral compounds; to build the best QSAR model
- [antivirals_SMILES.csv from datasets](datasets/antivirals_SMILES.csv) - external dataset used to calculate descriptors for DrugBank molecules to predict antiviral activity for drug repurposing
- [antivirals_predictions.csv](antivirals_predictions.csv) - the predictions with the best model for drug repurposing

Note for GColab execution:
[2022]

Replace the first cell with installs with:

```! pip install --quiet deepchem```

The new tensorflow 2 has no set_random_seed(), so you should replace the command:

```tf.set_random_seed(42)```

with 

```tf.random.set_seed(42)```

[OLD]

If you have problem with DeepChem installation, you should use:

```%tensorflow_version 1.x
!wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
!chmod +x Miniconda3-latest-Linux-x86_64.sh
!bash ./Miniconda3-latest-Linux-x86_64.sh -b -f -p /usr/local
!conda install -y -c deepchem -c rdkit -c conda-forge -c omnia deepchem-gpu=2.3.0 python=3.7
import sys
sys.path.append('/usr/local/lib/python3.7/site-packages/')
```

### Publication
Drugs Repurposing Using QSAR, Docking and Molecular Dynamics for Possible Inhibitors of the SARS-CoV-2 Mpro Protease, cited as Molecules 2020, 25(21), 5172; DOI: 10.3390/molecules25215172 | **Free PDF**: https://www.mdpi.com/1420-3049/25/21/5172

### Publication groups/institutions
- Grupo de Bio-Quimioinformática, Universidad de Las Américas, Quito 170513, Ecuador
- Facultad de Ingeniería y Ciencias Aplicadas, Universidad de Las Américas, Quito 170513, Ecuador
- Faculty of Computer Science, Centre for Information and Communications Technology Research (CITIC), University of A Coruna, 15007 A Coruña, Spain
- Biomedical Research Institute of A Coruña (INIBIC), University Hospital Complex of A Coruna (CHUAC), 15006 A Coruña, Spain
- Centro de Investigación Genética y Genómica, Facultad de Ciencias de la Salud Eugenio Espejo, Universidad UTE, Quito 170129, Ecuador
- Latin American Network for Implementation and Validation of Clinical Pharmacogenomics Guidelines (RELIVAF-CYTED), 28029 Madrid, Spain
- Carrera de Enfermería, Facultad de Ciencias de la Salud, Universidad de Las Américas, Quito 170513, Ecuador
- Escuela de Ciencias Físicas y Matemáticas, Universidad de Las Américas, Quito 170513, Ecuador
