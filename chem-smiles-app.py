# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 21:13:18 2021

@author: SHIVAM KUMAR TATWA
"""
import numpy as np            #####    IMPORTING NECESSESSORY LIBRARIES  #########
import pandas as pd
import streamlit as st
import pickle
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors


##############################################################################
# FUNCTION TO CALCULATE THE AROMATIC PROPORTION

def AromaticProportion(m):
  aromatic_atoms = [m.GetAtomWithIdx(i).GetIsAromatic() for i in range(m.GetNumAtoms())]
  aa_count = []
  for i in aromatic_atoms:
    if i==True:
      aa_count.append(1)
  AromaticAtom = sum(aa_count)
  HeavyAtom = Descriptors.HeavyAtomCount(m)
  AR = AromaticAtom/HeavyAtom
  return AR

#########################################################################
# FUNCTION TO PROCESS THE INPUT SMILES AND GENERATE THE DESCRIPTOR

def generate(smiles, verbose=False):

    moldata= []
    for elem in smiles:
        mol=Chem.MolFromSmiles(elem)
        moldata.append(mol)

    baseData= np.arange(1,1)
    i=0
    for mol in moldata:

        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_MolWt = Descriptors.MolWt(mol)
        desc_NumRotatableBonds = Descriptors.NumRotatableBonds(mol)
        desc_AromaticProportion = AromaticProportion(mol)

        row = np.array([desc_MolLogP,
                        desc_MolWt,
                        desc_NumRotatableBonds,
                        desc_AromaticProportion])

        if(i==0):
            baseData=row
        else:
            baseData=np.vstack([baseData, row])
        i=i+1

    columnNames=["MolLogP","MolWt","NumRotatableBonds","AromaticProportion"]
    descriptors = pd.DataFrame(data=baseData,columns=columnNames)

    return descriptors

#####################################################
# SETTING IMAGE AS THE TITLE

image = Image.open('chem-smiles-newlogo.jpg')

st.image(image, use_column_width=True)

#####################################################
# CREATING THE NAVBAR WITH BOOTSTRAP LIBRARY


st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #AA08A5;">
  <a class="navbar-brand" href="https://www.linkedin.com/in/shivam-kumar-tatwa-224089202/" target="_blank">Shivam Kumar Tatwa</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav">
      <li class="nav-item active">
        <a class="nav-link disabled" href="#">Home <span class="sr-only">(current)</span></a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="https://github.com/Shivam-Kumar-Tatwa" target="_blank">Github</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="https://www.kaggle.com/shivamkumartatwa" target="_blank">Kaggle</a>
      </li>
    </ul>
  </div>
</nav>
""", unsafe_allow_html=True)

#########################################################
# TITLE AND DESCRIPTION

st.write("""
# Chemical SMILES to Molecular Solubility Prediction Web-Application
**Description**:- This app predicts the **Solubility (LogS)** values of molecules!
Data obtained from the John S. Delaney. [ESOL:  Estimating Aqueous Solubility Directly from Molecular Structure](https://pubs.acs.org/doi/10.1021/ci034243x). ***J. Chem. Inf. Comput. Sci.*** 2004, 44, 3, 1000-1005.
***

**Note**:- ***SMILES (Simplified Molecular Input Line Entry System) is a chemical notation that allows a user to represent a chemical structure in a way that can be used by the computer. SMILES is an easily learned and flexible notation.***
***
""")

st.write(""" 


** Video Introduction**:-""")

video_file = open('streamlit-chem-smiles-app-2021-10-19-14-10-62.webm', 'rb')
video_bytes = video_file.read()
st.video(video_bytes)


##########################################################
# INPUT MOLECULE IN SIDE PANEL

st.sidebar.header('User Input Features')

## Read SMILES input
SMILES_input = "NCCCC\nCCC\nCN"

SMILES = st.sidebar.text_area("SMILES input", SMILES_input)
SMILES = "C\n" + SMILES #Adds C as a dummy, first item
SMILES = SMILES.split('\n')

st.header('Input SMILES')
SMILES[1:] # Skips the dummy first item

## Calculate molecular descriptors
st.header('Molecular Descriptors computed from input SMILES')
X = generate(SMILES)
X[1:] # Skips the dummy first item

#############################################################
# SETTING UP THE PRE-BUILT MODEL

# Reads in saved model
load_model = pickle.load(open('chem_smiles_model.pkl', 'rb'))

# Apply model to make predictions
prediction = load_model.predict(X)
#prediction_proba = load_model.predict_proba(X)

st.header("Predicted values of 'LogS' (Output Value)")
prediction[1:] # Skips the dummy first item


#lst=[]
#lst.append(prediction)
#dfm=pd.DataFrame(lst)
#st.bar_chart(dfm)

##############################################################
# SUCCESS CELEBRATION

st.balloons()

st.success('Congrats! You got the Predicted LogS values!!!')

