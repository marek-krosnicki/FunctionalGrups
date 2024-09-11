#!/usr/bin/env python
# coding: utf-8

# \#* - borrowed from Enders and coworkers. https://github.com/Ohio-State-Allen-Lab/FTIRMachineLearning/blob/main/cas_inchi.py

# In[29]:


import pandas as pd 
import numpy as np
from rdkit import Chem, RDLogger
import os
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import shutil
import collections.abc as c

RDLogger.DisableLog('rdApp.*')
pd.set_option('future.no_silent_downcasting', True)


# In[2]:


#*
smarts = {'alkane':'[CX4;H0,H1,H2,H4]',
                   'methyl':'[CH3]',
                   'alkene':'[CX3]=[CX3]',
                   'alkyne':'[CX2]#C',
                   'alcohols':'[#6][OX2H]',
                   'amines':'[NX3;H2,H1;!$(NC=O)]', 
                   'nitriles':'[NX1]#[CX2]', 
                   'aromatics':'[$([cX3](:*):*),$([cX2+](:*):*)]',
                   'alkyl halides':'[#6][F,Cl,Br,I]', 
                   'esters':'[#6][CX3](=O)[OX2H0][#6]', 
                   'ketones':'[#6][CX3](=O)[#6]',
                   'aldehydes':'[CX3H1](=O)[#6]', 
                   'carboxylic acids':'[CX3](=O)[OX2H1]', 
                   'ether': '[OD2]([#6])[#6]',
                   'acyl halides':'[CX3](=[OX1])[F,Cl,Br,I]',
                   'amides':'[NX3][CX3](=[OX1])[#6]',
                   'nitro':'[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]',
                   'heterocyclic': '[!#6;!R0]',
                   'aryl chlorides': '[Cl][c]',
                   'carboxylic esters': '[CX3;$([R0][#6]),$([H1R0])](=[OX1])[OX2][#6;!$(C=[O,N,S])]',
                   'alkyl aryl ethers': '[OX2](c)[CX4;!$(C([OX2])[O,S,#7,#15,F,Cl,Br,I])]',
                   'phenols': '[OX2H][c]'}

#Creates Mol structures from SMARTS codes.
func_grp_structs = {func_name : Chem.MolFromSmarts(func_smarts)\
                        for func_name, func_smarts in smarts.items()}


# In[3]:


#*
def identify_functional_groups(inchi) -> (list|None):
    '''
    Identifies funtional groups from InChI.

    Parameters
    ----------

    inchi : Any
        International Chemical Identifier.
    '''
    try:
        #Convert inchi to molecule
        mol = Chem.MolFromInchi(inchi, treatWarningAsError = True)   
        mol_func_grps = []

        #populate the list with binary values
        for _, func_struct in func_grp_structs.items():
            struct_matches = mol.GetSubstructMatches(func_struct)
            contains_func_grp = int(len(struct_matches) > 0)
            mol_func_grps.append(contains_func_grp)
        return mol_func_grps
    
    except:
        return None


# In[4]:


main_dir = os.path.join('..', 'ALL')


# In[5]:


def createFolders() -> None:
    '''
    Creates folders for each functional group.
    '''
    func_groups_path = os.path.join(main_dir, 'functional_groups')
    for func_group in smarts.keys():
        if func_group not in os.listdir(func_groups_path):
            func_group_dir = os.path.join(func_groups_path, func_group)
            os.mkdir(func_group_dir)


# In[6]:


#createFolders()


# In[7]:


def countFuncGroupDirectories() -> pd.DataFrame:
    '''
    Returns the size of each functional group directory(files that passed).
    '''
    func_groups_path = os.path.join(main_dir, 'functional_groups')
    func_groups_count = {}
    for func_group_dir in os.listdir(func_groups_path):
        func_groups_count[func_group_dir] = len(os.listdir(os.path.join(func_groups_path, func_group_dir)))
    return pd.DataFrame(func_groups_count, index = ['functional group count'])


# In[8]:


#countFuncGroupDirectories().T


# In[9]:


mol_dir_path = os.path.join(main_dir, 'mol')


# In[10]:


def identifyFuncGroupsDirectory() -> pd.DataFrame:
    '''
    Returns the distribution of functional group files.
    '''
    functional_groups = {}
    for molfile in os.listdir(mol_dir_path):
        try:
            mol = Chem.MolFromMolFile(os.path.join(mol_dir_path, molfile))
            func_groups = identify_functional_groups(Chem.MolToInchi(mol))
            functional_groups[molfile.split('.')[0]] = func_groups
        
        except:
            continue

    return pd.DataFrame(functional_groups, index = smarts.keys())


# In[11]:


#identified_func_groups = identifyFuncGroupsDirectory()


# In[12]:


#identified_func_groups.sum(axis=1)


# In[13]:


jdx_dir_path = os.path.join(main_dir, 'jdx')


# In[14]:


def normalizeData(data: np.ndarray|pd.Series|pd.DataFrame) -> np.ndarray|pd.Series|pd.DataFrame:
    '''
    Normalizes y axis to [0, 1].

    Parameters
    ----------

    data : ndarray|Series|DataFrame
        Data that is to be normalized.
    '''
    return (data - data.min()) / (data.max() - data.min())


# In[15]:


def umToInverseCmList(um: c.Iterable) -> c.Iterable:
    '''
    Returns a list of μm converted to cm-1.

    Parameters
    ----------

    um : Iterable
        Iterable containing μm data.
    '''
    
    return [1e4/(1.00027*float(i)) for i in um]    


# In[16]:


def umToInverseCm(um: float) -> float:
    '''
    Returns μm converted to cm-1.

    Parameters
    ----------

    um : float
        μm data.
    '''
    return 1e4/(1.00027*um)  


# In[17]:


def yUnitsCounter(unit: str) -> int:
    '''
    Counts y units of choice.

    Parameters
    ----------

    unit : str
        Name of unit to be counted.
    '''
    counter = 0
    for jdx_file_path in os.listdir(jdx_dir_path):
            with open(os.path.join(jdx_dir_path, jdx_file_path), 'r') as jdxfile:
                for _, line in enumerate(jdxfile):
                    if line.startswith(f'##YUNITS={unit}'):
                        counter += 1
            
    return counter


# In[18]:


#print(yUnitsCounter('ABSORBANCE'))


# In[24]:


def preprocessData(save_dirpath: str|os.PathLike|bytes) -> None:
    '''
    Converts x and y units, checks x unit range, normalizes data.

    Parameters
    ----------

    save_dirpath : str|os.PathLike|bytes
        Directory path to save preprocessed data.
    '''
    all_counter = 0
    passed_counter = 0
    for jdx_file_path in os.listdir(jdx_dir_path):
        with open(os.path.join(jdx_dir_path, jdx_file_path), 'r') as jdxfile:
            all_counter += 1
            jdxfile = jdxfile.read().split('\n')
            
            for index, line in enumerate(jdxfile):
                    
                if line.startswith('##XUNITS'):
                    if line.split('##XUNITS=')[1] == 'MICROMETERS':
                        um = True

                    else:
                        um = False


                elif line.startswith('##YUNITS'):
                    #Get spectra type (transmittance/absorbance)
                    spectraType = line.strip("##").split('=')[1]
                    if line.startswith('##YUNITS=TRANSMITTANCE'):
                        transm = True 

                    else:
                        transm = False


                elif line.startswith('##FIRSTX'):
                    #Check first x in the spectra
                    firstX = float(line.strip('##').split('=')[1])


                elif line.startswith('##LASTX'):
                    #Check last x in the spectra
                    lastX = float(line.strip('##').split('=')[1])
                    if um:
                        firstX = umToInverseCm(firstX)
                        lastX = umToInverseCm(lastX)
            
                    reverse = False

                    if lastX < firstX:
                        reverse = True
                        cp = lastX
                        lastX = firstX
                        firstX = cp
                    

                    if firstX > 670:
                        break  
                    
                    if lastX < 3775:                 
                        break

                    passed_counter += 1
                    

                elif line.startswith('##XYDATA'):
                    #Get spectra
                    spectra = jdxfile[index+1:-2]

                    try:
                        x, y = [line.split(' ')[0] for line in spectra], [line.split(' ')[1] for line in spectra]
                    except:
                        print(f'Corrupted file is: {jdx_file_path}')
                        break
                    if um:
                        x = umToInverseCmList(x)
                    
                    if reverse == True:
                        x.reverse()
                        y.reverse()

                        
                    y = normalizeData(CubicSpline(x, y)(np.arange(670, 3776, 1)))
                    if transm == True:
                        y = 1 - y                        
                        spectraType = "Converted_to_absorbance"
                        
                    
                    try:
                        df = pd.DataFrame([y, 
                                            pd.concat(
                                                [identified_func_groups.loc[:, jdx_file_path.split('-IR.jdx')[0]],
                                                pd.Series(
                                                    [np.nan for _ in range(len(y)-len(identified_func_groups.loc[:, jdx_file_path.split('-IR.jdx')[0]]))])
                                                ]
                                                ), 
                                                [spectraType] + [np.nan for _ in range(len(y)-1)]
                                            ], 
                                            index = ['y', 'funcGroups', 'spectraType']
                                        ).T
                        
                        df.to_csv(f"{os.path.join(save_dirpath, jdx_file_path.split('-IR.jdx')[0])}.csv", index=False)

                    except:
                        break
                    
                    break
                
    print(f'Number of all files: {all_counter}')
    print(f'Number of passed files: {passed_counter}')


# In[22]:


print(os.path.join(main_dir, 'preprocessed_data'))


# In[25]:


#preprocessData(os.path.join(main_dir, 'preprocessed_data'))


# In[26]:


preprocessed_data_path = os.path.join(main_dir, 'preprocessed_data')
functional_groups_path = os.path.join(main_dir, 'functional_groups')


# In[27]:


def distributeDataIntoDirs() -> None:
    '''
    Distributes spectroscopic files into directories of specific functional groups.
    '''
    for file_path in os.listdir(preprocessed_data_path):
        file = pd.read_csv(os.path.join(preprocessed_data_path, file_path))
        funcGroups = file.loc[:,'funcGroups'].dropna()
        funcGroups =  (funcGroups.astype(bool) * pd.Series(list(smarts.keys()))).replace('', np.nan).dropna().reset_index(drop=True)
        for funcGroup in funcGroups:
            shutil.copyfile(os.path.join(preprocessed_data_path, file_path), os.path.join(functional_groups_path, funcGroup, file_path))


# In[30]:


#distributeDataIntoDirs()

