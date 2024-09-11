#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import os
import time
import matplotlib.pyplot as plt
from sklearn import metrics
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from efficient_kan import KAN
import collections.abc as c


# In[2]:


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

func_group_names = pd.Series(smarts.keys())
func_group_names


# In[3]:


class FunctionalGroupsDataset(Dataset):
    def __init__(self, func_group, convert_to: str = None) -> None:
        self.convert_to = convert_to
        self.func_group_number = np.where(func_group_names.values==func_group)[0][0]
        self.main_dir = os.path.join('..', 'ALL')
        self.func_group = func_group

        #NIST IDs that passed preprocessing
        preprocessed_data_dir = pd.DataFrame(os.listdir(os.path.join(self.main_dir, 'preprocessed_data')))


        if func_group not in os.listdir(os.path.join(self.main_dir, 'functional_groups')):
            raise ValueError(f'{func_group} is not present in our database.')
        else:
            #All NIST IDs of specific functional group
            func_group_data_dir = pd.DataFrame(os.listdir(os.path.join(self.main_dir, 'functional_groups', func_group)))

        #NIST IDs of specific functional group that passed preprocessing
        to_sample = pd.merge(preprocessed_data_dir, func_group_data_dir, on = [0, 0], how = 'outer', indicator = True).query('_merge=="left_only"')[0]
        
        #Equinumerous dataset of preprocessed functional group NIST IDs and shuffled from every other functional groups
        if len(to_sample) < len(func_group_data_dir):
            func_group_data_dir = func_group_data_dir.sample(len(to_sample))
            self.data = pd.concat([to_sample.sample(len(to_sample)), func_group_data_dir], axis=0)
        else:
            self.data = pd.concat([to_sample.sample(len(func_group_data_dir)), func_group_data_dir], axis=0)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        file_path = os.path.join(self.main_dir, 'preprocessed_data', self.data.iloc[index][0])
        file = pd.read_csv(file_path)
        spectraType = file['spectraType'][0]



        #Converts data
        if not self.convert_to:
            spectra = torch.nan_to_num(torch.tensor(file['y'].values, requires_grad=True)).to(torch.float)

        elif self.convert_to.lower() not in ('absorbance', 'absorbancja', 'transmittance', 'transmitancja'):
            raise ValueError(f'Cant convert to {self.convert_to}.')

        elif self.convert_to.lower() != str(spectraType).lower():
            spectra = torch.nan_to_num(torch.tensor(np.abs(1-file['y'].values), requires_grad=True)).to(torch.float)
        
        else:
            spectra = torch.nan_to_num(torch.tensor(file['y'].values, requires_grad=True)).to(torch.float)

        #Reshapes it as required
        spectra = spectra.reshape(1,1,3106)
        
        #Prevents unknown problem with NaN from before
        funcGroup = torch.nan_to_num(torch.tensor(file['funcGroups'].values[self.func_group_number], requires_grad=True)).to(torch.float)

        return spectra, funcGroup


# In[4]:


func_groups_path = os.path.join('..', 'ALL', 'functional_groups')
func_groups_path


# In[5]:


#Creates hashmaps where data is further retrieved by functional group name and its whether its training or test set

def createHashmaps(test_ratio: float = 0.3,
                    batch_size: int = 128):
    
    '''
    Creates hashmaps containing FTIR data.

    Parameters
    ----------

    test_ratio : float, default 0.3
        Ratio of test dataset.
    
    batch_size : int, default 128
        Size of the batch.
    '''

    func_groups_data, func_groups_datasets, func_groups_dataloaders = {}, {}, {}
    for data_directory in os.listdir(func_groups_path):
        dataset = FunctionalGroupsDataset(data_directory, convert_to='absorbance')

        training_dataset, test_dataset = random_split(dataset, [1 - test_ratio, test_ratio], torch.Generator())

        func_groups_dataloaders[data_directory] = {'training': DataLoader(training_dataset, batch_size = batch_size, shuffle = True), 'test' : DataLoader(test_dataset, batch_size = batch_size, shuffle = False)}

    return func_groups_data, func_groups_datasets, func_groups_dataloaders


# In[6]:


func_groups_data, func_groups_datasets, func_groups_dataloaders = createHashmaps()


# In[7]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[8]:


class CNN_KAN(torch.nn.Module):
    '''
    Functional groups recognition CNN-KAN model.
    '''
    def __init__(self, kan_layers: list) -> None:
        super(CNN_KAN, self).__init__()


        kernel_size_1 = 5
        stride_conv_1 = 1

        stride_pool_1 = 3
        filter_size_1 = 3

        self.conv_1 = torch.nn.Conv1d(1, 10, kernel_size = kernel_size_1, stride = stride_conv_1)

        self.pool_1 = torch.nn.MaxPool1d(filter_size_1, stride_pool_1)

        self.conv_2 = torch.nn.Conv1d(10, 10, kernel_size = kernel_size_1, stride = stride_conv_1)

        self.pool_2 = torch.nn.MaxPool1d(filter_size_1, stride_pool_1)

        self.conv_3 = torch.nn.Conv1d(10, 10, kernel_size = kernel_size_1, stride = stride_conv_1)

        self.pool_3 = torch.nn.MaxPool1d(filter_size_1, stride_pool_1)

        self.kan = KAN([1130] + kan_layers).to(device)


    def forward(self, x):
        bs, c, h, w = x.shape
        x = x.reshape(bs, c, h * w)
        out = self.conv_1(x)
        out = F.tanh(out)
        out = self.pool_1(out)

        out = self.conv_2(out)
        out = F.tanh(out)
        out = self.pool_2(out)
        
        out = self.conv_3(out)
        out = F.tanh(out)
        out = self.pool_3(out)

        out = out.reshape(x.shape[0], 1, out.shape[2] * out.shape[1])

        out = self.kan(out)
        out = F.sigmoid(out)
        return out.squeeze(1)

    
def train(num_epochs: int, 
          loss_func, 
          group: str, 
          data_loaders: dict,
          weight_decay: float, 
          lambda_lr: float, 
          learning_rate: float = 1e-6, 
          kan_layers: c.Iterable = [5, 5, 1], 
          seed: int = 42, 
          plot: bool = False, 
          save: bool = False, 
          disable_verbose: bool = False, 
          iter: int = 0, 
          save_dirpath: str|os.PathLike|bytes = None) -> CNN_KAN:
    
    '''
    Training of the functional groups finding CNN-KAN model

    Parameters
    ----------

    num_epochs : int
        Number of epochs for model learning.
    
    loss_func : Any
        Loss function.

    group : str
        Name of functional group.

    data_loaders : dict
        Data loaders hashmap object.

    weight_decay : float
        Weight decay coefficient.

    lambda_lr: float
        A multiplicative factor for lambda function.

    learning_rate : float, default 1e-6
        Models starting learning rate.

    seed : int, default 42
        Seed for random number generator for replicability.

    plot : bool, default False
        If True then shows the loss plot.

    save : bool, default False
        If True then saves the loss plot.

    disable_verbose : bool, default False
        Whether to disable the entire progressbar wrapper.

    iter : int, default 0
        Number of model testing iteration.

    save_dirpath : str|os.PathLike|bytes, default None
        Directory path to save the plot.
    '''
     
    train_losses, test_losses = [], []


    model = CNN_KAN(kan_layers).to(device)

    torch.manual_seed(seed)

    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda = lambda epoch: lambda_lr ** epoch)


    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.
        with tqdm(data_loaders[group]['training'], disable = disable_verbose) as pbar:
            f1 = 0
            predicted_well_count = np.zeros(1)
            all_func_groups_count = np.zeros(1)
            for y_batch, func_group_batch in pbar:
                y_batch, func_group_batch = y_batch.to(device), func_group_batch.to(device)

                optimizer.zero_grad()
                output = model(y_batch).squeeze(1)
                loss = loss_func(output, func_group_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                for i in range(output.shape[0]):
                    #Counts well predicted functional groups
                    predicted_well = ((torch.round(output[i]) == func_group_batch[i]) * 1 * func_group_batch[i]).item()
                    predicted_well_count += predicted_well
                    all_func_groups_count += func_group_batch[i].item()

                f1 = metrics.f1_score(torch.Tensor.cpu(func_group_batch).detach().numpy(), torch.Tensor.cpu(torch.round(output)).detach().numpy())
                pbar.set_postfix(epoch = epoch, loss = train_loss, accuracy = (predicted_well_count/all_func_groups_count).item(), f1 = f1, lr = optimizer.param_groups[0]['lr'])

            
            train_loss /= len(data_loaders[group]['training'])
            train_losses.append(train_loss)


        model.eval()
        test_loss = 0.

        with torch.no_grad():
            with tqdm(data_loaders[group]['test'], disable = disable_verbose) as tbar:
                f1 = 0
                predicted_well_count = np.zeros(1)
                all_func_groups_count = np.zeros(1)
                for y_batch, func_group_batch in tbar:
                    y_batch, func_group_batch = y_batch.to(device), func_group_batch.to(device)

                    output = model(y_batch).squeeze(1)
                    loss = loss_func(output, func_group_batch)
                    test_loss += loss.item()

                    for i in range(output.shape[0]):
                        predicted_well = ((torch.round(output[i]) == func_group_batch[i]) * 1 * func_group_batch[i]).item()

                        predicted_well_count += predicted_well
                        all_func_groups_count += func_group_batch[i].item()
                    f1 = metrics.f1_score(torch.Tensor.cpu(func_group_batch).detach().numpy(), torch.Tensor.cpu(torch.round(output)).detach().numpy())
                    tbar.set_postfix(epoch = epoch, loss = test_loss, accuracy = (predicted_well_count/all_func_groups_count).item(), f1 = f1)

                test_loss /= len(data_loaders[group]['test'])
                test_losses.append(test_loss)

        scheduler.step()
        

    if plot:
        plt.plot(train_losses, label = 'training loss')
        plt.plot(test_losses, label = 'test loss')
        plt.legend()
        plt.title(f'{group}')
        plt.plot()
        plt.show()

    if save:
        plt.savefig(os.path.join(save_dirpath, f"Learning_progress_{group}_{str(iter)}.png"))

    return model


# In[9]:


def trainingMetrics(model: CNN_KAN, 
                 group: str,
                 data_loaders: dict,
                 iter: int = 0,
                 save_dirpath: str|os.PathLike|bytes = None) -> pd.DataFrame:
    
    '''
    Counts training metrics of the model.

    Parameters
    ----------

    model : CNN
        Model object to see the perfomance of.

    group : str
        Name of functional group.

    data_loaders : dict
        Data loaders hashmap object.

    iter : int, default 0
        Number of model testing iteration.

    save_dirpath : str|os.PathLike|bytes, default None
        Directory path to save the plot.
    '''

    model.eval()
    actual = []
    pred = []
    with torch.no_grad():
        predicted_well_count = np.zeros(1)
        all_func_groups_count = np.zeros(1)

        for y_batch, funcGroup in data_loaders[group]['training']:
            y_batch, funcGroup = y_batch.to(device), funcGroup.to(device)

            outputs = model(y_batch)
            outputs = torch.Tensor.cpu(outputs).detach().numpy()
            funcGroup = torch.Tensor.cpu(funcGroup).detach().numpy()

            pred.extend(outputs.reshape(-1).round())
            actual.extend(funcGroup.reshape(-1))

            for i in range(outputs.shape[0]):
                predicted_well = (np.round(outputs[i]) == funcGroup[i]) * 1 * funcGroup[i]
                    
                predicted_well_count += predicted_well
                all_func_groups_count += funcGroup[i]

        df = pd.concat(
            [
                pd.DataFrame([group]), 
                pd.DataFrame(predicted_well_count / all_func_groups_count), 
                pd.DataFrame(predicted_well_count), 
                pd.DataFrame(all_func_groups_count)
            ], 
            axis = 1, 
            ignore_index = True)
        df.columns = ['funcGroup', 'prediction acc', 'predicted well', 'number of groups']

        confusion_matrix = metrics.confusion_matrix(actual, pred)

        true_negative = confusion_matrix[0][0]
        true_positive = confusion_matrix[1][1]
        false_positive = confusion_matrix[0][1]
        false_negative = confusion_matrix[1][0]

        cfMatrixString= f"EM: {true_positive} {true_negative} {false_positive} {false_negative}"
        
        #True Positive Rate - czulosc
        TPR = true_positive / (true_positive + false_negative)

        #True Negative Rate - swoistosc
        TNR = true_negative / (true_negative + false_positive)

        print(f"TPR: {round(TPR * 100, 2)}%\tTNR: {round(TNR * 100, 2)}%")
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])
        cm_display.plot()
        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
        description=f"Iteration {iter} training. Group {group}. {cfMatrixString} TPR: {round(TPR * 100, 2)}%\tTNR: {round(TNR * 100, 2)}%\taccuracy: {round(accuracy * 100, 2)}%\r\n"
       
        global summary
        summary += description
        
        if save_dirpath is not None:
            plt.savefig(os.path.join(save_dirpath, f"training_{iter}.png"))
        plt.show()
        
        return df


# In[10]:


def testMetrics(model: CNN_KAN, 
                group: str,
                data_loaders: dict,
                iter: int = 0,
                save_dirpath: str|os.PathLike|bytes = None) -> pd.DataFrame:
    
    '''
    Counts test metrics of the model.

    Parameters
    ----------

    model : CNN
        Model object to see the perfomance of.

    group : str
        Name of functional group.

    data_loaders : dict
        Data loaders hashmap object.

    iter : int, default 0
        Number of model testing iteration.

    save_dirpath : str|os.PathLike|bytes, default None
        Directory path to save the plot.
    '''

    model.eval()
    actual = []
    pred = []
    with torch.no_grad():
        lista_grup = np.zeros(1)
        list_func = np.zeros(1)

        for y_batch, funcGroup in data_loaders[group]['test']:
            y_batch, funcGroup = y_batch.to(device), funcGroup.to(device)

            outputs = model(y_batch)
            outputs = torch.Tensor.cpu(outputs).detach().numpy()
            funcGroup = torch.Tensor.cpu(funcGroup).detach().numpy()

            pred.extend(outputs.reshape(-1).round())
            actual.extend(funcGroup.reshape(-1))


            for i in range(outputs.shape[0]):
                predicted_well = (np.round(outputs[i]) == funcGroup[i])*1 * funcGroup[i]
                
                lista_grup += predicted_well
                list_func += funcGroup[i]


        df = pd.concat(
            [
                pd.DataFrame([group]), 
                pd.DataFrame(lista_grup / list_func), 
                pd.DataFrame(lista_grup), 
                pd.DataFrame(list_func)
            ], 
            axis = 1, 
            ignore_index = True)
        df.columns = ['funcGroups', 'prediction acc', 'predicted well', 'number of groups']

        confusion_matrix = metrics.confusion_matrix(actual, pred)

        true_negative=confusion_matrix[0][0]
        true_positive=confusion_matrix[1][1]
        false_positive=confusion_matrix[0][1]
        false_negative=confusion_matrix[1][0]
        
        cfMatrixString= f"EM: {true_positive} {true_negative} {false_positive} {false_negative}"
     
        #True Positive Rate - czulosc
        TPR = true_positive / (true_positive + false_negative)

        #True Negative Rate - swoistosc
        TNR = true_negative / (true_negative + false_positive)

        print(f"TPR: {round(TPR * 100, 2)}%\tTNR: {round(TNR * 100, 2)}%")
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])
        cm_display.plot()
        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
        description = f"Iteration {iter} test. Group {group}. {cfMatrixString} TPR: {round(TPR * 100, 2)}%\tTNR: {round(TNR * 100, 2)}%\taccuracy: {round(accuracy * 100, 2)}%\r\n"
       
        global summary
        summary += description
        
        if save_dirpath is not None:
            plt.savefig(os.path.join(save_dirpath, f"test_{iter}.png"))
        plt.show()

        return df


# In[11]:


crossEnt = torch.nn.BCELoss()
group = 'alkene'

model = train(num_epochs = 2, 
            loss_func = crossEnt,
            group = group,
            data_loaders = func_groups_dataloaders,
            weight_decay = 1e-2,
            lambda_lr = 0.9,
            learning_rate = 1e-2,
            kan_layers = [1, 1, 1],
            seed = 42,
            plot = True,
            save = False,
            disable_verbose = False,
            iter = 0,
            save_dirpath = '.')
model



# In[12]:


total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")


# In[13]:


summary = ''


# In[14]:


def repeatabilityTest(method: str,
                      loss_func,
                      num_iters: int = 10,
                      num_epochs: int = 30):

    '''
    Test to check the repeatability of the model over all functional groups.

    Parameters
    ----------

    method : str
        Name to distinguish results from each other.
    
    loss_func : Any
        Loss function.
    
    num_iters : int, default 10
        Number of iterations for each functional group.

    num_epochs : int, default 30
        Number of epochs for each model training.
    '''

    for group in smarts.keys():
        
        group_WS = group.replace(' ', '')
        save_dirpath = os.path.join('.', 'results', method, group_WS)

    
        if not os.path.exists(save_dirpath):
            os.makedirs(save_dirpath)
            print("Directory created successfully!")
        else:
            print("Directory already exists!")


        for iteration in range(num_iters):

            start = time.time()

            _, _, func_groups_dataloaders = createHashmaps()

            
            model = train(num_epochs = num_epochs, 
                        loss_func = loss_func,
                        group = group,
                        data_loaders = func_groups_dataloaders,
                        weight_decay = 1e-2,
                        lambda_lr = 0.9,
                        learning_rate = 1e-2,
                        kan_layers = [1, 1, 1],
                        seed = 42,
                        plot = True,
                        save = False,
                        disable_verbose = False,
                        iter = 0,
                        save_dirpath = '.')
            
            print(trainingMetrics(model = model, 
                                   group = group,
                                   data_loaders = func_groups_dataloaders,
                                   iter = iteration,
                                   save_dirpath = save_dirpath))
            
            print(testMetrics(model = model, 
                                group = group,
                                data_loaders = func_groups_dataloaders,
                                iter = iteration,
                                save_dirpath = save_dirpath))
            
            global summary

            print(summary)
            print(f'Time per iteration: {time.time() - start}')

            with open(os.path.join(save_dirpath, 'stats.txt'), 'w') as f:
                f.write(summary)


# In[15]:


#Test to check the method working
'''
crossEnt = torch.nn.BCELoss()

repeatabilityTest(method = 'KAN31VIII',
                  loss_func = crossEnt,
                  num_iters = 1,
                  num_epochs = 2)

'''