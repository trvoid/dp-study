################################################################################
# Read a dataset and process it.                                               #
################################################################################

import sys, os
import numpy as np
import pandas as pd

################################################################################
# Constants                                                                    #
################################################################################

################################################################################
# Functions                                                                    #
################################################################################

def load_dataset(filepath):
    df = pd.read_csv(filepath)
    return df

def generate_spinner_data(N_size, p_light):
    spinner_data = np.random.rand(N_size) < p_light
    return spinner_data
    
def process_patient_data(patient_data, spinner_data):
    patient_data_output = np.copy(patient_data)
    patient_data_output[spinner_data == False] = patient_data[spinner_data == False] != True
    return patient_data_output
    
def print_usage(script_name):
    print(f'Usage: python {script_name} <N_patient> <has_disease_rate>')
    print('\tN_patient: number of all the patients')
    print('\thas_disease_rate: ratio of patients that have disease')
    
################################################################################
# Configuration                                                                #
################################################################################

dataset_dir = 'dataset'
p_light = 0.8

################################################################################
# Main                                                                         #
################################################################################

script_name = os.path.basename(sys.argv[0])

if len(sys.argv) != 3:
    print_usage(script_name)
    sys.exit(-1)

N_patient = int(sys.argv[1])
has_disease_rate = float(sys.argv[2])

filepath = f'{dataset_dir}/dataset_{N_patient}_{has_disease_rate}.csv'
df = load_dataset(filepath)

patient_data = df.iloc[:,0]
count_true = np.count_nonzero(patient_data == True)
print(f'Loaded a dataset: N_patient={N_patient}, count(True)={count_true}')

spinner_data = generate_spinner_data(N_patient, p_light)
patient_data_output = process_patient_data(patient_data, spinner_data)
count_true_output = np.count_nonzero(patient_data_output == True)

