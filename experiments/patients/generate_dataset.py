################################################################################
# Generate a dataset and save it to a file.                                    #
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

def generate_dataset(filepath, N_patient, has_disease_rate):
    patient_data = np.random.rand(N_patient) < has_disease_rate
    count_true = np.count_nonzero(patient_data == True)

    df = pd.DataFrame({'has_disease': patient_data})
    df.to_csv(filepath, index=False)
    
    print(f'Generated a dataset: N_patient={N_patient}, count(True)={count_true}')

def print_usage(script_name):
    print(f'Usage: python {script_name} <N_patient> <has_disease_rate>')
    print('\tN_patient: number of all the patients')
    print('\thas_disease_rate: ratio of patients that have disease')
    
################################################################################
# Configuration                                                                #
################################################################################

dataset_dir = 'dataset'

################################################################################
# Main                                                                         #
################################################################################

script_name = os.path.basename(sys.argv[0])

if len(sys.argv) != 3:
    print_usage(script_name)
    sys.exit(-1)

N_patient = int(sys.argv[1])
has_disease_rate = float(sys.argv[2])

if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

filepath = f'{dataset_dir}/dataset_{N_patient}_{has_disease_rate}.csv'
generate_dataset(filepath, N_patient, has_disease_rate)
