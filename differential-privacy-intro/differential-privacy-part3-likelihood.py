import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Number of all the patients
N_patient = 5

public_patient_data = np.array([True, False, True, True, True])
public_count_true = np.count_nonzero(public_patient_data == True)
#df = pd.DataFrame(public_patient_data.reshape(-1,N_patient), columns=['patient-0', 'patient-1', 'patient-2', 'patient-3', 'patient-4'])
#df['count(True)'] = public_count_true
#df

from itertools import combinations

def get_combinations(N_patient, N_true):
    combination_results_list = []
    count_true_list = []

    comb = combinations(np.arange(N_patient), N_true)
    for i in list(comb):
        combination_results = np.empty(N_patient, dtype=bool)
        combination_results[:] = False
        if len(i) > 0:
            combination_results[np.array(i)] = True
        combination_results_list.append(combination_results)
        count_true_list.append(np.count_nonzero(combination_results == True))
        
    return combination_results_list, count_true_list

def generate_patient_data(N_patient):
    patients_list = []
    count_true_list = []
    
    for i in range(N_patient + 1):
        N_true = N_patient - i
        temp_patients_list, temp_count_true_list = get_combinations(N_patient, N_true)
        patients_list += temp_patients_list
        count_true_list += temp_count_true_list
        
    return patients_list, count_true_list

original_patient_data_list, original_count_true_list = generate_patient_data(N_patient)
#df = pd.DataFrame(original_patient_data_list, columns=['patient-0', 'patient-1', 'patient-2', 'patient-3', 'patient-4'])
#df['count(True)'] = original_count_true_list
#df

def process_patient_data(patient_data, spinner_results):
    patient_data_output = np.copy(patient_data)
    patient_data_output[spinner_results == False] = patient_data[spinner_results == False] != True
    return patient_data_output

def get_processed_patient_data(patient_data, spinner_results_list):
    patient_data_output_list = []
    count_true_list = []

    for i in range(len(spinner_results_list)):
        patient_data_output = process_patient_data(patient_data, spinner_results_list[i])
        patient_data_output_list.append(patient_data_output)
        count_true_list.append(np.count_nonzero(patient_data_output == True))
        
    return patient_data_output_list, count_true_list

def generate_spinner_data(N_patient):
    spinners_list = []
    count_true_list = []
    
    for i in range(N_patient + 1):
        N_true = N_patient - i
        temp_spinners_list, temp_count_true_list = get_combinations(N_patient, N_true)
        spinners_list += temp_spinners_list
        count_true_list += temp_count_true_list
        
    return spinners_list, count_true_list

def get_spinner_probability_list(p_light, N_patient, spinner_count_true_list):
    p_dark = 1.0 - p_light

    spinner_probability_list = []

    for i in spinner_count_true_list:
        prob = pow(p_light, i) * pow(p_dark, N_patient - i)
        spinner_probability_list.append(prob)

    return spinner_probability_list

spinner_data_list, spinner_count_true_list = generate_spinner_data(N_patient)

p_light = 0.9
spinner_probability_list = get_spinner_probability_list(p_light, N_patient, spinner_count_true_list)

#df = pd.DataFrame(spinner_data_list, columns=['spinner-0', 'spinner-1', 'spinner-2', 'spinner-3', 'spinner-4'])
#df['count(True)'] = spinner_count_true_list
#df['probability'] = spinner_probability_list
#df

original_patient_data = original_patient_data_list[1]
output_patient_data_list, output_count_true_list = get_processed_patient_data(original_patient_data, spinner_data_list)
#df = pd.DataFrame(output_patient_data_list, columns=['patient-0', 'patient-1', 'patient-2', 'patient-3', 'patient-4'])
#df['count(True)'] = output_count_true_list
#df['probability'] = spinner_probability_list
#df

for i in range(len(output_patient_data_list)):
    if (public_patient_data == output_patient_data_list[i]).all():
        print(f'[ 1] Original{original_patient_data}')
        print(f'[{i:-2}]  Spinner{spinner_data_list[i]}')
        print(f' ->    Public{public_patient_data}')
        print(f' ** Probability: {spinner_probability_list[i]:.6f}')
        
def get_likelihood_list(original_patient_data_list, public_patient_data, spinner_data_list, spinner_probability_list):
    likelihood_list = []
    spinner_index_list = []
    for j in range(len(original_patient_data_list)):
        original_patient_data = original_patient_data_list[j]
        output_patient_data_list, output_count_true_list = get_processed_patient_data(original_patient_data, spinner_data_list)
        for i in range(len(output_patient_data_list)):
            if (public_patient_data == output_patient_data_list[i]).all():
                likelihood_list.append(spinner_probability_list[i])
                spinner_index_list.append(i)
    return likelihood_list, spinner_index_list

likelihood_list, spinner_index_list = get_likelihood_list(original_patient_data_list, public_patient_data, spinner_data_list, spinner_probability_list)

for i in range(len(original_patient_data_list)):
    spinner_index = spinner_index_list[i]
    print(f'[{i:-2}] Original{original_patient_data_list[i]} [{spinner_index:-2}] Spinner{spinner_data_list[spinner_index]} ** Probability: {likelihood_list[i]:.6f}')
    
def plot_likelihood_distribution(likelihood_list):
    plt.plot(np.arange(len(likelihood_list)), likelihood_list, marker='o')
    plt.title(f'For p_light={p_light}')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.xlabel('original data index')
    plt.ylabel('likelihood')
    plt.show()
    
plot_likelihood_distribution(likelihood_list)

def estimate_original_count_true(original_count_true_list, likelihood_list):
    estimate = 0
    for i in range(len(original_count_true_list)):
        estimate += original_count_true_list[i] * likelihood_list[i]
    return estimate

p_light_list = [0.9, 0.75, 0.6, 0.5]

plt.figure(figsize=(16,4))

for i in range(len(p_light_list)):
    p_light = p_light_list[i]
    spinner_probability_list = get_spinner_probability_list(p_light, N_patient, spinner_count_true_list)
    likelihood_list, _ = get_likelihood_list(original_patient_data_list, public_patient_data, spinner_data_list, spinner_probability_list)
    
    estimate = estimate_original_count_true(original_count_true_list, likelihood_list)
    print(f'Estimate original count true: {estimate:.2f} for p_light={p_light}')
    
    plt.subplot(141+i)
    plt.plot(np.arange(len(likelihood_list)), likelihood_list, marker='o')
    plt.title(f'For p_light={p_light}')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.xlabel('original data index')
    plt.ylabel('likelihood')

plt.show()
