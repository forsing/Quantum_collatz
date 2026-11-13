# Quantum_collatz 



import random
import numpy as np
from pathlib import Path
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

random.seed(39)
np.random.seed(39)



FILES = (
    Path("/data/loto7_4718_k90.csv"),
    Path("/data/loto7_4718_k90_loto_2981.csv"),
    Path("/data/loto7_4718_k90_loto_plus_1737.csv"),
)



def get_collatz_features(numbers):
    combined_val = int(np.sum(numbers))
    if combined_val <= 0:
        combined_val = 1
    parity_vector = []
    n = combined_val
    steps = 0
    for _ in range(6):
        if n % 2 == 0:
            parity_vector.append(0)
            n = n // 2
        else:
            parity_vector.append(1)
            n = 3 * n + 1
        steps += 1
    while n > 1 and steps < 1000:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        steps += 1
    return parity_vector, steps

def run_quantum_collatz_circuit(parity_vector, stopping_time):
    qc = QuantumCircuit(6)
    for idx, bit in enumerate(parity_vector):
        if bit == 1:
            qc.x(idx)
    rotation_angle = float((stopping_time % 360) * np.pi / 180.0)
    for i in range(6):
        qc.ry(rotation_angle, i)
    for i in range(5):
        qc.cx(i, i + 1)
    state = Statevector(qc)
    amplitudes = state.data
    probabilities = np.array(np.abs(amplitudes) ** 2, dtype=np.float64)
    norm = np.sum(probabilities)
    if norm > 0:
        probabilities = probabilities / norm
    return probabilities

def predict_next_combination(filepath):
    if not filepath.exists():
        return None, 0.0, 0.0
    draws = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 7:
                try:
                    draws.append([int(x) for x in parts[:7]])
                except ValueError:
                    continue
    if len(draws) < 15:
        return None, 0.0, 0.0
    
    total_transitions = len(draws) - 1
    train_size = int(total_transitions * 0.7)
    val_size = int(total_transitions * 0.15)
    
    all_features = []
    for i in range(len(draws)):
        p_vec, s_time = get_collatz_features(draws[i])
        amps = run_quantum_collatz_circuit(p_vec, s_time)
        all_features.append(amps)
    all_features = np.array(all_features, dtype=np.float64)
    
    X = all_features[:-1]
    
    target_matrix = np.zeros((len(draws) - 1, 39), dtype=np.float64)
    for idx in range(len(draws) - 1):
        for num in draws[idx + 1]:
            if 1 <= num <= 39:
                target_matrix[idx, num - 1] = 1.0
                
    X_train = X[:train_size]
    y_train = target_matrix[:train_size]
    
    X_val = X[train_size:train_size + val_size]
    y_val = target_matrix[train_size:train_size + val_size]
    
    X_test = X[train_size + val_size:]
    y_test = target_matrix[train_size + val_size:]
    
    weights_train = np.linalg.pinv(X_train).dot(y_train)
    
    val_loss = 0.0
    if len(X_val) > 0:
        val_preds = X_val.dot(weights_train)
        for idx, pred in enumerate(val_preds):
            actual_indices = np.where(y_val[idx] == 1.0)[0] + 1
            actual_nums = set([int(x) for x in actual_indices])
            pred_indices = np.argsort(pred)[-7:] + 1
            pred_nums = set([int(x) for x in pred_indices])
            matches = len(actual_nums.intersection(pred_nums))
            val_loss += (7 - matches)
        val_loss /= len(X_val)
        
    test_loss = 0.0
    if len(X_test) > 0:
        test_preds = X_test.dot(weights_train)
        for idx, pred in enumerate(test_preds):
            actual_indices = np.where(y_test[idx] == 1.0)[0] + 1
            actual_nums = set([int(x) for x in actual_indices])
            pred_indices = np.argsort(pred)[-7:] + 1
            pred_nums = set([int(x) for x in pred_indices])
            matches = len(actual_nums.intersection(pred_nums))
            test_loss += (7 - matches)
        test_loss /= len(X_test)
        
    final_weights = np.linalg.pinv(X).dot(target_matrix)
    last_features = all_features[-1]
    next_distribution = last_features.dot(final_weights)
    
    next_indices = np.argsort(next_distribution)[-7:]
    next_combination = sorted([int(idx + 1) for idx in next_indices])
    
    return next_combination, float(val_loss), float(test_loss)

def main():
    print("Metoda: Collatz (3n+1) Stopping-Time & Parity-Vector Quantum Circuit")
    print("Posebnost: Preslikavanje 6-bitnih pariteta u kvantne amplitude 64 stanja")
    print("----------------------------------------------------------------------")
    for path in FILES:
        name = path.stem
        next_comb, val_metric, test_metric = predict_next_combination(path)
        if next_comb is not None:
            print(f"Fajl: {name}")
            print(f"Validacioni gubici: {val_metric:.4f}")
            print(f"Test gubici: {test_metric:.4f}")
            print(f"NEXT: {next_comb}")
        else:
            print(f"Fajl: {name} nema dovoljno podataka.")

if __name__ == "__main__":
    main()



"""
Metoda: Collatz (3n+1) Stopping-Time & Parity-Vector Quantum Circuit
Posebnost: Preslikavanje 6-bitnih pariteta u kvantne amplitude 64 stanja
----------------------------------------------------------------------

Fajl: loto7_4718_k90
Validacioni gubici: 5.7171
Test gubici: 5.7574
NEXT: [7, x, 11, y, 22, z, 31]

Fajl: loto7_4718_k90_loto_2981
Validacioni gubici: 5.6532
Test gubici: 5.7248
NEXT: [8, x, 17, y, 25, z, 32]

Fajl: loto7_4718_k90_loto_plus_1737
Validacioni gubici: 5.7385
Test gubici: 5.7471
NEXT: [1, x, 11, y, 26, z, 37]
"""
