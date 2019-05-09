'''
Created on May 8, 2019

@author: kwibu
'''
import numpy as np
import matplotlib.pyplot as plt
from model import QCL
from qulacs import Observable

SAMPLE_SIZE = 256
BATCH_SIZE = 10
N_QUBITS = 6
DEPTH = 6
EPOCHS = 256
GRADIENT_METHOD='steepest'
ALPHA = 0.1

loss_lists = []

for batch_size in [1, 10, 100]:
    X = np.random.uniform(-1.0, 1.0, (SAMPLE_SIZE, 1))
    Y = X**2
    Pauli_string = "Z 0"
    observable = Observable(N_QUBITS)
    observable.add_operator(1.0,Pauli_string)
    
    qcl = QCL(N_QUBITS, DEPTH)
    loss_list1 = qcl.fit_regression(X, Y, observable, max_itr = EPOCHS, batch_size = BATCH_SIZE, gradient_method=GRADIENT_METHOD, step_size = ALPHA)
    
    
    pred = []
    for x in X:
        pred.append(qcl.get_expectation(x, qcl.theta, observable))
    """
    plt.scatter(X, pred)
    plt.scatter(X, Y)
    plt.xlabel("x")
    plt.ylabel("f(x)=x^2")
    plt.show()
    """
"""
plt.plot(range(EPOCHS), loss_list1, label='steepest descent')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
"""


qcl = QCL(N_QUBITS, DEPTH)
loss_list2 = qcl.fit_regression(X, Y, observable, max_itr = EPOCHS, batch_size = BATCH_SIZE, gradient_method='quasi-newton', step_size = ALPHA)

plt.plot(range(EPOCHS), loss_list1, label='steepest descent')
plt.plot(range(EPOCHS), loss_list2, label='quasi-newton')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()

pred = []
for x in X:
    pred.append(qcl.get_expectation(x, qcl.theta, observable))
plt.scatter(X, pred)
plt.scatter(X, Y)
plt.show()
