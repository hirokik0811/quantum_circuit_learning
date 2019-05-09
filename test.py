'''
Created on May 8, 2019

@author: kwibu
'''
import numpy as np
import matplotlib.pyplot as plt
from model import QCL
from qulacs import Observable

SAMPLE_SIZE = 256
BATCH_SIZE = 1
N_QUBITS = 6
DEPTH = 6
EPOCHS = 256
GRADIENT_METHOD='steepest'
ALPHA = 0.04

loss_lists = {}
X_orig = np.random.uniform(-2.0, 2.0, (SAMPLE_SIZE, 1))
Y = X_orig**2
X = X_orig/2
for batch_size in [1]:
    for method in ['steepest']:#['steepest', 'quasi-newton']:
        Pauli_string = "Z 0"
        observable = Observable(N_QUBITS)
        observable.add_operator(1.0,Pauli_string)
        
        qcl = QCL(N_QUBITS, DEPTH)
        loss_lists['%s_bs_%d' % (method, batch_size)] = qcl.fit_regression(X, Y, observable, max_itr = EPOCHS, batch_size = batch_size, gradient_method=method, step_size = ALPHA)
    
        pred = []
        for x in X:
            pred.append(qcl.amp*qcl.get_expectation(x, qcl.theta, observable))
        
        plt.scatter(X, pred)
        plt.scatter(X, Y)
        plt.xlabel("x")
        plt.ylabel("f(x)=x^2")
        plt.savefig('./images/batchsize_%d_method_%s.png' % (batch_size, method))
        plt.close()
        
for key, loss_list in loss_lists.items():
    plt.plot(range(EPOCHS), loss_list, label=key)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.savefig('./images/convergence.png')
plt.close()

