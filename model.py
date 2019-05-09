'''
Created on Apr 30, 2019

@author: kwibu
'''
import numpy as np
import cupy as cp
from scipy import linalg
import itertools
import tqdm
from qulacs import QuantumCircuit, QuantumState

class QCL:
    def __init__(self, n_qubits, depth):
        """
        Quantum Circuit Learning
        https://arxiv.org/abs/1803.00745
        
        Arguments:
        n_qubits: int, number of qubits
        depth: int, depth of unitary gates (see the paper). 
        """
        self.n_qubits = n_qubits
        self.depth = depth
        self.theta = np.random.uniform(0.0, 2*np.pi, (n_qubits, 3*depth))
        def ising_approx(T, trotter_steps=1000):
            """
            Generate a unitary to simulate time evolution of a quantum state with Ising model Hamiltonian:
            H = sum_{j=1}^N (aj*Xj) + sum_{j=1}^N sum_{k=1}^{j-1} (Jjk*Zj*Zk)
            
            Arguments:
            T: int, evolution time
            trotter_steps: int, number of steps for Trotter-Suzuki approximation.
            
            Return:
            unitary: NxN numpy array, unitary for the simulation
            """
            I = np.eye(2)
            X = np.array([[0.0, 1.0], [1.0, 0.0]])
            Z = np.array([[1.0, 0.0], [0.0, -1.0]])
            a = np.random.uniform(-1.0, 1.0, (self.n_qubits))
            J = np.random.uniform(-1.0, 1.0, (self.n_qubits, self.n_qubits))
            
            def multi_tensorprod(*args):
                prod = np.ones(1, dtype=float)
                for mat in args:
                    prod = np.kron(prod, mat)
                return prod
            def multi_matmul(*args):
                for i, mat in enumerate(args):
                    if i == 0:
                        prod = mat
                    else:
                        prod = np.matmul(prod, mat)
                return prod
            for step in range(trotter_steps):
                e1 = [linalg.expm(-1j*(T/trotter_steps)*multi_tensorprod(*[a[j]*X if i == j else I for i in range(self.n_qubits)]))
                      for j in range(self.n_qubits)]
                e2 = [linalg.expm(-1j*(T/trotter_steps)*multi_tensorprod(*[J[j, k]*Z if i == j else Z if i == k else I for i in range(self.n_qubits)]))
                      for j, k in itertools.combinations(range(self.n_qubits), 2)]
                ising_step = multi_matmul(*(e1+e2))
                if step == 0:
                    result = ising_step
                else:
                    result = multi_matmul(*[result, ising_step])
            return result
        self.hamiltonian = ising_approx(10, 100)
        
    def unitary_x(self, circuit, x):
        """
        Prepare the quantum state encoding x: U(x)|0>
        Arguments:
        circuit: qiskit.QuantumCircuit, the target quantum circuit to encode x.
        x: d-dimensional numpy vector, the sample vector to be encoded.
        
        Return:
        circuit: qiskit.QuantumCircuit, the target quantum circuit to encode x.
        """
        
        for i in range(self.n_qubits):
            target_ind = i % x.shape[0] # the target dimension of x to be encoded in the ith qubit
            circuit.add_RZ_gate(i, np.arccos(x[target_ind]**2))
            circuit.add_RY_gate(i, np.arcsin(x[target_ind]))
        return circuit
    
    def unitary_theta(self, circuit, theta):
        """
        Apply the unitary U(theta) to the input state: U(theta)U(x)|0>
        
        Arguments:
        circuit: qiskit.QuantumCircuit, the target quantum circuit to encode x.
        theta: dx1 numpy array (vector), n_qubitsx(3*D) numpy array, the decision variables.
        
        Return:
        circuit: qiskit.QuantumCircuit, the target quantum circuit to encode x.
        """
        
        for d in range(self.depth):
            circuit.add_dense_matrix_gate(list(range(self.n_qubits)), self.hamiltonian)
            for i in range(self.n_qubits):
                circuit.add_U3_gate(i, theta[i, 3*d], theta[i, 3*d+1], theta[i, 3*d+2])
        return circuit
        
    def apply_circuit(self, x, theta):
        """
        Apply unitary gates U(x) and U(theta) to |0>
        
        Argument:
        x: dx1 numpy array (vector), the sample vector to be encoded.
        theta: n_qubitsx(3*D) numpy array, the decision variables.
        
        Return:
        circuit: quantum state after applying the circuit with the given theta.
        """
            
        state = QuantumState(self.n_qubits)
        state.set_zero_state()
        
        # Construct a circuit
        circuit = QuantumCircuit(self.n_qubits)
        circuit = self.unitary_x(circuit, x)
        circuit = self.unitary_theta(circuit, theta)
        
        # apply the circuit to the zero state
        circuit.update_quantum_state(state)
        return state
    
    def get_expectation(self, x, theta, observable):
        """
        Get the expectation value of the observable.
        x: dx1 numpy array (vector), the sample vector to be encoded.
        theta: n_qubitsx(3*D) numpy array, the decision variables.
        
        Return:
        value: float, the expectation value
        """
        state = self.apply_circuit(x, theta)
        value = observable.get_expectation_value(state)
        return value
            
    def fit_regression(self, X, Y, observable, max_itr = 1024, batch_size = 32, gradient_method='steepest', step_size = 0.1, xi = 1.0):
        """
        Fit a regression model given the training data x and y by gradient based optimization. 
        
        Arguments:
        X: mxd numpy array, d-dimensional training data with m samples.
        Y: mx1 numpy array, training label.
        max_itr: int, the maximum number of SGD iterations.
        batch_size: int, if greater than one, the variables are updated with mini-batch SGD.
        gradient_method: string, method to update the variables. 'steepest', 'quasi-newton' are supported. 
        step_size: float, step size for updates with steepest gradient descent. 
        xi: float, constant for stochastic Quasi-Newton method. If 1.0, it is BFGS, if 0.0, it is DFP. 
        
        Return:
        loss_list: list of float, list of losses over iterations.
        """
        loss_list = []
        gradient_list = []
        D_list = [cp.eye(self.n_qubits*3*self.depth)]
        for itr in tqdm.tqdm(range(max_itr)):
            batch_gradient_list = []
            batch_loss_list = []
            for m in range(batch_size):
                ind = np.random.randint(X.shape[0])
                x = X[ind]
                y = Y[ind]
                grad = np.zeros((self.n_qubits, 3*self.depth))
                loss = self.get_expectation(x, self.theta, observable)-y
                batch_loss_list.append(loss**2)
                for i in range(self.n_qubits):
                    for j in range(3*self.depth):
                        plus = self.theta.copy()
                        plus[i, j] += np.pi/2
                        minus = self.theta.copy()
                        minus[i, j] -= np.pi/2
                        B_plus = self.get_expectation(x, plus, observable)
                        B_minus = self.get_expectation(x, minus, observable)
                        grad[i, j] = (B_plus-B_minus)*loss # Partial Gradient
                batch_gradient_list.append(grad)
            mean_grad = sum(batch_gradient_list)/batch_size
            gradient_list.append(mean_grad)
            loss_list.append(sum(batch_loss_list)/batch_size)
            print("loss: %.2f" % loss_list[-1])
            # Update with gradient method
            if gradient_method == 'steepest':
                self.theta -= step_size*mean_grad
            if gradient_method == 'quasi-newton':
                q = cp.reshape(cp.ravel(cp.array(mean_grad - gradient_list[-2])) if itr > 1 else cp.ravel(cp.array(mean_grad)), [-1, 1])
                prev_theta = self.theta.copy()
                self.theta -= cp.asnumpy(cp.reshape(step_size*cp.matmul(D_list[-1], cp.array(cp.transpose(cp.ravel(mean_grad)))), [self.n_qubits, 3*self.depth]))
                p = cp.reshape(cp.ravel(cp.array(self.theta - prev_theta)), [-1, 1])
                t = cp.matmul(cp.matmul(cp.transpose(cp.array(q)), D_list[-1]), q)
                v = p/cp.matmul(cp.transpose(p), q) - cp.matmul(D_list[-1], q)/t
                D = D_list[-1] + cp.matmul(p, cp.transpose(p))/cp.matmul(cp.transpose(p), q)\
                    - cp.matmul(cp.matmul(cp.matmul(D_list[-1], q), cp.transpose(q)), D_list[-1])/cp.dot(cp.matmul(cp.transpose(q), D_list[-1]), q)\
                    + xi*t*cp.matmul(v, cp.transpose(v))
                D_list.append(D)
                
        return loss_list
    
        
        