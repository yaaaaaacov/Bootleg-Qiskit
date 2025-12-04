import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import itertools

class Gate: 
    def __init__(self, name, matrix, n_qbits):
        self.name = name
        self.matrix = matrix
        self.n_qbits = n_qbits

    def __repr__(self):
        return f"{self.name}"
    
class HGate(Gate):
    def __init__(self, n_targets = 1):
        H = 1/np.sqrt(2) * np.array([[1, 1], 
                                     [1,-1]], dtype = np.float32)
        matrix = H.copy()
        if n_targets > 1:
            for _ in range(1, n_targets):
                matrix = np.kron(matrix, H)
        super().__init__("H", matrix, n_qbits = n_targets)

class XGate(Gate):
    def __init__(self, n_targets = 1):
        X = np.array([[0, 1], 
                      [1, 0]], dtype = np.float32)
        matrix = X.copy()
        if n_targets > 1:
            for _ in range(1, n_targets):
                matrix = np.kron(matrix, X)
        super().__init__("X", matrix, n_qbits = n_targets)

class YGate(Gate):
    def __init__(self, n_targets = 1):
        Y = np.array([[0, -1j], 
                      [1j, 0]], dtype = np.complex64)
        matrix = Y.copy()
        if n_targets > 1:
            for _ in range(1, n_targets):
                matrix = np.kron(matrix, Y)
        super().__init__("Y", matrix, n_qbits = n_targets)

class ZGate(Gate):
    def __init__(self, n_targets = 1):
        Z = np.array([[1, 0], 
                      [0, -1]], dtype = np.float32)
        matrix = Z.copy()
        if n_targets > 1:
            for _ in range(1, n_targets):
                matrix = np.kron(matrix, Z)
        super().__init__("Z", matrix, n_qbits = n_targets)

class RXGate(Gate):
    def __init__(self, theta, n_targets = 1):
        RX = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)], 
                       [-1j*np.sin(theta/2), np.cos(theta/2)]], dtype = np.complex64)
        matrix = RX.copy()
        if n_targets > 1:
            for _ in range(1, n_targets):
                matrix = np.kron(matrix, RX)
        super().__init__(f"RX({np.round(theta/np.pi,2)}π)", matrix, n_qbits = 1)

class RYGate(Gate):
    def __init__(self, theta, n_targets = 1):
        RY = np.array([[np.cos(theta/2),-np.sin(theta/2)], 
                       [np.sin(theta/2), np.cos(theta/2)]], dtype = np.float32)
        matrix = RY.copy()
        if n_targets > 1:
            for _ in range(1, n_targets):
                matrix = np.kron(matrix, RY)
        super().__init__(f"RY({np.round(theta/np.pi,2)}π)", matrix, n_qbits = 1)

class RZGate(Gate):
    def __init__(self, theta, n_targets = 1):
        RZ = np.array([[np.exp(-1j*theta/2),0], 
                       [0, np.exp(1j*theta/2)]], dtype = np.complex64)
        matrix = RZ.copy()
        if n_targets > 1:
            for _ in range(1, n_targets):
                matrix = np.kron(matrix, RZ)
        super().__init__(f"RZ({np.round(theta/np.pi,2)}π)", matrix, n_qbits = 1)

class MeasureGate():
    def __init__(self, basis = None, name = None):
        if basis is None:
            basis = np.array([[1,0],
                              [0,-1]], dtype = np.float32) # computational basis (Pauli-Z)
            name = 'Z'
        self.basis = basis
        self.name = name
        self.n_qbits = len(basis)
    def __repr__(self):
        return "\\" + self.name + "/"
    
class Visualizer:
    def draw(circuit):
        n_steps = len(circuit.ops)

        slices = np.full(shape = (n_steps, circuit.n_qbits + circuit.n_cbits), fill_value = None)
        for t in range(len(circuit.ops)):
            if circuit.ops[t]["controls"] is not None:
                low = min(circuit.ops[t]["controls"] + circuit.ops[t]["targets"])
                high = max(circuit.ops[t]["controls"] + circuit.ops[t]["targets"])
                slices[t][low:high] = '|'
                slices[t][circuit.ops[t]["controls"]] = '●'

            slices[t][circuit.ops[t]["targets"]] = str(circuit.ops[t]["gate"])

        
        slices = slices.T
        str_lengths = np.vectorize(lambda x: len(str(x)) if x is not None else 0)(slices)
        col_lengths = np.amax(str_lengths, axis = 0)

        for q in range(circuit.n_qbits):
            wire = f"q{q}" + " " * (4 - len(str(q)))
            for t in range(n_steps):
                wire += "-" * (col_lengths[t] + 2) if str_lengths[q,t] == 0 else f"-{slices[q,t]}-" + "-" * (col_lengths[t] - len(slices[q,t]))
            
            print(wire)
        print('')

    def state_vector(psi, tol = 1e-12):
        n = int(np.log2(len(psi)))
        out = []
        first = True
        for i, amp in enumerate(psi):
            if abs(amp) < tol:
                continue

            bits = format(i, f'0{n}b')

            # ---- Determine sign ----
            if first:
                sign = ""
            else:
                sign = " + " if amp.real > 0 and amp.imag == 0 else \
                    " + " if amp.imag > 0 and amp.real == 0 else \
                    " + " if np.real(amp) > 0 else " - "

            # Use absolute amplitude after deciding sign
            a = abs(amp)

            # ---- Format coefficient ----
            if np.isclose(amp.imag, 0):  
                # purely real
                if np.isclose(a, 1):
                    coeff = ""       # 1 → omit
                else:
                    coeff = f"{a:.4f}"
            elif np.isclose(amp.real, 0):
                # purely imaginary
                if np.isclose(a, 1):
                    coeff = "i"      # 1j → i
                else:
                    coeff = f"{a:.4f}i"
            else:
                # fully complex
                coeff = f"({amp.real:.4f}{amp.imag:+.4f}i)"

            term = f"{sign}{coeff}|{bits}⟩"
            out.append(term)
            first = False
        
        return "".join(out)
    
    def bitstring_histogram(outputs, title = ''):
        runs, n_cbits = outputs.shape

        # Convert rows like [0., 0., 1.] → "001"
        bitstrings = [''.join(str(int(b)) for b in row) for row in outputs]

        # Count the observed frequencies
        counts = Counter(bitstrings)

        # Generate all possible bitstrings of length n_cbits
        all_bitstrings = [''.join(bits) for bits in itertools.product('01', repeat=n_cbits)]

        # Frequencies for all bitstrings (0 if not observed)
        frequencies = [counts[b] for b in all_bitstrings]

        # Plot
        plt.figure(figsize=(10, 4))
        plt.bar(all_bitstrings, frequencies)
        plt.xlabel("Measurement outcome")
        plt.ylabel("Frequency")
        plt.title(title)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
    
        return frequencies
    
            
class Circuit:
    def __init__(self, n_qbits, n_cbits):
        self.n_qbits = n_qbits # qubit channels
        self.n_cbits = n_cbits # classical channels
        self.ops = [] # order of operations

    def add_gate(self, gate, targets, controls = None):
        self.ops.append({
            "gate": gate,
            "targets": targets,
            "controls": controls
        })

    def remove_gate(self, idx):
        self.ops.remove(idx)

    def measure(self, qbits, cbits):
        self.ops.append({
            "gate": MeasureGate(),
            "targets": qbits,
            "controls": None,
            "cbits": cbits
        })
    
    def draw(self):
        Visualizer.draw(self)

class Simulator:
    def __init__(self, circuit):
        self.circuit = circuit
        self.state = np.zeros(2**circuit.n_qbits, dtype = np.float32)
        self.state[0] = 1
        self.cbits = np.zeros(circuit.n_cbits, dtype = np.uint8)

    def reset_state(self, state_vector = None, normalize = True):
        if state_vector is None:
            state_vector = np.zeros(2**self.circuit.n_qbits, dtype = np.float32)
            state_vector[0] = 1
        if normalize:
            state_vector = state_vector / np.linalg.norm(state_vector)
        self.state = state_vector

    def expand(self, matrix, targets, controls = None):

        n = self.circuit.n_qbits
        k = int(np.log2(len(matrix)))

        # pad operator with identities
        U = matrix.copy()
        if k < n:
            for _ in range(n - k):
                U = np.kron(U, np.eye(2, dtype = np.uint8))
        
        # permute operator to act on the correct qbits
        P = np.zeros_like(U)
        for i in range(2**n):
            bits = [(i >> b) & 1 for b in reversed(range(n))]  # MSB first
            new_bits = [0]*n

            # move source qubits (MSB) to targets
            for s, t in zip(range(k), targets):
                new_bits[t] = bits[s]

            # fill in the remaining qubits in order
            remaining_source = [q for q in range(n) if q not in range(k)]
            remaining_target = [q for q in range(n) if q not in targets]
            for s, t in zip(remaining_source, remaining_target):
                new_bits[t] = bits[s]

            # convert back to integer index
            j = sum(b << (n-1-k) for k, b in enumerate(new_bits))
            P[j, i] = 1
           
        U = P @ U @ P.T
        
        # make controlled
        if controls is not None:
            multi_projector = 1
            projector = np.array([[0,0],[0,1]], dtype = np.uint8)

            for i in range(self.circuit.n_qbits):
                if i in controls:
                    multi_projector = np.kron(multi_projector, projector)
                else:
                    multi_projector = np.kron(multi_projector, np.eye(2))
            
            U = np.eye(len(U), dtype = np.uint8) - multi_projector + multi_projector @ U
        
        return U

    def run(self, show_state = True):
        state = self.state
        for t in range(len(self.circuit.ops)):
            op = self.circuit.ops[t]

            if isinstance(op["gate"], MeasureGate):
                basis = op["gate"].basis
                cbits = op["cbits"]

                projectors = [np.outer(basis[:,col], basis[:,col].conj()) for col in range(len(basis))]
                projectors_exp = [self.expand(P, op["targets"]) for P in projectors]

                probabilities = np.array([np.real(self.state.conj().T @ P @ self.state) for P in projectors_exp], dtype = np.float32)
                probabilities /= probabilities.sum() # normalize to prevent floating-point error issues
                result = np.random.choice(np.arange(len(basis), dtype = np.uint8), p = probabilities)
                
                self.cbits[cbits] = np.array([int(b) for b in format(result, f"0{len(cbits)}b")], dtype = np.uint8)
                self.state = projectors_exp[result] @ self.state / np.sqrt(probabilities[result])

            else:
                self.state = self.expand(op["gate"].matrix, op["targets"], op["controls"]) @ self.state
                
        if show_state:
            print(Visualizer.state_vector(state) + " ---> " + Visualizer.state_vector(self.state))
    
    def run_measurements(self, runs, init_state = None, cbits_idx = None, title = ''):
        if cbits_idx is None:
            cbits_idx = np.ones_like(self.cbits, dtype = np.uint8)
        outputs = np.zeros((runs,len(cbits_idx)), dtype = np.uint8)
        for i in range(runs):
            self.reset_state(init_state)
            self.run(show_state=False)
            outputs[i] = self.cbits[cbits_idx]
        frequencies = Visualizer.bitstring_histogram(outputs, title)
        return outputs, frequencies
                
        
        
        
            


