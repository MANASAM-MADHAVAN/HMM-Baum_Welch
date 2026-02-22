import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class HMM:
    def __init__(self, n_states, n_observations):
        self.n_states = n_states
        self.n_observations = n_observations

        self.A = np.random.rand(n_states, n_states)
        self.A /= self.A.sum(axis=1, keepdims=True)

        self.B = np.random.rand(n_states, n_observations)
        self.B /= self.B.sum(axis=1, keepdims=True)

        self.pi = np.random.rand(n_states)
        self.pi /= self.pi.sum()

    def forward(self, O):
        T = len(O)
        alpha = np.zeros((T, self.n_states))

        alpha[0] = self.pi * self.B[:, O[0]]

        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t - 1] * self.A[:, j]) * self.B[j, O[t]]

        return alpha

    def backward(self, O):
        T = len(O)
        beta = np.zeros((T, self.n_states))
        beta[-1] = 1

        for t in reversed(range(T - 1)):
            for i in range(self.n_states):
                beta[t, i] = np.sum(
                    self.A[i] * self.B[:, O[t + 1]] * beta[t + 1]
                )

        return beta

    def baum_welch(self, O, max_iter=10):
        T = len(O)
        likelihoods = []

        for iteration in range(max_iter):
            alpha = self.forward(O)
            beta = self.backward(O)

            xi = np.zeros((T - 1, self.n_states, self.n_states))
            gamma = np.zeros((T, self.n_states))

            for t in range(T - 1):
                denom = np.sum(alpha[t] * beta[t])
                for i in range(self.n_states):
                    numer = alpha[t, i] * self.A[i] * self.B[:, O[t + 1]] * beta[t + 1]
                    xi[t, i] = numer / denom

            gamma = np.sum(xi, axis=2)
            gamma = np.vstack((gamma, np.sum(xi[-1], axis=0)))

            self.pi = gamma[0]

            for i in range(self.n_states):
                self.A[i] = np.sum(xi[:, i, :], axis=0) / np.sum(gamma[:-1, i])

            for i in range(self.n_states):
                for k in range(self.n_observations):
                    mask = (O == k)
                    self.B[i, k] = np.sum(gamma[mask, i]) / np.sum(gamma[:, i])

            likelihood = np.sum(alpha[-1])
            likelihoods.append(likelihood)

            print(f"Iteration {iteration+1}, P(O|λ) = {likelihood}")

        return likelihoods


if __name__ == "__main__":

    O = np.array([0, 1, 2, 1, 0])
    n_states = 2
    n_observations = 3

    model = HMM(n_states, n_observations)
    likelihoods = model.baum_welch(O, max_iter=10)

    print("\nTransition Matrix A:\n", model.A)
    print("\nEmission Matrix B:\n", model.B)
    print("\nInitial Distribution pi:\n", model.pi)
    print("\nFinal Probability P(O | λ):", likelihoods[-1])

    plt.figure()
    plt.plot(range(1, len(likelihoods)+1), likelihoods)
    plt.xlabel("Iterations")
    plt.ylabel("P(O | λ)")
    plt.title("Likelihood vs Iterations")
    plt.show()

    G = nx.DiGraph()

    for i in range(n_states):
        G.add_node(f"S{i}")

    for i in range(n_states):
        for j in range(n_states):
            G.add_edge(f"S{i}", f"S{j}", weight=round(model.A[i][j], 2))

    pos = nx.circular_layout(G)
    labels = nx.get_edge_attributes(G, 'weight')

    plt.figure()
    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title("State Transition Diagram")
    plt.show()