import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class HMM:
    def __init__(self, n_states, n_observations):
        self.N = n_states
        self.M = n_observations

        # Random initialization
        self.A = np.random.rand(self.N, self.N)
        self.A = self.A / self.A.sum(axis=1, keepdims=True)

        self.B = np.random.rand(self.N, self.M)
        self.B = self.B / self.B.sum(axis=1, keepdims=True)

        self.pi = np.random.rand(self.N)
        self.pi = self.pi / self.pi.sum()


    def forward(self, O):
        T = len(O)
        alpha = np.zeros((T, self.N))

        alpha[0] = self.pi * self.B[:, O[0]]

        for t in range(1, T):
            for j in range(self.N):
                alpha[t, j] = np.sum(alpha[t - 1] * self.A[:, j]) * self.B[j, O[t]]

        return alpha


    def backward(self, O):
        T = len(O)
        beta = np.zeros((T, self.N))
        beta[-1] = 1

        for t in reversed(range(T - 1)):
            for i in range(self.N):
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

            xi = np.zeros((T - 1, self.N, self.N))
            gamma = np.zeros((T, self.N))

      
            for t in range(T - 1):
                for i in range(self.N):
                    numerator = (
                        alpha[t, i]
                        * self.A[i, :]
                        * self.B[:, O[t + 1]]
                        * beta[t + 1, :]
                    )
                    denominator = np.sum(numerator)
                    xi[t, i, :] = numerator / denominator


            gamma[:-1] = np.sum(xi, axis=2)
            gamma[-1] = alpha[-1] / np.sum(alpha[-1])

         
            self.pi = gamma[0]

 
            for i in range(self.N):
                for j in range(self.N):
                    self.A[i, j] = np.sum(xi[:, i, j]) / np.sum(gamma[:-1, i])


            for i in range(self.N):
                for k in range(self.M):
                    numerator = 0
                    denominator = np.sum(gamma[:, i])

                    for t in range(T):
                        if O[t] == k:
                            numerator += gamma[t, i]

                    self.B[i, k] = numerator / denominator

            prob = np.sum(alpha[-1])
            likelihoods.append(prob)

        return likelihoods

    def visualize_likelihood(self, likelihoods):
        plt.plot(likelihoods)
        plt.xlabel("Iterations")
        plt.ylabel("P(O | lambda)")
        plt.title("Likelihood vs Iterations")
        plt.show()

    def visualize_states(self):
        G = nx.DiGraph()

        for i in range(self.N):
            for j in range(self.N):
                G.add_edge(f"S{i}", f"S{j}", weight=round(self.A[i, j], 2))

        pos = nx.circular_layout(G)
        nx.draw(G, pos, with_labels=True)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.title("State Transition Diagram")
        plt.show()



if __name__ == "__main__":

    O = np.array([0, 1, 0, 2, 1])

    n_states = 2
    n_observations = 3

    model = HMM(n_states, n_observations)

    likelihoods = model.baum_welch(O, max_iter=10)

    print("\nTransition Matrix A:\n", model.A)
    print("\nEmission Matrix B:\n", model.B)
    print("\nInitial Distribution pi:\n", model.pi)
    print("\nFinal Probability P(O | lambda):", likelihoods[-1])

    model.visualize_likelihood(likelihoods)
    model.visualize_states()