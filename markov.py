from typing import Tuple
import numpy as np
import numpy.linalg as la


def simulate_model(matrix: np.ndarray, initial_state: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, list]:
    """
    Simulates markov chain. Returns final state and stddev history

    :param matrix: matrix of traversal probabilities
    :param initial_state: initial state vector
    :param eps: minimum stddev to continue iterations
    :return: tuple of final state and list of stddevs
    """
    iterations_count = 0
    state = initial_state
    history = []

    stddev = eps + 1
    while stddev >= eps:
        iterations_count += 1
        prev_state = state

        # Next step
        state = np.dot(state, matrix)

        # Compute stddev for stop indicator and for history
        stddev = la.norm(state - prev_state)
        history.append(stddev)

    return state, history


def solve_model(matrix: np.ndarray) -> np.ndarray:
    """
    Computes final state analytically. Returns final state

    :param matrix: matrix of traversal probabilities
    :return: final state
    """
    # Searching for eigenvector of transposed matrix
    _, eigenvectors = la.eig(matrix.T)

    # At [:, 0] our target vector is located
    final_state = eigenvectors[:, 0]

    # Do this to make sum(final_state) == 1 (as sum of probabilities must equal to 1)
    final_state /= np.sum(final_state)

    return final_state
