import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_space_size, action_space_size, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay_rate=0.001, min_epsilon=0.01):
        """
        Initializes the Q-Learning agent.

        Args:
            state_space_size (tuple): Dimensions of the state space (e.g., (grid_height, grid_width)).
            action_space_size (int): Number of possible actions.
            learning_rate (float): Alpha (α) in the Q-learning update rule.
            discount_factor (float): Gamma (γ) in the Q-learning update rule.
            epsilon (float): Initial exploration rate.
            epsilon_decay_rate (float): Rate at which epsilon decays.
            min_epsilon (float): Minimum value for epsilon.
        """
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon = min_epsilon

        # Initialize Q-table with zeros
        self.q_table = np.zeros(state_space_size + (action_space_size,))

    def choose_action(self, state):
        """
        Chooses an action using an epsilon-greedy policy.

        Args:
            state (tuple): Current state of the agent.

        Returns:
            int: The chosen action.
        """
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_space_size - 1)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def update_q_table(self, state, action, reward, next_state):
        """
        Updates the Q-table using the Q-learning formula.

        Q(s,a) = Q(s,a) + α [R + γ max_a' Q(s',a') - Q(s,a)]

        Args:
            state (tuple): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (tuple): Next state.
        """
        old_value = self.q_table[state + (action,)]
        next_max = np.max(self.q_table[next_state])

        new_value = old_value + self.learning_rate * (reward + self.discount_factor * next_max - old_value)
        self.q_table[state + (action,)] = new_value

    def decay_epsilon(self):
        """
        Decays the epsilon value.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay_rate)

    def get_q_table(self):
        """
        Returns the current Q-table.
        """
        return self.q_table

if __name__ == "__main__":
    # Example: Simple Grid World
    grid_height = 4
    grid_width = 4
    state_space_size = (grid_height, grid_width)
    action_space_size = 4 # 0: up, 1: down, 2: left, 3: right

    agent = QLearningAgent(state_space_size, action_space_size)

    # Simulate a few steps
    current_state = (0, 0)
    print(f"Initial Q-table for state {current_state}: {agent.get_q_table()[current_state]}")

    # Step 1
    action = agent.choose_action(current_state)
    next_state = (0, 1) # Assume moving right
    reward = -1
    agent.update_q_table(current_state, action, reward, next_state)
    print(f"Q-table after 1 update for state {current_state}: {agent.get_q_table()[current_state]}")
    current_state = next_state

    # Step 2
    action = agent.choose_action(current_state)
    next_state = (1, 1) # Assume moving down
    reward = -1
    agent.update_q_table(current_state, action, reward, next_state)
    print(f"Q-table after 2 updates for state {current_state}: {agent.get_q_table()[current_state]}")
    agent.decay_epsilon()
    print(f"Epsilon after decay: {agent.epsilon:.4f}")

    # More complex example: Training loop (simplified)
    print("\nSimulating a training episode...")
    agent = QLearningAgent(state_space_size, action_space_size, epsilon=0.5, epsilon_decay_rate=0.1, min_epsilon=0.1)
    current_state = (0, 0)
    for _ in range(10):
        action = agent.choose_action(current_state)
        # Simulate environment step (next_state, reward)
        next_state = (random.randint(0, grid_height-1), random.randint(0, grid_width-1))
        reward = random.uniform(-1, 1)
        agent.update_q_table(current_state, action, reward, next_state)
        agent.decay_epsilon()
        current_state = next_state
    print("Simulated training episode complete.")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Sample Q-value: {agent.get_q_table()[0,0,0]:.4f}")
