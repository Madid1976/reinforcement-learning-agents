import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_space_size, action_space_size, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay_rate=0.001):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.q_table = np.zeros((state_space_size, action_space_size))

    def choose_action(self, state):
        # Exploration-exploitation trade-off
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randint(0, self.action_space_size - 1) # Explore
        else:
            return np.argmax(self.q_table[state, :]) # Exploit

    def learn(self, state, action, reward, next_state, done):
        # Q-learning update rule
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state, :])

        new_value = old_value + self.learning_rate * (reward + self.discount_factor * next_max - old_value)
        self.q_table[state, action] = new_value

        # Decay exploration rate
        if done:
            self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate - self.exploration_decay_rate)

    def get_q_table(self):
        return self.q_table

    def reset_exploration_rate(self):
        self.exploration_rate = 1.0

if __name__ == "__main__":
    # Dummy environment for demonstration (e.g., a simple 1D grid world)
    class DummyEnvironment:
        def __init__(self, num_states=10, num_actions=2):
            self.num_states = num_states
            self.num_actions = num_actions
            self.current_state = 0
            self.goal_state = num_states - 1

        def reset(self):
            self.current_state = 0
            return self.current_state

        def step(self, action):
            if action == 0: # Move right
                self.current_state = min(self.current_state + 1, self.num_states - 1)
            else: # Move left
                self.current_state = max(self.current_state - 1, 0)

            reward = 1 if self.current_state == self.goal_state else 0
            done = self.current_state == self.goal_state
            return self.current_state, reward, done, {}

    env = DummyEnvironment()
    agent = QLearningAgent(env.num_states, env.num_actions)

    print("Training Q-Learning Agent...")
    num_episodes = 1000
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}: Total Reward = {total_reward}, Exploration Rate = {agent.exploration_rate:.2f}")

    print("\nTraining complete. Learned Q-table:")
    print(agent.get_q_table())

    # Test the trained agent
    print("\nTesting trained agent...")
    state = env.reset()
    done = False
    path = [state]
    while not done:
        action = np.argmax(agent.get_q_table()[state, :]) # Choose best action
        state, reward, done, _ = env.step(action)
        path.append(state)
    print(f"Path taken by agent: {path}")
    print("Q-Learning Agent demonstration finished.")
