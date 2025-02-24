# Import necessary modules
import numpy as np
import random
import pygame
import sys
from IPython.display import clear_output
from time import sleep

# initialise Pygame
pygame.init()

# snakeEnv setup
class SnakeEnv:
    def __init__(self, grid_size=10, cell_size=40):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.reset()

    def reset(self):
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]  # initialise snake position
        self.direction = (0, 1)  # initialise position of right
        self.food = self._place_food()
        self.done = False
        self.screen = pygame.display.set_mode((self.grid_size * self.cell_size, self.grid_size * self.cell_size))
        return self.get_state()

    def step(self, action):
        # Update direction
        if action == 0 and self.direction != (1, 0):  # Up
            self.direction = (-1, 0)
        elif action == 1 and self.direction != (-1, 0):  # Down
            self.direction = (1, 0)
        elif action == 2 and self.direction != (0, 1):  # Left
            self.direction = (0, -1)
        elif action == 3 and self.direction != (0, -1):  # Right
            self.direction = (0, 1)

        # Update Head Position
        new_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])

        # Check if the snake hit the wall
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size or
            new_head in self.snake):
            self.done = True
            return self.get_state(), -1, self.done, {}

        # Update Snake Body
        self.snake.insert(0, new_head)
        if new_head == self.food:
            reward = 1
            self.food = self._place_food()  # Place new food
        else:
            reward = 0
            self.snake.pop()  # Remove snake tails

        return self.get_state(), reward, self.done, {}

    def _place_food(self):
        while True:
            food = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if food not in self.snake:
                return food

    def get_state(self):
        return (self.snake[0], self.food, tuple(self.snake))

    def render(self):
        self.screen.fill((0, 0, 0))  # Background color setting
        for body in self.snake:
            pygame.draw.rect(self.screen, (0, 255, 0), (body[1] * self.cell_size, body[0] * self.cell_size, self.cell_size, self.cell_size))
        pygame.draw.rect(self.screen, (255, 0, 0), (self.food[1] * self.cell_size, self.food[0] * self.cell_size, self.cell_size, self.cell_size))
        pygame.display.update()

# Q-learning algothrim
class QLearningAgent:
    def __init__(self, action_space, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.action_space))
        else:
            return np.argmax(self.q_table.get(state, [0.0] * self.action_space))

    def update_q_table(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.action_space
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0] * self.action_space

        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

# train and visualisation
def train_snake(env, agent, num_episodes=1000):
    rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        while not env.done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update_q_table(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            env.render()  # render game interface
            sleep(0.1)  # pause to speculate
        rewards.append(total_reward)
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")
    return rewards

# Main function
if __name__ == "__main__":
    env = SnakeEnv(grid_size=10, cell_size=40)
    agent = QLearningAgent(action_space=4)
    rewards = train_snake(env, agent, num_episodes=100)
    pygame.quit()