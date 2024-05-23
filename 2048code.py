import random
import os
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def initialize_board():
    board = [[0 for _ in range(4)] for _ in range(4)]
    add_new_tile(board)
    add_new_tile(board)
    return board

def add_new_tile(board):
    empty_cells = [(i, j) for i in range(4) for j in range(4) if board[i][j] == 0]
    if empty_cells:
        row, col = random.choice(empty_cells)
        board[row][col] = 2 if random.random() < 0.9 else 4

def is_game_over(board):
    for row in board:
        if 0 in row:
            return False
        for i in range(3):
            if row[i] == row[i + 1]:
                return False
    for col in range(4):
        for i in range(3):
            if board[i][col] == board[i + 1][col]:
                return False
    return True

def merge(row):
    merged_row = [num for num in row if num != 0]
    for i in range(len(merged_row) - 1):
        if merged_row[i] == merged_row[i + 1]:
            merged_row[i] *= 2
            merged_row[i + 1] = 0
    merged_row = [num for num in merged_row if num != 0]
    merged_row.extend([0] * (len(row) - len(merged_row)))
    return merged_row

def move_left(board):
    for i in range(4):
        board[i] = merge(board[i])

def move_right(board):
    for i in range(4):
        board[i] = merge(board[i][::-1])[::-1]

def move_up(board):
    for i in range(4):
        col = [board[j][i] for j in range(4)]
        merged_col = merge(col)
        for j in range(4):
            board[j][i] = merged_col[j]

def move_down(board):
    for i in range(4):
        col = [board[j][i] for j in range(4)]
        merged_col = merge(col[::-1])[::-1]
        for j in range(4):
            board[j][i] = merged_col[j]

def display_board(board):
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\n".join([" ".join([str(num).rjust(5) for num in row]) for row in board]))

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, min(batch_size, len(self.memory)))
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def calculate_reward(prev_board, next_board):
    prev_sum = sum([sum(row) for row in prev_board])
    next_sum = sum([sum(row) for row in next_board])
    return next_sum - prev_sum

def main():
    state_size = 16  # 4x4 board size
    action_size = 4  # WASD actions
    agent = DQNAgent(state_size, action_size)
    board = initialize_board()
    state = np.array(board).reshape(1, state_size)
    while True:
        display_board(board)
        if is_game_over(board):
            print("Game over!")
            break
        action = agent.act(state)
        next_board = np.array(board)
        if action == 0:
            move_up(next_board)
        elif action == 1:
            move_left(next_board)
        elif action == 2:
            move_down(next_board)
        elif action == 3:
            move_right(next_board)
        else:
            print("Invalid action!")
            continue
        next_state = next_board.reshape(1, state_size)
        reward = calculate_reward(board, next_board)
        agent.remember(state, action, reward, next_state, is_game_over(next_board))
        state = next_state
        board = next_board.tolist()
        add_new_tile(board)
        agent.replay(batch_size=32)

if __name__ == "__main__":
    main()

