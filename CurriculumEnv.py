import numpy as np
import random 
class CurriculumEnv:
   """Wrapper around environment to modify reward for curriculum learning.

   :param env: Environment to learn in
   :type env: PettingZoo-style environment
   :param lesson: Lesson settings for curriculum learning
   :type lesson: dict
   """

   def __init__(self, env, lesson):
      self.env = env
      self.lesson = lesson

   def fill_replay_buffer(self, memory, opponent):
      """Fill the replay buffer with experiences collected by taking random actions in the environment.

      :param memory: Experience replay buffer
      :type memory: AgileRL experience replay buffer
      """
      print("Filling replay buffer ...")

      pbar = tqdm(total=memory.memory_size)
      while len(memory) < memory.memory_size:
         # Randomly decide whether random player will go first or second
         if random.random() > 0.5:
               opponent_first = False
         else:
               opponent_first = True

         mem_full = len(memory)
         self.reset()  # Reset environment at start of episode
         observation, reward, done, truncation, _ = self.last()

         (
               p1_state,
               p1_state_flipped,
               p1_action,
               p1_next_state,
               p1_next_state_flipped,
         ) = (None, None, None, None, None)
         done, truncation = False, False

         while not (done or truncation):
               # Player 0's turn
               p0_action_mask = observation["action_mask"]
               p0_state, p0_state_flipped = transform_and_flip(observation, player = 0)
               if opponent_first:
                  p0_action = self.env.action_space("player_0").sample(p0_action_mask)
               else:
                  if self.lesson["warm_up_opponent"] == "random":
                     p0_action = opponent.get_action(
                           p0_action_mask, p1_action, self.lesson["block_vert_coef"]
                     )
                  else:
                     p0_action = opponent.get_action(player=0)
               self.step(p0_action)  # Act in environment
               observation, env_reward, done, truncation, _ = self.last()
               p0_next_state, p0_next_state_flipped = transform_and_flip(observation, player = 0)

               if done or truncation:
                  reward = self.reward(done=True, player=0)
                  memory.save_to_memory_vect_envs(
                     np.concatenate(
                           (p0_state, p1_state, p0_state_flipped, p1_state_flipped)
                     ),
                     [p0_action, p1_action, 6 - p0_action, 6 - p1_action],
                     [
                           reward,
                           LESSON["rewards"]["lose"],
                           reward,
                           LESSON["rewards"]["lose"],
                     ],
                     np.concatenate(
                           (
                              p0_next_state,
                              p1_next_state,
                              p0_next_state_flipped,
                              p1_next_state_flipped,
                           )
                     ),
                     [done, done, done, done],
                  )
               else:  # Play continues
                  if p1_state is not None:
                     reward = self.reward(done=False, player=1)
                     memory.save_to_memory_vect_envs(
                           np.concatenate((p1_state, p1_state_flipped)),
                           [p1_action, 6 - p1_action],
                           [reward, reward],
                           np.concatenate((p1_next_state, p1_next_state_flipped)),
                           [done, done],
                     )

                  # Player 1's turn
                  p1_action_mask = observation["action_mask"]
                  p1_state, p1_state_flipped = transform_and_flip(observation, player = 1)
                  if not opponent_first:
                     p1_action = self.env.action_space("player_1").sample(
                           p1_action_mask
                     )
                  else:
                     if self.lesson["warm_up_opponent"] == "random":
                           p1_action = opponent.get_action(
                              p1_action_mask, p0_action, LESSON["block_vert_coef"]
                           )
                     else:
                           p1_action = opponent.get_action(player=1)
                  self.step(p1_action)  # Act in environment
                  observation, env_reward, done, truncation, _ = self.last()
                  p1_next_state, p1_next_state_flipped = transform_and_flip(observation, player = 1)

                  if done or truncation:
                     reward = self.reward(done=True, player=1)
                     memory.save_to_memory_vect_envs(
                           np.concatenate(
                              (p0_state, p1_state, p0_state_flipped, p1_state_flipped)
                           ),
                           [p0_action, p1_action, 6 - p0_action, 6 - p1_action],
                           [
                              LESSON["rewards"]["lose"],
                              reward,
                              LESSON["rewards"]["lose"],
                              reward,
                           ],
                           np.concatenate(
                              (
                                 p0_next_state,
                                 p1_next_state,
                                 p0_next_state_flipped,
                                 p1_next_state_flipped,
                              )
                           ),
                           [done, done, done, done],
                     )

                  else:  # Play continues
                     reward = self.reward(done=False, player=0)
                     memory.save_to_memory_vect_envs(
                           np.concatenate((p0_state, p0_state_flipped)),
                           [p0_action, 6 - p0_action],
                           [reward, reward],
                           np.concatenate((p0_next_state, p0_next_state_flipped)),
                           [done, done],
                     )

         pbar.update(len(memory) - mem_full)
      pbar.close()
      print("Replay buffer warmed up.")
      return memory

   def check_winnable(self, lst, piece):
      """Checks if four pieces in a row represent a winnable opportunity, e.g. [1, 1, 1, 0] or [2, 0, 2, 2].

      :param lst: List of pieces in row
      :type lst: List
      :param piece: Player piece we are checking (1 or 2)
      :type piece: int
      """
      return lst.count(piece) == 3 and lst.count(0) == 1

   def check_vertical_win(self, player):
      """Checks if a win is vertical.

      :param player: Player who we are checking, 0 or 1
      :type player: int
      """
      board = np.array(self.env.env.board).reshape(6, 7)
      piece = player + 1

      column_count = 7
      row_count = 6

      # Check vertical locations for win
      for c in range(column_count):
         for r in range(row_count - 3):
               if (
                  board[r][c] == piece
                  and board[r + 1][c] == piece
                  and board[r + 2][c] == piece
                  and board[r + 3][c] == piece
               ):
                  return True
      return False

   def check_three_in_row(self, player):
      """Checks if there are three pieces in a row and a blank space next, or two pieces - blank - piece.

      :param player: Player who we are checking, 0 or 1
      :type player: int
      """
      board = np.array(self.env.env.board).reshape(6, 7)
      piece = player + 1

      # Check horizontal locations
      column_count = 7
      row_count = 6
      three_in_row_count = 0

      # Check vertical locations
      for c in range(column_count):
         for r in range(row_count - 3):
               if self.check_winnable(board[r : r + 4, c].tolist(), piece):
                  three_in_row_count += 1

      # Check horizontal locations
      for r in range(row_count):
         for c in range(column_count - 3):
               if self.check_winnable(board[r, c : c + 4].tolist(), piece):
                  three_in_row_count += 1

      # Check positively sloped diagonals
      for c in range(column_count - 3):
         for r in range(row_count - 3):
               if self.check_winnable(
                  [
                     board[r, c],
                     board[r + 1, c + 1],
                     board[r + 2, c + 2],
                     board[r + 3, c + 3],
                  ],
                  piece,
               ):
                  three_in_row_count += 1

      # Check negatively sloped diagonals
      for c in range(column_count - 3):
         for r in range(3, row_count):
               if self.check_winnable(
                  [
                     board[r, c],
                     board[r - 1, c + 1],
                     board[r - 2, c + 2],
                     board[r - 3, c + 3],
                  ],
                  piece,
               ):
                  three_in_row_count += 1

      return three_in_row_count

   def reward(self, done, player):
      """Processes and returns reward from environment according to lesson criteria.

      :param done: Environment has terminated
      :type done: bool
      :param player: Player who we are checking, 0 or 1
      :type player: int
      """
      reward = 0
      if done:
         reward = self.lesson["rewards"]["win"]
         
      else:
         agent_three_count = self.check_three_in_row(1 - player)
         opp_three_count = self.check_three_in_row(player)
         if (agent_three_count + opp_three_count) == 0:
               reward = self.lesson["rewards"]["play_continues"]
         else:
               reward = (
                  self.lesson["rewards"]["three_in_row"] * agent_three_count
                  + self.lesson["rewards"]["opp_three_in_row"] * opp_three_count
               )
      return reward

   def last(self):
      """Wrapper around PettingZoo env last method."""
      return self.env.last()

   def step(self, action):
      """Wrapper around PettingZoo env step method."""
      self.env.step(action)

   def reset(self):
      """Wrapper around PettingZoo env reset method."""
      self.env.reset()
      
      
class Opponent:
   """Connect 4 opponent to train and/or evaluate against.

   :param env: Environment to learn in
   :type env: PettingZoo-style environment
   :param difficulty: Difficulty level of opponent, 'random', 'weak' or 'strong'
   :type difficulty: str
   """

   def __init__(self, env, difficulty):
      self.env = env.env
      self.difficulty = difficulty
      if self.difficulty == "random":
         self.get_action = self.random_opponent
      elif self.difficulty == "weak":
         self.get_action = self.weak_rule_based_opponent
      else:
         self.get_action = self.strong_rule_based_opponent
      self.num_cols = 7
      self.num_rows = 6
      self.length = 4
      self.top = [0] * self.num_cols

   def update_top(self):
      """Updates self.top, a list which tracks the row on top of the highest piece in each column."""
      board = np.array(self.env.env.board).reshape(self.num_rows, self.num_cols)
      non_zeros = np.where(board != 0)
      rows, cols = non_zeros
      top = np.zeros(board.shape[1], dtype=int)
      for col in range(board.shape[1]):
         column_pieces = rows[cols == col]
         if len(column_pieces) > 0:
               top[col] = np.min(column_pieces) - 1
         else:
               top[col] = 5
      full_columns = np.all(board != 0, axis=0)
      top[full_columns] = 6
      self.top = top

   def random_opponent(self, action_mask, last_opp_move=None, block_vert_coef=1):
      """Takes move for random opponent. If the lesson aims to randomly block vertical wins with a higher probability, this is done here too.

      :param action_mask: Mask of legal actions: 1=legal, 0=illegal
      :type action_mask: List
      :param last_opp_move: Most recent action taken by agent against this opponent
      :type last_opp_move: int
      :param block_vert_coef: How many times more likely to block vertically
      :type block_vert_coef: float
      """
      if last_opp_move is not None:
         action_mask[last_opp_move] *= block_vert_coef
      action = random.choices(list(range(self.num_cols)), action_mask)[0]
      return action

   def weak_rule_based_opponent(self, player):
      """Takes move for weak rule-based opponent.

      :param player: Player who we are checking, 0 or 1
      :type player: int
      """
      self.update_top()
      max_length = -1
      best_actions = []
      for action in range(self.num_cols):
         possible, reward, ended, lengths = self.outcome(
               action, player, return_length=True
         )
         if possible and lengths.sum() > max_length:
               best_actions = []
               max_length = lengths.sum()
         if possible and lengths.sum() == max_length:
               best_actions.append(action)
      best_action = random.choice(best_actions)
      return best_action

   def strong_rule_based_opponent(self, player):
      """Takes move for strong rule-based opponent.

      :param player: Player who we are checking, 0 or 1
      :type player: int
      """
      self.update_top()

      winning_actions = []
      for action in range(self.num_cols):
         possible, reward, ended = self.outcome(action, player)
         if possible and ended:
               winning_actions.append(action)
      if len(winning_actions) > 0:
         winning_action = random.choice(winning_actions)
         return winning_action

      opp = 1 if player == 0 else 0
      loss_avoiding_actions = []
      for action in range(self.num_cols):
         possible, reward, ended = self.outcome(action, opp)
         if possible and ended:
               loss_avoiding_actions.append(action)
      if len(loss_avoiding_actions) > 0:
         loss_avoiding_action = random.choice(loss_avoiding_actions)
         return loss_avoiding_action

      return self.weak_rule_based_opponent(player)  # take best possible move

   def outcome(self, action, player, return_length=False):
      """Takes move for weak rule-based opponent.

      :param action: Action to take in environment
      :type action: int
      :param player: Player who we are checking, 0 or 1
      :type player: int
      :param return_length: Return length of outcomes, defaults to False
      :type player: bool, optional
      """
      if not (self.top[action] < self.num_rows):  # action column is full
         return (False, None, None) + ((None,) if return_length else ())

      row, col = self.top[action], action
      piece = player + 1

      # down, up, left, right, down-left, up-right, down-right, up-left,
      directions = np.array(
         [
               [[-1, 0], [1, 0]],
               [[0, -1], [0, 1]],
               [[-1, -1], [1, 1]],
               [[-1, 1], [1, -1]],
         ]
      )  # |4x2x2|

      positions = np.array([row, col]).reshape(1, 1, 1, -1) + np.expand_dims(
         directions, -2
      ) * np.arange(1, self.length).reshape(
         1, 1, -1, 1
      )  # |4x2x3x2|
      valid_positions = np.logical_and(
         np.logical_and(
               positions[:, :, :, 0] >= 0, positions[:, :, :, 0] < self.num_rows
         ),
         np.logical_and(
               positions[:, :, :, 1] >= 0, positions[:, :, :, 1] < self.num_cols
         ),
      )  # |4x2x3|
      d0 = np.where(valid_positions, positions[:, :, :, 0], 0)
      d1 = np.where(valid_positions, positions[:, :, :, 1], 0)
      board = np.array(self.env.env.board).reshape(self.num_rows, self.num_cols)
      board_values = np.where(valid_positions, board[d0, d1], 0)
      a = (board_values == piece).astype(int)
      b = np.concatenate(
         (a, np.zeros_like(a[:, :, :1])), axis=-1
      )  # padding with zeros to compute length
      lengths = np.argmin(b, -1)

      ended = False
      # check if winnable in any direction
      for both_dir in board_values:
         # |2x3|
         line = np.concatenate((both_dir[0][::-1], [piece], both_dir[1]))
         if "".join(map(str, [piece] * self.length)) in "".join(map(str, line)):
               ended = True
               break

      # ended = np.any(np.greater_equal(np.sum(lengths, 1), self.length - 1))
      draw = True
      for c, v in enumerate(self.top):
         draw &= (v == self.num_rows) if c != col else (v == (self.num_rows - 1))
      ended |= draw
      reward = (-1) ** (player) if ended and not draw else 0

      return (True, reward, ended) + ((lengths,) if return_length else ())
   
   
def transform_and_flip(observation, player):
   """Transforms and flips observation for input to agent's neural network.

   :param observation: Observation to preprocess
   :type observation: dict[str, np.ndarray]
   :param player: Player, 0 or 1
   :type player: int
   """
   state = observation["observation"]
   # Pre-process dimensions for PyTorch (N, C, H, W)
   state = np.moveaxis(state, [-1], [-3])
   if player == 1:
      # Swap pieces so that the agent always sees the board from the same perspective
      state[[0, 1], :, :] = state[[1, 0], :, :]
   state_flipped = np.expand_dims(np.flip(state, 2), 0)
   state = np.expand_dims(state, 0)
   return state, state_flipped