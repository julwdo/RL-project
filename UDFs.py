import gymnasium as gym
from gymnasium import spaces
from itertools import combinations
from math import sqrt, ceil
import numpy as np
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class ChaoticChef(gym.Env):    
    def __init__(self, p_slip=0.05, max_steps=50, recipes=None, index_to_ingredient=None, verbose=False):
        super().__init__()
        
        if index_to_ingredient is None:
            self.index_to_ingredient = {
                0: "tomato",
                1: "onion",
                2: "cheese",
                3: "lettuce"
                }
        else:
            self.index_to_ingredient = index_to_ingredient
            
        if recipes is None:
            self.recipes = [
                ("salad", {"tomato", "lettuce"}, 4.5),
                ("cheesy mix", {"cheese", "onion"}, 4.9)
                ]
        else:
            self.recipes = recipes
        
        num_ingredients = len(self.index_to_ingredient)
        self.nrows = self.ncols = ceil(sqrt(num_ingredients))
        
        self.p_slip = p_slip
        self.max_steps = max_steps

        self.ingredient_pairs = self._get_ingredient_pairs(self.recipes)
        
        self.actions = {
            0: "up",
            1: "down",
            2: "left",
            3: "right",
            4: "stop"
            }
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Discrete(self.nrows * self.ncols)

        self.seed()
        
        self.verbose = verbose
                
    @staticmethod
    def _get_ingredient_pairs(recipes):
        ingredient_pairs = set()
        for _, ingredients, _ in recipes:
            for pair in combinations(ingredients, 2):
                ingredient_pairs.add(frozenset(pair))
        return ingredient_pairs
    
    def _log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def reset(self, seed=None, options=None):
        self.seed(seed)
        super().reset(seed=seed)

        self.current_pos = self.np_random.choice(self.nrows * self.ncols)
        self.previous_pos = None
        self.collected = {self.current_pos}
        self.steps = 0
        self.terminated = False
        self.truncated = False
        
        self.best_recipe = None
                
        return self.current_pos, {}
    
    def _move(self, pos, action):
        row, col = divmod(pos, self.ncols)
        if action == 0:
            row -= 1
        elif action == 1:
            row += 1
        elif action == 2:
            col -= 1
        elif action == 3:
            col += 1
                
        return row * self.ncols + col
    
    def _is_valid_move(self, action):
        row, col = divmod(self.current_pos, self.ncols)
        if action == 0 and row == 0:
            return False
        if action == 1 and row == self.nrows - 1:
            return False
        if action == 2 and col == 0:
            return False
        if action == 3 and col == self.ncols - 1:
            return False
        new_pos = self._move(self.current_pos, action)
        if new_pos in self.collected:
            return False
        
        return True
    
    def step(self, action):
        available_actions = [a for a in range(self.action_space.n) if a == 4 or self._is_valid_move(a)]
        
        curr_ing = self.index_to_ingredient.get(self.current_pos)
        self._log(f"\nCurrent position: {self.current_pos} ({curr_ing})")
        self._log(f"Available actions: {[(a, self.actions[a]) for a in available_actions]}")
        
        self.steps += 1
        self._log(f"Step: {self.steps}")

        if action not in available_actions:
            if self.steps >= self.max_steps:
                self._log(f"Invalid action {action} ({self.actions[action]}). Max steps reached. Auto cooking.")
                action = 4
                
                self.truncated = True
                reward = self._calculate_cooking_reward() - 2
                
                collected_ings = [self.index_to_ingredient.get(p) for p in self.collected]
                self._log(f"Ingredients: {collected_ings}")
                self._log(f'Best recipe matched: "{self.best_recipe}", Reward: {reward:.2f}')
                
                return self.current_pos, reward, self.terminated, self.truncated, {"info": "AutoCooked"}
            else:
                reward = -1
                self._log(f"Invalid action {action} ({self.actions[action]}). Penalty. Reward: {reward}")
                return self.current_pos, reward, self.terminated, self.truncated, {"info": "InvalidAction"}

        if action != 4 and self.np_random.random() < self.p_slip:
            slip_choices = [a for a in available_actions if a != action and a != 4]
            chosen_action = self.np_random.choice(slip_choices) if slip_choices else action
        else:
            chosen_action = action
            
        slipped = action != chosen_action
        self._log(f"Chosen: {self.actions[action]} ({action}), Executed: {self.actions[chosen_action]} ({chosen_action}), Slipped: {slipped}")
    
        if chosen_action == 4:
            self.terminated = True
            reward = self._calculate_cooking_reward()
            
            collected_ings = [self.index_to_ingredient.get(p) for p in self.collected]
            self._log(f"Cooking. Ingredients: {collected_ings}")
            self._log(f'Best recipe matched: "{self.best_recipe}", Reward: {reward:.2f}')
                
            return self.current_pos, reward, self.terminated, self.truncated, {"info": "Cooked"}

        new_pos = self._move(self.current_pos, chosen_action)
        
        self.previous_pos = self.current_pos
        self.current_pos = new_pos

        self.collected.add(self.current_pos)
        
        new_ing = self.index_to_ingredient.get(self.current_pos)
        self._log(f"Moved to: {self.current_pos} ({new_ing})")
        
        if self.steps >= self.max_steps:
            self._log(f"Max steps reached. Auto cooking.")
            action = 4

            self.truncated = True
            reward = self._calculate_cooking_reward() - 2
            
            collected_ings = [self.index_to_ingredient.get(p) for p in self.collected]
            self._log(f"Ingredients: {collected_ings}")
            self._log(f'Best recipe matched: "{self.best_recipe}", Reward: {reward:.2f}')
            
            return self.current_pos, reward, self.terminated, self.truncated, {"info": "AutoCooked"}
            
        reward = self._calculate_movement_reward()
        self._log(f"Reward: {reward}")

        return self.current_pos, reward, self.terminated, self.truncated, {"info": "Moved"}
    
    def _calculate_movement_reward(self):
        ing1 = self.index_to_ingredient.get(self.previous_pos)
        ing2 = self.index_to_ingredient.get(self.current_pos)
        
        pair = frozenset({ing1, ing2})
        return 2 if pair in self.ingredient_pairs else -1
        
    def _calculate_cooking_reward(self):
        collected_ingredients = {self.index_to_ingredient[pos] for pos in self.collected}
                
        best_score = float('-inf')
        best_recipe = None
        
        for recipe_name, ingredients, recipe_rating in self.recipes:
            common = collected_ingredients.intersection(ingredients)
            waste = collected_ingredients.difference(ingredients)
            
            correct_ratio = len(common) / len(ingredients)
            waste_ratio = len(waste) / len(collected_ingredients)
            
            score = (correct_ratio - waste_ratio) * recipe_rating
                            
            if score > best_score:
                best_score = score
                best_recipe = recipe_name
        
        self.best_recipe = best_recipe
        
        return best_score
    
gym.register(id="ChaoticChef-v0", entry_point=ChaoticChef)

class BudgetedChaoticChef(ChaoticChef):
    def __init__(self,
                 p_slip=0.05,
                 max_steps=50,
                 recipes=None,
                 index_to_ingredient=None,
                 ingredient_to_cost=None,
                 max_budget=10,
                 revenue_multiplier=1.0,
                 verbose=False):
        super().__init__(p_slip=p_slip, max_steps=max_steps, recipes=recipes, index_to_ingredient=index_to_ingredient, verbose=verbose)
        
        if ingredient_to_cost is None:
            self.ingredient_to_cost = {
                "tomato": 2,
                "onion": 1,
                "cheese": 3,
                "lettuce": 2
                }
        else:
            self.ingredient_to_cost = ingredient_to_cost

        self.max_budget = max_budget
        self.revenue_multiplier = revenue_multiplier
                
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.nrows * self.ncols),
            spaces.Box(low=0, high=self.max_budget, shape=(1,), dtype=np.float32)
        ))
        
    def reset(self, seed=None, options=None):
        state, info = super().reset(seed=seed, options=options)
        
        cost = self._get_ingredient_cost(self.current_pos)
        self.budget = self.max_budget - cost
            
        return (state, np.array([self.budget], dtype=np.float32)), info
    
    def _get_ingredient_cost(self, position):
        ingredient = self.index_to_ingredient.get(position)
        return self.ingredient_to_cost.get(ingredient)
        
    def step(self, action):
        available_actions = [a for a in range(self.action_space.n) if a == 4 or self._is_valid_move(a)]
            
        affordable_actions = []
            
        for a in available_actions:
            if a == 4:
                affordable_actions.append(a)
                continue
            
            new_pos = self._move(self.current_pos, a)
            
            cost = self._get_ingredient_cost(new_pos)
            
            if cost <= self.budget:
                affordable_actions.append(a)
                
        curr_ing = self.index_to_ingredient.get(self.current_pos)
        self._log(f"\nCurrent position: {self.current_pos} ({curr_ing})")
        self._log(f"Current budget: {self.budget:.2f}")
        self._log(f"Valid and affordable actions: {[(a, self.actions[a]) for a in affordable_actions]}")
            
        self.steps += 1
        self._log(f"Step: {self.steps}")
            
        if action not in affordable_actions:
            if self.steps >= self.max_steps:
                self._log(f"Invalid action {action} ({self.actions[action]}). Max steps reached. Auto cooking.")
                action = 4
                
                self.truncated = True
                reward = self._calculate_cooking_reward() - 2
                revenue = max(0, reward * self.revenue_multiplier)
                self.budget += revenue
                                    
                collected_ings = [self.index_to_ingredient.get(p) for p in self.collected]
                self._log(f"Ingredients: {collected_ings}")
                self._log(f'Best recipe matched: "{self.best_recipe}", Reward: {reward:.2f}, Revenue: {revenue:.2f}')
                
                self._log(f"New budget: {self.budget:.2f}")
                return (self.current_pos, np.array([self.budget], dtype=np.float32)), reward, self.terminated, self.truncated, {"info": "AutoCooked"}
            else:
                reward = -1
                self._log(f"Invalid action {action} ({self.actions[action]}). Penalty. Reward: {reward}")
                return (self.current_pos, np.array([self.budget], dtype=np.float32)), reward, self.terminated, self.truncated, {"info": "InvalidAction"}
            
        if action != 4 and self.np_random.random() < self.p_slip:
            slip_choices = [a for a in affordable_actions if a != action and a != 4]
            chosen_action = self.np_random.choice(slip_choices) if slip_choices else action
        else:
            chosen_action = action
                
        slipped = action != chosen_action
        self._log(f"Chosen: {self.actions[action]} ({action}), Executed: {self.actions[chosen_action]} ({chosen_action}), Slipped: {slipped}")
        
        if chosen_action == 4:
            self.terminated = True
            reward = self._calculate_cooking_reward()
            revenue = max(0, reward * self.revenue_multiplier)
            self.budget += revenue
            
            collected_ings = [self.index_to_ingredient.get(p) for p in self.collected]
            self._log(f"Cooking. Ingredients: {collected_ings}")
            self._log(f'Best recipe matched: "{self.best_recipe}", Reward: {reward:.2f}, Revenue: {revenue:.2f}')
            
            self._log(f"New budget: {self.budget:.2f}")    
            return (self.current_pos, np.array([self.budget], dtype=np.float32)), reward, self.terminated, self.truncated, {"info": "Cooked"}
        
        new_pos = self._move(self.current_pos, chosen_action)
        
        cost = self._get_ingredient_cost(new_pos)
        
        self.budget -= cost
        
        self.previous_pos = self.current_pos
        self.current_pos = new_pos

        self.collected.add(self.current_pos)
        
        new_ing = self.index_to_ingredient.get(self.current_pos)
        self._log(f"Moved to: {self.current_pos} ({new_ing})")
        self._log(f"Paid cost: {cost}, Remaining budget: {self.budget:.2f}")
        
        if self.steps >= self.max_steps:
            self._log(f"Max steps reached. Auto cooking.")
            action = 4
            
            self.truncated = True
            reward = self._calculate_cooking_reward() - 2
            revenue = max(0, reward * self.revenue_multiplier)
            self.budget += revenue
                        
            collected_ings = [self.index_to_ingredient.get(p) for p in self.collected]
            self._log(f"Ingredients: {collected_ings}")
            self._log(f'Best recipe matched: "{self.best_recipe}", Reward: {reward:.2f}, Revenue: {revenue:.2f}')
            
            self._log(f"New budget: {self.budget:.2f}")
            return (self.current_pos, np.array([self.budget], dtype=np.float32)), reward, self.terminated, self.truncated, {"info": "AutoCooked"}
        
        reward = self._calculate_movement_reward()
        self._log(f"Reward: {reward}")

        return (self.current_pos, np.array([self.budget], dtype=np.float32)), reward, self.terminated, self.truncated, {"info": "Moved"}
    
gym.register(id="BudgetedChaoticChef-v0", entry_point=BudgetedChaoticChef)
    
class SARSAAgent:
    def __init__(self, env, eta=0.01, gamma=0.9,
                 epsilon=1.0, epsilon_min=0.1, epsilon_decay=None):
        self.env = env
        self.eta = eta
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))

        self.logs = {
            "epsilons": [],
            "episode_lengths": [],
            "invalid_actions": [],
            "episode_rewards": [],
            "td_errors": [],
            "truncated": [],
            "final_cooking_rewards": []
            }

    def greedy_policy(self, state):
        return np.argmax(self.Q[state])

    def _choose_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return self.greedy_policy(state)

    def _update(self, state, action, reward, next_state, next_action, done):
        current_q = self.Q[state, action]
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * self.Q[next_state, next_action]
        td_error = target_q - current_q
        self.Q[state, action] += self.eta * td_error
        return td_error

    def train(self, num_episodes=1000, log_every=10):
        if self.epsilon_decay is None:
            self.epsilon_decay = self.epsilon / (num_episodes / 2)

        for ep in tqdm(range(num_episodes), desc="Training"):
            state, _ = self.env.reset()
            done = False
            
            episode_reward = 0
            episode_length = 0
            invalid_count = 0
            td_errors = []
            truncated = False
            
            action = self._choose_action(state)

            while not done:
                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                episode_length += 1
                episode_reward += reward
                
                if info.get("info") == "InvalidAction":
                    invalid_count += 1
                
                done = terminated or truncated
                
                next_action = self._choose_action(next_state)

                td_error = self._update(state, action, reward, next_state, next_action, done)
                td_errors.append(td_error)

                state, action = next_state, next_action

            self.logs["epsilons"].append(self.epsilon)
            self.logs["episode_lengths"].append(episode_length)
            self.logs["invalid_actions"].append(invalid_count)
            self.logs["episode_rewards"].append(episode_reward)
            self.logs["td_errors"].append(np.mean(td_errors))
            self.logs["truncated"].append(truncated)
            self.logs["final_cooking_rewards"].append(reward)
            
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

            if (ep + 1) % log_every == 0:
                print(
                    f"Episode {ep+1} | Epsilon = {self.epsilon:.3f} | Steps = {episode_length} | "
                    f"Invalid Actions = {invalid_count} | Reward = {episode_reward:.2f} | "
                    f"Avg TD Error = {np.mean(td_errors):.4f} | Truncated = {truncated} | "
                    f"Cooking Reward = {reward:.2f}"
                    )
                
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
            )
        
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DDQNAgent:
    def __init__(self, env, state_dim, gamma=0.99, lr=1e-3,
                 epsilon=1, epsilon_min=0.1, epsilon_decay=None,
                 batch_size=64, target_update_freq=10):
        
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        self.state_dim = state_dim
        self.action_dim = env.action_space.n
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.online_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()
        
        self.logs = {
            "epsilons": [],
            "episode_lengths": [],
            "invalid_actions": [],
            "episode_rewards": [],
            "losses": [],
            "truncated": [],
            "final_cooking_rewards": []
            }
        
    def preprocess_state(self, state):
        vec = np.zeros(self.state_dim, dtype=np.float32)
        vec[state] = 1
        return vec
    
    def greedy_policy(self, state):
        state_v = torch.tensor(self.preprocess_state(state)).unsqueeze(0).to(self.device)
        q_values = self.online_net(state_v)
        return q_values.argmax().item()
    
    def _choose_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return self.greedy_policy(state)
    
    def _store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def _update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states_v = torch.tensor(np.array([self.preprocess_state(s) for s in states]), device=self.device)
        actions_v = torch.tensor(actions, device=self.device)
        rewards_v = torch.tensor(rewards, device=self.device)
        next_states_v = torch.tensor(np.array([self.preprocess_state(s) for s in next_states]), device=self.device)
        dones_v = torch.tensor(dones, device=self.device, dtype=torch.bool)
        
        q_values = self.online_net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        
        next_actions = self.online_net(next_states_v).argmax(dim=1)
        next_q_values = self.target_net(next_states_v).gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
        next_q_values[dones_v] = 0.0
        
        expected_q_values = rewards_v + self.gamma * next_q_values
        
        loss = nn.MSELoss()(q_values, expected_q_values.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def train(self, num_episodes=1000, log_every=10):
        if self.epsilon_decay is None:
            self.epsilon_decay = self.epsilon / (num_episodes / 2)
            
        for ep in tqdm(range(num_episodes), desc="Training"):
            state, _ = self.env.reset()
            done = False
            
            episode_reward = 0
            episode_length = 0
            invalid_count = 0
            losses = []
            truncated = False
            
            while not done:
                action = self._choose_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                episode_length += 1
                episode_reward += reward
            
                if info.get("info") == "InvalidAction":
                    invalid_count += 1
                    
                done = terminated or truncated
                
                self._store_transition(state, action, reward, next_state, done)
                
                loss = self._update()
                losses.append(loss if loss is not None else 0.0)
                
                state = next_state 
                
            self.logs["epsilons"].append(self.epsilon)
            self.logs["episode_lengths"].append(episode_length)
            self.logs["invalid_actions"].append(invalid_count)
            self.logs["episode_rewards"].append(episode_reward)
            self.logs["losses"].append(np.mean(losses))
            self.logs["truncated"].append(truncated)
            self.logs["final_cooking_rewards"].append(reward)
            
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
            
            if (ep + 1) % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())
                
            if (ep + 1) % log_every == 0:
                print(
                    f"Episode {ep+1} | Epsilon = {self.epsilon:.3f} | Steps = {episode_length} | "
                    f"Invalid Actions = {invalid_count} | Reward = {episode_reward:.2f} | "
                    f"Avg Loss = {np.mean(losses):.4f} | Truncated = {truncated} | "
                    f"Cooking Reward = {reward:.2f}"
                    )
                
class BudgetedDDQNAgent(DDQNAgent):
    def __init__(self, env, state_dim, **kwargs):
        ingredient_dim = env.observation_space.spaces[0].n
        self.max_budget = float(env.observation_space.spaces[1].high[0])
        state_dim = ingredient_dim + 1

        super().__init__(env=env, state_dim=state_dim, **kwargs)
        self.ingredient_dim = ingredient_dim
        
    def preprocess_state(self, state):
        ingredient, budget = state
        one_hot = np.zeros(self.ingredient_dim, dtype=np.float32)
        one_hot[ingredient] = 1.0
        normalized_budget = np.array([budget[0] / self.max_budget], dtype=np.float32)
        return np.concatenate([one_hot, normalized_budget])
    
class BudgetedLinearSARSAAgent:
    def __init__(self, env, eta=0.01, gamma=0.9, epsilon=1.0, epsilon_min=0.1, epsilon_decay=None):
        self.env = env
        self.eta = eta
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
               
        self.ingredient_dim = env.observation_space.spaces[0].n
        self.max_budget = float(env.observation_space.spaces[1].high[0])
        
        self.action_dim = env.action_space.n
        
        self.theta = np.zeros(self.ingredient_dim + 1 + self.action_dim)
        
        self.logs = {
            "epsilons": [],
            "episode_lengths": [],
            "invalid_actions": [],
            "episode_rewards": [],
            "td_errors": [],
            "truncated": [],
            "final_cooking_rewards": []
            }
    
    def preprocess_state(self, state):
        ingredient, budget = state
        one_hot = np.zeros(self.ingredient_dim, dtype=np.float32)
        one_hot[ingredient] = 1.0
        normalized_budget = np.array([budget[0] / self.max_budget], dtype=np.float32)
        return np.concatenate([one_hot, normalized_budget])
    
    def _featurize(self, state, action):
        state_feat = self.preprocess_state(state)
        action_onehot = np.zeros(self.action_dim, dtype=np.float32)
        action_onehot[action] = 1.0
        return np.concatenate([state_feat, action_onehot])
    
    def _q_value(self, state, action):
        phi = self._featurize(state, action)
        return np.dot(self.theta, phi)
    
    def greedy_policy(self, state):
        q_vals = [self._q_value(state, a) for a in range(self.action_dim)]
        return np.argmax(q_vals)
     
    def _choose_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return self.greedy_policy(state)

    def _update(self, state, action, reward, next_state, next_action, done):
        phi = self._featurize(state, action)        
        current_q = np.dot(self.theta, phi)
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * self._q_value(next_state, next_action)
        td_error = target_q - current_q
        self.theta += self.eta * td_error * phi
        return td_error

    def train(self, num_episodes=1000, log_every=10):
        if self.epsilon_decay is None:
            self.epsilon_decay = self.epsilon / (num_episodes / 2)

        for ep in tqdm(range(num_episodes), desc="Training"):
            state, _ = self.env.reset()
            done = False
            
            episode_reward = 0
            episode_length = 0
            invalid_count = 0
            td_errors = []
            truncated = False
            
            action = self._choose_action(state)

            while not done:
                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                episode_length += 1
                episode_reward += reward
                
                if info.get("info") == "InvalidAction":
                    invalid_count += 1
                
                done = terminated or truncated
                
                next_action = self._choose_action(next_state)

                td_error = self._update(state, action, reward, next_state, next_action, done)
                td_errors.append(td_error)

                state, action = next_state, next_action

            self.logs["epsilons"].append(self.epsilon)
            self.logs["episode_lengths"].append(episode_length)
            self.logs["invalid_actions"].append(invalid_count)
            self.logs["episode_rewards"].append(episode_reward)
            self.logs["td_errors"].append(np.mean(td_errors))
            self.logs["truncated"].append(truncated)
            self.logs["final_cooking_rewards"].append(reward)
            
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

            if (ep + 1) % log_every == 0:
                print(
                    f"Episode {ep+1} | Epsilon = {self.epsilon:.3f} | Steps = {episode_length} | "
                    f"Invalid Actions = {invalid_count} | Reward = {episode_reward:.2f} | "
                    f"Avg TD Error = {np.mean(td_errors):.4f} | Truncated = {truncated} | "
                    f"Cooking Reward = {reward:.2f}"
                    )