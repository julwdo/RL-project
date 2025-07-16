# ChaoticChef Reinforcement Learning Environments

This project develops and analyzes reinforcement learning agents in custom environments simulating a chef navigating city markets to collect ingredients and cook recipes under various constraints.

**Key Components:**
- **Data Collection and Preprocessing:** Extraction and cleaning of pasta recipes from a Kaggle dataset, selecting 25 unique ingredients to construct a 5×5 ingredient grid for the environment.
- **ChaoticChef Environment:** A grid-world where the agent moves to collect ingredients, maximizing dish quality by balancing ingredient combinations and minimizing waste. Includes stochastic slipping to model market chaos.
- **BudgetedChaoticChef Environment:** An extension introducing monetary budgets and ingredient costs, requiring agents to maximize revenue while adhering to budget constraints.
- **Reinforcement Learning Methods:** Training of agents using tabular SARSA, linear function approximation SARSA, and Double Deep Q-Networks (DDQN) to evaluate performance under different state representations and constraints.
- **Evaluation Metrics:** Analysis of episode length, invalid actions, rewards, temporal difference errors, and loss values to assess agent behavior and learning stability.

**Dataset:** Pasta recipes sourced from the [Food Recipes](https://www.kaggle.com/datasets/sarthak71/food-recipes) Kaggle Dataset, filtered and processed for environment compatibility.
