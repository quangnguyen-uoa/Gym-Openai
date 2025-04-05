import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from agent import TRPOAgent
import simple_driving
import time
import os

def train_agent(env_name="SimpleDriving-v0", 
                seed=42, 
                batch_size=5000, 
                iterations=100,
                max_episode_length=250, 
                verbose=True,
                save_path="agent.pth"):
    """
    Train an agent to navigate to the goal marker.
    
    Parameters:
    -----------
    env_name : str
        Name of the environment
    seed : int
        Random seed for reproducibility
    batch_size : int
        Number of steps to collect before updating the policy
    iterations : int
        Number of policy updates
    max_episode_length : int
        Maximum number of steps per episode
    verbose : bool
        Whether to print training progress
    save_path : str
        Path to save the trained model
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print("Starting training process...")
    print(f"Environment: {env_name}")
    print(f"Seed: {seed}")
    print(f"Batch size: {batch_size}")
    print(f"Iterations: {iterations}")
    print(f"Max episode length: {max_episode_length}")
    
    # Create the environment with discrete actions
    env = gym.make(env_name, isDiscrete=True)
    env.seed(seed)
    
    # Get the actual observation space dimensions
    # The environment defines 8 dimensions but getExtendedObservation() returns only 2
    obs_dim = 2  # [ballPosInCar[0], ballPosInCar[1]]
    action_dim = 1  # Discrete action (0-8)
    
    print(f"Observation space dimension: {obs_dim}")
    print(f"Action space dimension: {action_dim}")
    
    # Define the policy network
    # Input: 2-dimensional observation space (goal position relative to car)
    # Output: 1-dimensional action space (discrete action)
    policy = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, 64), 
        torch.nn.Tanh(),
        torch.nn.Linear(64, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, action_dim)
    )
    
    print("Policy network architecture:")
    print(policy)
    
    # Create the agent
    agent = TRPOAgent(policy=policy)
    
    # Train the agent
    print("\nStarting agent training...")
    agent.train(
        env_name=env_name,
        seed=seed,
        batch_size=batch_size,
        iterations=iterations,
        max_episode_length=max_episode_length,
        verbose=verbose
    )
    
    # Save the trained model
    agent.save_model(save_path)
    print(f"\nModel saved to {save_path}")
    
    return agent

def evaluate_agent(agent, env_name="SimpleDriving-v0", episodes=5, render=True):
    """
    Evaluate the trained agent.
    
    Parameters:
    -----------
    agent : TRPOAgent
        Trained agent
    env_name : str
        Name of the environment
    episodes : int
        Number of episodes to evaluate
    render : bool
        Whether to render the environment
    """
    print(f"\nEvaluating agent for {episodes} episodes...")
    
    # Create the environment with discrete actions
    env = gym.make(env_name, isDiscrete=True)
    
    total_rewards = []
    total_steps = []
    successful_episodes = 0
    
    for episode in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"\nEpisode {episode+1}:")
        print("Starting position:", obs)
        
        while not done:
            action = agent(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if render:
                env.render()
                time.sleep(0.01)
        
        total_rewards.append(total_reward)
        total_steps.append(steps)
        
        # Check if the episode was successful (reached the goal)
        if done and reward > 0:  # Assuming positive reward indicates success
            successful_episodes += 1
            print(f"Episode {episode+1}: SUCCESS! Reached the goal in {steps} steps with reward {total_reward:.2f}")
        else:
            print(f"Episode {episode+1}: FAILED! Ended after {steps} steps with reward {total_reward:.2f}")
    
    # Print summary statistics
    avg_reward = sum(total_rewards) / len(total_rewards)
    avg_steps = sum(total_steps) / len(total_steps)
    success_rate = successful_episodes / episodes * 100
    
    print("\nEvaluation Summary:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Success Rate: {success_rate:.2f}%")
    
    return avg_reward, avg_steps, success_rate

def main():
    # Train the agent
    agent = train_agent(
        env_name="SimpleDriving-v0",
        seed=42,
        batch_size=5000,
        iterations=50,
        max_episode_length=250,
        verbose=True,
        save_path="agent.pth"
    )
    
    # Evaluate the agent
    avg_reward, avg_steps, success_rate = evaluate_agent(agent, episodes=5, render=True)
    
    # Determine if the training was successful
    if success_rate >= 60:  # 60% success rate is considered good
        print("\nCONCLUSION: The agent has successfully learned to navigate to the goal marker!")
    else:
        print("\nCONCLUSION: The agent's performance is not yet satisfactory. Consider training for more iterations or adjusting hyperparameters.")

if __name__ == "__main__":
    main()
