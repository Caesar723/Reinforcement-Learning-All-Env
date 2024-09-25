import gymnasium as gym



# 创建环境，使用新 API
env = gym.make('Humanoid-v4', render_mode='human')

# 重置环境并设置种子
state = env.reset()

# 渲染环境
observation, info = env.reset(seed=42)

while True:
    env.render()  # 确保调用渲染
    action = env.action_space.sample()  # 随机选择一个动作
    
    state, reward, done, info,_ = env.step(action)
    
    # if done:
    #     break
    
env.close()