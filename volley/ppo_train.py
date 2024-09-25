import torch
import torch.distributed
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader



def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Act_Net(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.action = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1)
        )
        
        orthogonal_init(self.action[0])
        orthogonal_init(self.action[2])
        orthogonal_init(self.action[4],gain=0.1)
        

    def forward(self,x):
        
        action=self.action(x)
        #value=self.value(x)
        dist=torch.distributions.Categorical(action)
        return dist,dist.entropy()
    
    def predict(self,x):
        return self.action(x)
    

class Val_Net(nn.Module):
    def __init__(self, state_size):
        super().__init__()
        self.value=nn.Sequential(
            nn.Linear(state_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        orthogonal_init(self.value[0])
        orthogonal_init(self.value[2])
        orthogonal_init(self.value[4])
    def forward(self,x):
        
        value=self.value(x)
        
        return value

class RunningMeanStd:
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n )

class Normalization:#Trick 2—State Normalization
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        #print(x)
        # Whether to update the mean and std,during the evaluating,update=Flase
        x=np.array(x)
        if update:  
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)
        return x
    

class RewardScaling:#Trick 4—Reward Scaling

    def __init__(self, gamma):
        self.shape = 1  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x[0]

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)


class Agent_PPO:

    def __init__(self,state_size, action_size,device="cpu") -> None:
        self.normalization=Normalization(state_size)
        self.reward_scale=RewardScaling(0.99)
        self.gamma=0.99
        self.lambd=0.95
        self.clip_para=0.2
        self.epochs=15
        self.max_step=3000000
        self.total_step=0
        self.lr=3e-4
        

        self.state=[]
        self.reward=[]
        self.action=[]
        self.next_state=[]
        self.done=[]


        
        if torch.cuda.is_available():
            self.device=torch.device(device)
        else:
            self.device=torch.device("cpu")
        self.action_size=action_size
        self.model_act=Act_Net(state_size, action_size).to(self.device)
        self.model_val=Val_Net(state_size).to(self.device)
        self.MSEloss=nn.MSELoss()
        self.opti_act=optim.Adam(self.model_act.parameters(),lr=self.lr, eps=1e-5)#Trick 9—Adam Optimizer Epsilon Parameter
        self.opti_val=optim.Adam(self.model_val.parameters(),lr=self.lr, eps=1e-5)
        self.scheduler_action = StepLR(self.opti_act, step_size=50, gamma=0.99)
        self.scheduler_value = StepLR(self.opti_val, step_size=50, gamma=0.99)


        self.init_graph()
    # def lr_decay(self, total_steps):
    #     lr = self.lr * (1 - total_steps / self.max_step)
    #     for p in self.opti.param_groups:
    #         p['lr'] = lr
    def init_graph(self):
        self.rewards = 0
        self.step=0
        self.rewards_store=[]
        self.best_mean_reward=0
        fig, self.ax = plt.subplots()
        self.line, = self.ax.plot(self.rewards, label='Total Rewards per Episode', color='b')

        # 设置图表标题和标签
        self.ax.set_title('PPO Training Rewards Over Episodes')
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Total Reward')
        self.ax.legend()
        self.ax.grid(True)

    


    def store(self,state,action,reward,next_state,done):

        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.next_state.append(next_state)
        self.done.append(done)

        self.graph_on_step(reward)

    def clean(self):
        self.state=[]
        self.reward=[]
        self.action=[]
        self.next_state=[]
        self.done=[]
        self.reward_scale.reset()

    def choose_act(self,state,mask:torch.Tensor=None):
        state=torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            result=self.model_act.predict(state)
            #print(result,mask)
            if mask!=None and mask.any():
                result[mask==False]=0
        #print(result)

        dist=torch.distributions.Categorical(result)
        #entropy = dist.entropy().mean()
        act=dist.sample().item()
        return act
    

    def normalize_adv(self,adv:torch.Tensor):
        return ((adv - adv.mean()) / (adv.std() + 1e-5))

    def advantage_cal(self,delta:torch.Tensor,done:torch.Tensor):
        advantage_list=[]
        advantage=0
        #print(delta.cpu().numpy())
        for reward,done in zip(delta.cpu().numpy()[::-1],done.cpu().numpy()[::-1]):
            if done:
                advantage=0
            advantage=reward+self.lambd*self.gamma*advantage
            # print(advantage)
            # print()
            advantage_list.insert(0,advantage)
        #print(advantage_list)
        return np.array(advantage_list)


    def train(self):
        state=torch.FloatTensor(np.array(self.state)).to(self.device)
        action=torch.LongTensor(np.array(self.action)).unsqueeze(1).to(self.device)
        done=torch.FloatTensor(np.array(self.done)).unsqueeze(1).to(self.device)
        next_state=torch.FloatTensor(np.array(self.next_state)).to(self.device)
        #print(self.reward)
        reward=torch.FloatTensor(np.array(self.reward)).unsqueeze(1).to(self.device)

        
        with torch.no_grad():
            
            a_pro,entropy=self.model_act(state)
            
            
            v=self.model_val(state)
            v_=self.model_val(next_state)
            delta=reward+self.gamma*v_*(1-done)-v
            
            advantage=self.advantage_cal(delta,done)
            
            advantage=torch.FloatTensor(advantage).detach().to(self.device)
            rewards=advantage+v

            advantage=self.normalize_adv(advantage)
            
            
           
            old_prob_log=a_pro.log_prob(action.squeeze(-1))
            

        dataset = TensorDataset(state, action, done, next_state,advantage,old_prob_log,rewards)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

        for _ in range(self.epochs):
            for batch in dataloader:
                state, action, done, next_state,advantage,old_prob_log,rewards=batch
                a_pro,entropy=self.model_act(state)#Trick 5—Policy Entropy
                v=self.model_val(state)
                
                new_prob_log=a_pro.log_prob(action.squeeze(-1))
                
                
                rate=torch.exp(new_prob_log-old_prob_log.detach()).unsqueeze(1)
                
                surr1=rate*advantage
                surr2=torch.clamp(rate,1-self.clip_para,1+self.clip_para)*advantage
                
                act_loss=-torch.min(surr1,surr2).squeeze(1)-0.01*entropy
                
                self.opti_act.zero_grad()
                

                act_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.model_act.parameters(), 0.5)
                self.opti_act.step()
                self.scheduler_action.step()

                self.opti_val.zero_grad()
                val_loss=F.mse_loss(rewards,v)
                print(val_loss)
                val_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_val.parameters(), 0.5)
                self.opti_val.step()
                self.scheduler_value.step() # Trick 6:learning rate Decay

        self.graph_on_rollout_end()
        self.clean()
    
    def graph_on_rollout_end(self) -> None:
        if self.best_mean_reward<self.rewards/self.step:
            self.best_mean_reward=self.rewards/self.step
            torch.save(self.model_act, 'model_complete_act.pth')
            torch.save(self.model_val, 'model_complete_val.pth')
        self.rewards_store.append(self.rewards)
        self.rewards = 0
        self.step=0
        self.line.set_data(range(len(self.rewards_store)), self.rewards_store)
    
        self.ax.set_xlim(0, len(self.rewards_store))
        self.ax.set_ylim(min(self.rewards_store) - 5, max(self.rewards_store) + 5)
        plt.savefig('ppo_training_reward.png')
        
    def graph_on_step(self,reward):
            self.step+=1
            self.rewards+=reward





if __name__=="__main__":
    agent=Agent_PPO(6,2,)
    print(agent.choose_act([0,0,0,0,0,0]))
    agent.store([0,0,5,0,7,0],1,0.3,[0,0,0,2,0,0],0)
    agent.store([0,0,0,0,0,0],1,0.3,[0,0,0,2,0,0],0)
    agent.store([0,0,0,0,0,0],0,0.3,[0,0,0,2,0,0],0)
    agent.store([0,0,0,0,0,0],1,0.3,[0,0,0,2,0,0],0)
    agent.store([0,0,0,0,0,0],1,0.3,[0,0,0,2,0,0],0)
    agent.store([0,0,0,0,0,0],1,0.3,[0,0,0,2,0,0],1)
    
    agent.train()