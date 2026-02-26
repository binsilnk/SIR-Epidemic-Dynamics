import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches


class main():
  def __init__(self
               ,N = 10 # population size
               ,L = 100
               ,dt = 0.5
               ,beta = 5.0
               ,gamma=0.05
               ,time_step = 1000 # total time
               ,speed=5
               ,diff_coff = 0.5
               ,r_c = 1.5):

    self.N = N
    self.L = L
    self.dt = dt
    self.beta = beta
    self.gamma = gamma
    self.time_step = time_step
    self.speed = speed
    self.diff_coff = diff_coff
    self.step_size = np.sqrt(4*self.diff_coff*self.dt)*self.speed
    self.r_c = r_c
    self.R0 = self.beta/self.gamma

    self.position = None
    self.states = None
    self.S_list = []
    self.I_list = []
    self.R_list = []
    self.T_list = []
    
  def initial_position(self):
    self.position = np.random.uniform(0,self.L,size=(self.N,2))
    self.states = np.zeros(self.N,dtype=int)
    initial_infected =np.random.choice(self.N,2,replace=False)
    self.states[initial_infected] = 1

    current_time = 0.0
    self.S_list = [self.N - len(initial_infected)]
    self.I_list = [len(initial_infected)]
    self.R_list = [0]
    self.T_list = [current_time]

    return self.position,self.states,self.S_list,self.I_list,self.R_list,self.T_list

  def setup(self):
    self.fig , (ax_map,ax_curve) = plt.subplots(1,2,figsize = (13,6) , gridspec_kw = {"width_ratios":[1.3,1]})
    self.color_map = np.array(['tab:blue','tab:red','tab:green'])
    # left graph
    self.scatter = ax_map.scatter(self.position[:,0],self.position[:,1],c = self.color_map[self.states],s=15,alpha=0.8)
    ax_map.set_xlim(0,self.L)
    ax_map.set_ylim(0,self.L)
    ax_map.set_title("Spatial spread of SIR epidemic\n(using 2D Random walk)",size = 10)
    ax_map.legend(handles=[
        patches.Patch(color='tab:blue', label='Susceptible'),
        patches.Patch(color='tab:red',  label='Infected'),
        patches.Patch(color='tab:green', label='Recovered'),
        patches.Patch(color='None', label=f'Contact radius = {self.r_c}', linewidth=2)], loc='upper right')
    # Right graph
    self.line_S, = ax_curve.plot([], [], 'tab:blue', lw=2, label='Susceptible')
    self.line_I, = ax_curve.plot([], [], 'tab:red',   lw=2, label='Infected')
    self.line_R, = ax_curve.plot([], [], 'tab:green', lw=2, label='Recovered')

    ax_curve.set_xlim(0,self.time_step*self.dt)
    ax_curve.set_ylim(0,self.N*2)
    ax_curve.set_title(f"Epidemic curves($R_0$={self.R0: .1f})",size = 10)
    ax_curve.set_xlabel("Time")
    ax_curve.set_ylabel("No. of individuals")
    ax_curve.legend()
    ax_curve.grid(True,alpha=0.3)

    self.time_text = ax_map.text(0.02, 0.95, '', transform=ax_map.transAxes,fontsize=12,bbox=dict(boxstyle="round", facecolor="wheat"))

  def update(self,frame):
    pos = self.position
    state = self.states
    current_time = (frame+1)*self.dt

    angle = np.random.uniform(0,2*np.pi,self.N)
    disp = self.step_size*np.column_stack((np.cos(angle),np.sin(angle)))
    pos += disp

    pos[:,0]= np.where(pos[:,0]<0, -pos[:,0],pos[:,0])
    pos[:,0]= np.where(pos[:,0]>self.L, 2*self.L-pos[:,0],pos[:,0])
    pos[:,1]= np.where(pos[:,1]<0, -pos[:,1],pos[:,1])
    pos[:,1]= np.where(pos[:,1]>self.L, 2*self.L-pos[:,1],pos[:,1])

    I_idx=np.where(state==1)[0]

    for i in I_idx:

      dpos = pos - pos[i]
      dist = np.sqrt(dpos[:,0]**2 +dpos[:,1]**2 )   # distance formula

      close_S = (state==0) & (dist < self.r_c)
      n_close = np.count_nonzero(close_S)

      if n_close > 0:
        # probability of no infection is np.exp(-beta*dt)

        prob = 1 - np.exp(-self.beta*self.dt) # at least one infection

        n_inf = np.random.binomial(n_close,prob)

        if n_inf > 0 :
          new_inf = np.random.choice(np.where(close_S)[0],n_inf,replace=False)
          state[new_inf] = 1

        # recover
      if np.random.random() < self.gamma*self.dt:
        state[i] = 2
    # update plot_values
    self.scatter.set_offsets(pos)
    self.scatter.set_color(self.color_map[state])

    self.S_list.append(np.sum(state==0))
    self.I_list.append(np.sum(state==1))
    self.R_list.append(np.sum(state==2))

    self.T_list.append(current_time)

    # set data
    self.line_S.set_data(self.T_list,self.S_list)
    self.line_I.set_data(self.T_list,self.I_list)
    self.line_R.set_data(self.T_list,self.R_list)
    max_time = max(float(current_time), 0) + 1
    self.fig.axes[1].set_xlim(0,max_time)
    self.fig.axes[1].set_ylim(0,self.N*1.04)
    

    self.time_text.set_text(f'Time:{current_time:.1f}')

    return (self.scatter,self.line_S,self.line_I,self.line_R,self.time_text)
  def Run(self):
    self.initial_position()
    self.setup()

    n_frame = int(self.time_step/self.dt)

    anime = FuncAnimation(self.fig,self.update,frames=n_frame,interval=40,blit=False,repeat=False)
    plt.tight_layout()
    plt.show()
    return anime





  
 