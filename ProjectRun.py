from SIR_Model import main

if __name__ == "__main__":
  sim = main(N = 10000,L = 100,dt = 0.01
              ,beta = 2.0
              ,gamma=0.05
              ,time_step = 100 # total time
              ,speed=2
              ,diff_coff = 0.05
              ,r_c = 1.5)
  anime=sim.Run()