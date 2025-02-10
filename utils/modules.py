import numpy as np
import scipy as sp
import pandas as pd
import os

class net:
    def __init__(self, seed = 219, dt = 0.01, Δt = 0.1, N = 1000, g = 1.5, τ = 1, p = 1, μ = 0.5, output_num = 1, print_progress = False):
        
        self.dt = dt # integration time step 
        self.Δt = Δt # training update interval

        self.N = N
        self.g = g
        self.τ = τ

        self.output_num = output_num # allow for multiple outputs

        self.print_progress = print_progress

        self.p = p # this is not mentioned in the paper, connectivity matrix J property (see below)
        self.μ = μ # this is not mentioned in the paper, connectivity matrix J property (see below)

        #############################################
        ### generate random connectivity matrix J ###
        #############################################

        # p -> sparsity parameter, fraction of non-zero entries
        # μ -> asymmetric parameter that determines the eigen spectrum
        # μ = 0: all imaginary eigenvalues
        # μ = 0.5: circular distribution 
        # μ = 1: all real eigenvalues

        np.random.seed(seed)
        std = 1 / np.sqrt(self.N * self.p)
        θ = np.pi * self.μ
        self.J = std * np.random.randn(self.N, self.N)
        self.J = np.cos(θ/2) * (self.J + self.J.T) / np.sqrt(2)  + np.sin(θ/2) * (self.J - self.J.T) / np.sqrt(2)
        if self.p != 1:
            self.J *= (sp.sparse.random(N, N, density = p).toarray() != 0).astype(int)

        #########################################
        ### generate random feedback vector K ###
        #########################################

        # K's shape is (N, output_num)
        self.K = 2 * np.random.rand(self.N, self.output_num) - 1 # random number between -1 and 1

        #################################
        ### set initial network state ###
        #################################

        self.reset_states(seed = int(np.random.rand() * 100)) # this seed is in general different from the given, but will be correlated

    def φ(self, x):

        return np.tanh(x)

    def reset_states(self, seed = 42, use_random = True, h_set = None, z_set = None, w_set = None):

        if use_random:
            self.h0 = np.random.randn(self.N)
            self.z0 = np.zeros((self.output_num))
            self.w0 = np.zeros((self.N, self.output_num))
        else:
            self.h0 = h_set
            self.z0 = z_set
            self.w0 = w_set

    # basic dynamic equation
    def dhdt(self, h, r, I):

        return - h / self.τ + self.g * self.J.dot(r) + I

    def simulate_free(self, t_max, t_ini = 0, keep_last_state = True):

        # simulate the network dynamics without inputs

        tN = int((t_max - t_ini) / self.dt)

        T = np.zeros(tN)
        h = np.zeros((tN, self.N))
        r = np.zeros((tN, self.N))

        T[0] = t_ini
        h[0] = self.h0
        r[0] = self.φ(h[0])

        for i in range(tN - 1):

            if (i % (tN//10) == 0) & self.print_progress:
                print(f'Free simulation progress: {i/tN*100:.2f}%')

            T[i + 1] = T[i] + self.dt
            h[i + 1] = h[i] + self.dt * self.dhdt(h[i], r[i], 0)
            r[i + 1] = self.φ(h[i + 1])

        if keep_last_state:
            self.h0 = h[-1]

        return [T, h, r]

    def simulate_train(self, t_max, target, α = 1, mode = 'FORCE', t_ini = 0, γ = 0.0, keep_last_state = True):

        tN = int((t_max - t_ini) / self.dt)

        T = np.zeros(tN)
        h = np.zeros((tN, self.N))
        r = np.zeros((tN, self.N))

        T[0] = t_ini
        h[0] = self.h0
        r[0] = self.φ(h[0])

        # additoinal objects for training
        # w : readout layer
        # z : output
        # e : error

        w = np.zeros((tN, self.N, self.output_num))
        z = np.zeros((tN, self.output_num))
        e = np.zeros((tN, self.output_num))

        w[0] = self.w0
        z[0] = self.z0
        e[0] = 0

        P = np.eye(self.N) * α # running estimate of inverse correlation

        for i in range(tN - 1):

            if (i % (tN//10) == 0) & self.print_progress:
                print(f'Training progress: {i/tN*100:.2f}%')

            T[i + 1] = T[i] + self.dt
            target_val = target(T[i + 1])
            
            if mode == 'FORCE':
                I = self.K.dot(z[i])
            elif mode == 'echo':
                I = target_val

            h[i + 1] = h[i] + self.dt * self.dhdt(h[i], r[i], I)
            r[i + 1] = self.φ(h[i + 1])
            z[i + 1] = w[i].T.dot(r[i + 1]).flatten()
            e[i + 1] = z[i + 1] - target_val

            if i % (self.Δt // self.dt) == 0:
                P *= 1 / (1 - γ)
                Pr = P.dot(r[i + 1])
                P =  P - np.outer(Pr, r[i + 1]).dot(P)/(1 + r[i + 1].dot(Pr))
                Δw = np.outer(P.dot(r[i + 1]), e[i + 1])
                w[i + 1] = w[i] - Δw
            else:
                w[i + 1] = w[i]

        if keep_last_state:
            self.h0 = h[-1]
            self.z0 = z[-1]
            self.w0 = w[-1]
            self.e0 = e[-1]

        return [T, h, r, w, z, e, P]

    def simulate_test(self, t_max, target, h0 = None, w0 = None, z0 = None, e0 = None, t_ini = 0, use_last_state = True, keep_last_state = True):

        tN = int((t_max - t_ini) / self.dt)

        T = np.zeros(tN)
        h = np.zeros((tN, self.N))
        r = np.zeros((tN, self.N))

        T[0] = t_ini

        z = np.zeros((tN, self.output_num))
        e = np.zeros((tN, self.output_num))

        if use_last_state:
            h[0] = self.h0
            w = self.w0
            z[0] = self.z0
            e[0] = self.e0
        else:
            h[0] = h0
            w = w0
            z[0] = z0
            e[0] = e0

        r[0] = self.φ(h[0])

        for i in range(tN - 1):

            if (i % (tN//10) == 0) & self.print_progress:
                print(f'Testing progress: {i/tN*100:.2f}%')

            T[i + 1] = T[i] + self.dt

            I = self.K.dot(z[i])

            h[i + 1] = h[i] + self.dt * self.dhdt(h[i], r[i], I)
            r[i + 1] = self.φ(h[i + 1])
            z[i + 1] = w.T.dot(r[i + 1]).flatten()
            e[i + 1] = z[i + 1] - target(T[i + 1])

        if keep_last_state:
            self.h0 = h[-1]
            self.z0 = z[-1]
            self.w0 = w[-1]
            self.e0 = e[-1]

        return [T, h, r, w, z, e]
        
# utility function for sinusoidal parameter sweep
def test_single(inputs):
    
    # inputs: 
    # log10a, log10p, log10g, γ, seed, count, path, mode

    amp = 10 ** inputs[0]
    period = 10 ** inputs[1]
    g = 10 ** inputs[2]
    γ = inputs[3]
    seed = inputs[4]
    count = inputs[5]
    path = inputs[6]
    mode = inputs[7]
    os.makedirs(path, exist_ok = True)
    save_name = f'{path}/{count:05d}.csv'
    if os.path.isfile(save_name):
        pass
    else:
        def target(_t):
            return amp * np.sin(2 * np.pi * _t / period)
    
        network = net(seed = seed, g = g)
        RES_TR = network.simulate_train(400, target = target, mode = mode, γ = γ)
        RES_TE = network.simulate_test(800, target = target, t_ini = RES_TR[0][-1])
    
        # downsample - otherwise there is too much data
        TT = np.array([*RES_TR[0], *RES_TE[0]])[::10]
        ZZ = np.array([*np.reshape(RES_TR[4], -1), *np.reshape(RES_TE[4], -1)])[::10]
        EE = np.array([*np.reshape(RES_TR[5], -1), *np.reshape(RES_TE[5], -1)])[::10]
    
        df = pd.DataFrame({'T' : TT, 'Z' : ZZ, 'E' : EE})
        df.to_csv(save_name, index = False)