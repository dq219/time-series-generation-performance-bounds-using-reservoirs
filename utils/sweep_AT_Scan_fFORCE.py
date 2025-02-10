# Script for submitting jobs to HPC

from modules import test_single
from multiprocessing import Pool
from multiprocessing import cpu_count
import numpy as np
import argparse


log10g = np.log10(1.5)
γ = 0.002 # for forgetful Force

parser = argparse.ArgumentParser(description="Process inputs.")
parser.add_argument("-seed0", type=int, default= 0, help="Model choice")
parser.add_argument("-seed1", type=int, default= 1, help="Model choice")

args = parser.parse_args()

path = f'AT_Scan/fFORCE/RUN_{args.seed0}_{args.seed1}'

if __name__ == '__main__':
    
    pool = Pool()
    INPUTS = []
    count = 0
    for log10a in np.linspace(-1, np.log10(20), 24):
        for log10p in np.linspace(0, np.log10(200), 24):
            for seed in np.arange(args.seed0, args.seed1):
                INPUTS.append([log10a, log10p, log10g, γ, seed, count, path, 'FORCE'])
                count += 1

    pool.map(test_single, INPUTS, chunksize = 1)
    pool.terminate()
