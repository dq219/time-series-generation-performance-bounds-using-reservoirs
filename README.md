# Fundamental performance bounds on time-series generation using reservoir computing

This code repository contains minimal implementations of FORCE, ESN, and forgetful FORCE algorithms for learning time-series generation using reservoir computing. Three basic notebooks are provided:

- [run_demo.ipynb](./run_demo.ipynb) is a notebook that illustrates the basic use of the [network class](./utils/modules.py), including the three training algorithms.
- [run_sweep_Dt.ipynb](./run_sweep_Dt.ipynb) performs parameter sweep over the training update interval to produce the Nyquist sampling results.
- [run_sweep_chaos.ipynb](./run_sweep_chaos.ipynb) takes 135 [curated chaotic trajectories](https://github.com/williamgilpin/dysts) and trains networks to learn them.
- [run_sweep_NnEI.ipynb](./run_sweep_NnEI.ipynb) compares the performance of three matrices with distinct eigenspectra, including two from the Ginibre ensemble with different number of neurones, and one with distinct cell types that modify the spectra density.

The [utils](./utils) folder also contains scripts used to submit jobs to the HPC to perform the amplitude-period parameter sweep.

Please cite the [publication](https://arxiv.org/abs/2410.20393) if you find the code helpful for your work. Get in touch with dq219@mit.edu or 000.gdz@gmail.com for any questions.

Thanks for your interest! :)

## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.