<!-- ABOUT THE PROJECT -->
## About The Project

This repository is part of the bachelor thesis "Tiefe Neuronale Netze Mit Gewöhnlichen Differentialgleichungen" by Jonas Arruda.
Thus, this project provides an implementation of stable neuronal networks motivated by ordinary differential equations.

ODE-CNNs.ipynb: Implementation of the networks together with training and testing routines.

stability-regions.ipynb: Depiction of the stability regions of the ODE-solvers used as basis for the neural networks.

<!-- Packages -->
## Packages

The following packages are need to run the project:
* numpy
* matplotlib
* cv2
* tensorflow (2.0 or higher)
* tensorflow_datasets
* [Python utilities for reading the STL-10 dataset](https://github.com/mttk/STL10)

The following datasets are used:
* [CIFAR 10/100 (comes with tensorflow.keras.datasets)](https://www.cs.toronto.edu/~kriz/cifar.html)
* [STL 10](https://cs.stanford.edu/~acoates/stl10/)
* [Oxford-IIIT Pet Dataset (comes with tensorflow_datasets](https://www.robots.ox.ac.uk/~vgg/data/pets/)
* [Berkeley DeepDrive](https://bdd-data.berkeley.edu)

<!-- Usage -->
## Usage

First you have to specify the follwing parameters:
* cycle = 1 # experiment number
* num_classes = 20 # number of classes: 10=CIFAR10, 100=CIFAR100, '10b'=STL10, 3=Oxford IIIt Pet, 20=bdd100k
* train=True # boolean: whether you want to train all the different networks for a given dataset. If you want to train only a specific network, you have to change the run_train_test function.
* test_noise=False # boolean: whether you want to test your networks against noise. This function will load networks from the folder within the current cycle.
* test_adversarial_attack=False # boolean: whether you want to test your networks against an adversarial attack
* test_blur=False # boolean: whether you want to test your networks against blurring the images

<!-- CONTACT -->
## Contact

Jonas Arruda - jonas.arruda@uni-bonn.de

Project Link: [Ode-Nets](https://github.com/arrjon/ode-nets)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* [TensorFlow](https://www.tensorflow.org/tutorials/images/segmentation)
* [Deep Neural Networks Motivated by Partial Differential Equation by Ruthotto and Haber (2018)](https://arxiv.org/abs/1804.04272)
* [IMEXnet - A Forward Stable Deep Neuronal Network by Haber et al. (2019)](https://arxiv.org/pdf/1903.02639.pdf)
* [Partial Differential Equations For Optimizing Deep Neural Networks by Chaudhari et al. (2017)](https://arxiv.org/abs/1704.04932)
* [Python utilities for reading the STL-10 dataset by Tutek et al.](https://github.com/mttk/STL10)


