# IsingTransition

We implement a classical Monte Carlo simulation of the square lattice Ising model, which we use to generate spin configurations at various temperatures. We then present several notebooks which use a variety of machine learning techniques, including binary image classification with a convolutional neural network, where our aim is to determine if a given configuration belongs to the ordered $(T &lt; T_c)$ or disordered $(T &gt; T_c)$ phases. Here $T_c$ is the (exactly known) temperature below which a long-ranged magnetic ordering emerges in the square lattice Ising model.

Below is a description of the notebooks in this repository:

1. <b>Ising-MoteCarlo.ipynb</b>: An implementation of a classical Monte Carlo simulation for the square lattice Ising model. We save snapshots of configurations which we will then use to train ML algorithms.

2. <b>Ising-ML-scikit-learn.ipynb</b>: We employ a variety of algorithms for binary classification of our snapshots including K-Nearest Neighbors, Random Forest, and Support Vector Classifier (from the scikit-learn library) and XGBoost Classifier. 

3. <b>Ising-PyTorch-NN.ipynb</b>: We train a neural network (NN) in PyTorch to perform binary classification, and explore tuning the values of hyperparameters such as weight decay (i.e. $L2$ penalty in Adam optimizer), dropout probability, and the number of units in our hidden layers.

4. <b>Ising-PyTorch-CNN.ipynb</b>: We train a convolutional neural network (CNN) in PyTorch to classify our $12 \times 12$ snapshots of lattice configurations. First we will train an initial baseline CNN before tuning hyperparameters such as weight decay, dropout probability, the number of convolutional filters used, and number of units in our hidden layer. 

We shall see that our CNN performs best out of all the methods we have explored in this project, achieving an accuracy of $97.4\%$ on our test set.