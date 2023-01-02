# Compiling
g++ -Wall -O3 -fopenmp -I/usr/include/eigen3 integrate_FBZ_sbc.cpp -o spinberry

# Parameters 
lam=0.3        # Rashba term
ex=0.4         # Exchange term
minFermi=-4    # smallest fermi energy
NFermi=100     # number of fermi energies
NPontos=1024    # square root of the number of k-points to be used in the k-space integration
delta=0.000001     # discretization to use in the numerical derivative

./spinberry $lam $ex $minFermi $NFermi $NPontos $delta > data.dat

# Output format -- fermi is only real but rest is real followed by imag
# fermi total aroundK aroundKp rest conv1 conv2 conv3
# conv1 - less K
# conv2 - less E
# conv3 - less K and E

