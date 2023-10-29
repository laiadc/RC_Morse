# Machine learning calculation of quantum dynamics and vibrational eigenstates by reservoir computing

Using Reservoir computing to study the the coupled Morse Hamiltonian, which models the vibrational stretching of the H$_2$O molecule and isotopic derivatives.

The results are illustrated in a set of [Jupyter](https://jupyter.org/) notebooks. The whole code is the result of the work in <a href = "https://arxiv.org/abs/" target="_blank"> this paper</a>. Any contribution or idea to continue the lines of the proposed work will be very welcome.

**Remark**: We recommend the readers to view the notebooks locally and in *Trusted* mode for nicer presentation and correct visualization of the figures. 

In this work, we use Reservoir Computing to efficiently calculate the vibrational eigenstates of a Hamiltonian within a specific energy range. The proposed methodology uses reservoir computing, a highly relevant and efficient machine learning algorithm that predicts the quantum dynamics by propagating an initial wavepacket in time, followed by Fourier transform. As an illustration, we apply our method to a coupled Morse Hamiltonian, which models the vibrational stretching of the H$_2$O molecule and isotopic derivatives. The results indicate that the proposed method is promising for efficiently calculating the eigenstates 
of complex quantum systems, particularly at high energies.

<p align="center"><img src="https://github.com/laiadc/RC_Morse/blob/main/wavefunctions3.png"  align=middle width=600pt />
</p>
<p align="center">
<em>Example of the eigenfunctions functions obtained in this work, for different energies. </em>
</p>


## Notebooks

All the notebooks used for this work can be found inside the folder **notebooks** .

**Remark**: Some of the training data could not be uploaded because it exceeded the maximum size allowed by GitHub. The notebooks provide the code to obtain such training data. 

### [ReservoirComputing_CoupledMorse.ipynb](https://github.com/laiadc/RC_Morse/blob/main/ReservoirComputing_CoupledMorse.ipynb)
Application of the adapted Reservoir Computing model to obtain the eigenenergies and eigenfunctions of the coupled Morse Hamiltonian.

### [Variational method.ipynb](https://github.com/laiadc/RC_Morse/blob/main/Variational%20method.ipynb)
Using the variational method to obtain the Morse eigenenergies and eigenfunctions.

### [Figures-Morse.ipynb](https://github.com/laiadc/RC_Morse/blob/main/Figures-Morse.ipynb)
This notebook summarizes the results of the paper and provides the figures provided in the paper.

### BibTex reference format for citation for the Code
```
@misc{Domingo2023,
title={Machine learning calculation of quantum dynamics and vibrational
eigenstates by reservoir computing},
url={https://github.com/laiadc/RC_Morse/},
note={Using Reservoir computing to compute the eigenstates of complex Hamiltonians.},
author={L. Domingo and J. Rifà},
  year={2023}
}


