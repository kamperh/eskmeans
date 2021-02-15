Embedded Segmental K-Means (ES-KMeans)
======================================

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/kamperh/eskmeans/blob/master/license.md)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kamperh/eskmeans/blob/master/examples/eskmeans_example.ipynb)


Overview
--------
Unsupervised acoustic word segmentation and clustering using the embedded
segmental K-means (ES-KMeans) algorithm. The algorithm is described in:

> H. Kamper, K. Livescu, and S. J. Goldwater, "An embedded segmental K-means
> model for unsupervised segmentation and clustering of speech," in *Proc.
> ASRU*, 2017. [[arXiv](https://arxiv.org/abs/1703.08135)]

Please cite this paper if you use the code.


Installation
------------
Dependencies can be installed in a conda environment:

    conda env create -f environment.yml
    conda activate eskmeans

Perform unit tests:

    nosetests -v


Examples
--------
A number of example notebooks are given in `examples/`. The
[examples/eskmeans_example.ipynb](examples/eskmeans_example.ipynb) notebook
provides a step-by-step example of the ES-KMeans algorithm. This notebook can
also be opened directly in a [Colab
notebook](https://colab.research.google.com/github/kamperh/eskmeans/blob/master/examples/eskmeans_example.ipynb).


Recipe
------
The code here only provides the main algorithm and some supporting utilities. A
complete recipe where ES-KMeans is applied to the Buckeye English and NCHLT
Xitsonga datasets is available
[here](https://github.com/kamperh/bucktsong_eskmeans).


License
-------
Copyright (C) 2021 Herman Kamper

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

The module [utils/theta_oscillator.py](utils/theta_oscillator.py) is a
derivation of Adriana Stan's Python implementation available at
https://github.com/speech-utcluj/thetaOscillator-syllable-segmentation, which
was released under the same license. Concrete changes to Adriana's code are
listed in the documentation at the top of [the
module](utils/theta_oscillator.py). I also list Adriana as a contributor below.


Contributors
------------
- [Herman Kamper](http://www.kamperh.com/)
- [Adriana Stan](http://adrianastan.com/)
- [Karen Livescu](http://ttic.uchicago.edu/~klivescu/)
- [Sharon Goldwater](http://homepages.inf.ed.ac.uk/sgwater/)
