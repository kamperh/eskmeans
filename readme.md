Embedded Segmental K-Means (ES-KMeans)
======================================

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/kamperh/eskmeans/blob/master/license.md)


Overview
--------
Unsupervised acoustic word segmentation and clustering using the embedded
segmental K-means (ES-KMeans) algorithm. The algorithm is described in:

- H. Kamper, K. Livescu, and S. J. Goldwater, "An embedded segmental k-means
  model for unsupervised segmentation and clustering of speech," in *Proc.
  ASRU*, 2017. [[arXiv](https://arxiv.org/abs/1703.08135)]

Please cite this paper if you use the code.


Installation
------------
Dependencies can be installed in a conda environment:

    conda env create -f environment.yml
    conda activate eskmeans

Perform unit tests:

    nosetests -v


Recipe
------
**To-do.**



Contributors
------------
- [Herman Kamper](http://www.kamperh.com/)
- [Karen Livescu](http://ttic.uchicago.edu/~klivescu/)
- [Sharon Goldwater](http://homepages.inf.ed.ac.uk/sgwater/)
