# RUN: env SUPPORTLIB=%mlir_c_runner_utils %PYTHON %s | FileCheck %s

import numpy as np
import os
import sys
import tempfile

_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_SCRIPT_PATH)

from tools import mlir_pytaco_api as pt
from tools import testing_utils as utils

###### This PyTACO part is taken from the TACO open-source project. ######
# See http://tensor-compiler.org/docs/scientific_computing/index.html.

compressed = pt.compressed
dense = pt.dense

# Define formats for storing the sparse matrix and dense vectors.
csr = pt.format([dense, compressed])
dv = pt.format([dense])

# Load a sparse matrix stored in the matrix market format) and store it
# as a CSR matrix.  The matrix in this test is a reduced version of the data
# downloaded from here:
# https://www.cise.ufl.edu/research/sparse/MM/Boeing/pwtk.tar.gz
# In order to run the program using the matrix above, you can download the
# matrix and replace this path to the actual path to the file.
A = pt.read(os.path.join(_SCRIPT_PATH, "data/pwtk.mtx"), csr)

# These two lines have been modified from the original program to use static
# data to support result comparison.
x = pt.from_array(np.full((A.shape[1],), 1, dtype=np.float32))
z = pt.from_array(np.full((A.shape[0],), 2, dtype=np.float32))

# Declare the result to be a dense vector
y = pt.tensor([A.shape[0]], dv)

# Declare index vars
i, j = pt.get_index_vars(2)

# Define the SpMV computation
y[i] = A[i, j] * x[j] + z[i]

##########################################################################

# Perform the SpMV computation and write the result to file
with tempfile.TemporaryDirectory() as test_dir:
  golden_file = os.path.join(_SCRIPT_PATH, "data/gold_y.tns")
  out_file = os.path.join(test_dir, "y.tns")
  pt.write(out_file, y)
  #
  # CHECK: Compare result True
  #
  print(f"Compare result {utils.compare_sparse_tns(golden_file, out_file)}")
