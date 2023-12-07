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
# See http://tensor-compiler.org/docs/data_analytics/index.html.

compressed = pt.compressed
dense = pt.dense

# Define formats for storing the sparse tensor and dense matrices.
csf = pt.format([compressed, compressed, compressed])
rm = pt.format([dense, dense])

# Load a sparse three-dimensional tensor from file (stored in the FROSTT
# format) and store it as a compressed sparse fiber tensor. We use a small
# tensor for the purpose of testing. To run the program using the data from
# the real application, please download the data from:
# http://frostt.io/tensors/nell-2/
B = pt.read(os.path.join(_SCRIPT_PATH, "data/nell-2.tns"), csf)

# These two lines have been modified from the original program to use static
# data to support result comparison.
C = pt.from_array(np.full((B.shape[1], 25), 1, dtype=np.float32))
D = pt.from_array(np.full((B.shape[2], 25), 2, dtype=np.float32))

# Declare the result to be a dense matrix.
A = pt.tensor([B.shape[0], 25], rm)

# Declare index vars.
i, j, k, l = pt.get_index_vars(4)

# Define the MTTKRP computation.
A[i, j] = B[i, k, l] * D[l, j] * C[k, j]

##########################################################################

# Perform the MTTKRP computation and write the result to file.
with tempfile.TemporaryDirectory() as test_dir:
    golden_file = os.path.join(_SCRIPT_PATH, "data/gold_A.tns")
    out_file = os.path.join(test_dir, "A.tns")
    pt.write(out_file, A)
    #
    # CHECK: Compare result True
    #
    print(f"Compare result {utils.compare_sparse_tns(golden_file, out_file)}")
