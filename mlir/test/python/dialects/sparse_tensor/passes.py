# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.passmanager import *

from mlir.dialects import sparse_tensor as st


def run(f):
  print('\nTEST:', f.__name__)
  f()
  return f


# CHECK-LABEL: TEST: testSparseTensorPass
@run
def testSparseTensorPass():
  with Context() as context:
    PassManager.parse('any(sparsification)')
    PassManager.parse('any(sparse-tensor-conversion)')
  # CHECK: SUCCESS
  print('SUCCESS')
