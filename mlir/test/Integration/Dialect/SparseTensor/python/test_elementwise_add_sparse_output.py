# RUN: SUPPORT_LIB=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext %PYTHON %s | FileCheck %s

import ctypes
import numpy as np
import os
import sys

import mlir.all_passes_registration

from mlir import ir
from mlir import runtime as rt
from mlir import execution_engine
from mlir import passmanager
from mlir.dialects import sparse_tensor as st
from mlir.dialects import builtin
from mlir.dialects.linalg.opdsl import lang as dsl

_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_SCRIPT_PATH)
from tools import np_to_sparse_tensor as test_tools

# TODO: Use linalg_structured_op to generate the kernel after making it to
# handle sparse tensor outputs.
_KERNEL_STR = """
#DCSR = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ]
}>

#trait_add_elt = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A
    affine_map<(i,j) -> (i,j)>,  // B
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = A(i,j) + B(i,j)"
}

func @sparse_add_elt(
    %arga: tensor<3x4xf64, #DCSR>, %argb: tensor<3x4xf64, #DCSR>) -> tensor<3x4xf64, #DCSR> {
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %argx = sparse_tensor.init [%c3, %c4] : tensor<3x4xf64, #DCSR>
  %0 = linalg.generic #trait_add_elt
    ins(%arga, %argb: tensor<3x4xf64, #DCSR>, tensor<3x4xf64, #DCSR>)
    outs(%argx: tensor<3x4xf64, #DCSR>) {
      ^bb(%a: f64, %b: f64, %x: f64):
        %1 = arith.addf %a, %b : f64
        linalg.yield %1 : f64
  } -> tensor<3x4xf64, #DCSR>
  return %0 : tensor<3x4xf64, #DCSR>
}

func @main(%ad: tensor<3x4xf64>, %bd: tensor<3x4xf64>) -> tensor<3x4xf64, #DCSR>
  attributes { llvm.emit_c_interface } {
  %a = sparse_tensor.convert %ad : tensor<3x4xf64> to tensor<3x4xf64, #DCSR>
  %b = sparse_tensor.convert %bd : tensor<3x4xf64> to tensor<3x4xf64, #DCSR>
  %0 = call @sparse_add_elt(%a, %b) : (tensor<3x4xf64, #DCSR>, tensor<3x4xf64, #DCSR>) -> tensor<3x4xf64, #DCSR>
  return %0 : tensor<3x4xf64, #DCSR>
}
"""


class _SparseCompiler:
  """Sparse compiler passes."""

  def __init__(self):
    self.pipeline = (
        f'sparsification,'
        f'sparse-tensor-conversion,'
        f'builtin.func(linalg-bufferize,convert-linalg-to-loops,convert-vector-to-scf),'
        f'convert-scf-to-std,'
        f'func-bufferize,'
        f'arith-bufferize,'
        f'builtin.func(tensor-bufferize,finalizing-bufferize),'
        f'convert-vector-to-llvm{{reassociate-fp-reductions=1 enable-index-optimizations=1}},'
        f'lower-affine,'
        f'convert-memref-to-llvm,'
        f'convert-std-to-llvm,'
        f'reconcile-unrealized-casts')

  def __call__(self, module: ir.Module):
    passmanager.PassManager.parse(self.pipeline).run(module)


def _run_test(support_lib, kernel):
  """Compiles, runs and checks results."""
  module = ir.Module.parse(kernel)
  _SparseCompiler()(module)
  engine = execution_engine.ExecutionEngine(
      module, opt_level=0, shared_libs=[support_lib])

  # Set up numpy inputs and buffer for output.
  a = np.array(
      [[1.1, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 6.6, 0.0]],
      np.float64)
  b = np.array(
      [[1.1, 0.0, 0.0, 2.8], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
      np.float64)

  mem_a = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(a)))
  mem_b = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(b)))

  # The sparse tensor output is a pointer to pointer of char.
  out = ctypes.c_char(0)
  mem_out = ctypes.pointer(ctypes.pointer(out))

  # Invoke the kernel.
  engine.invoke('main', mem_a, mem_b, mem_out)

  # Retrieve and check the result.
  rank, nse, shape, values, indices = test_tools.sparse_tensor_to_coo_tensor(
      support_lib, mem_out[0], np.float64)

  # CHECK: PASSED
  if np.allclose(values, [2.2, 2.8, 6.6]) and np.allclose(
      indices, [[0, 0], [0, 3], [2, 2]]):
    print('PASSED')
  else:
    quit('FAILURE')


def test_elementwise_add():
  # Obtain path to runtime support library.
  support_lib = os.getenv('SUPPORT_LIB')
  assert support_lib is not None, 'SUPPORT_LIB is undefined'
  assert os.path.exists(support_lib), f'{support_lib} does not exist'
  with ir.Context() as ctx, ir.Location.unknown():
    _run_test(support_lib, _KERNEL_STR)


test_elementwise_add()
