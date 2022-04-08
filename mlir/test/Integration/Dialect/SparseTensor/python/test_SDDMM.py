# RUN: SUPPORT_LIB=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
# RUN:   %PYTHON %s | FileCheck %s

import ctypes
import numpy as np
import os
import sys

from mlir import ir
from mlir import runtime as rt

from mlir.dialects import sparse_tensor as st
from mlir.dialects import builtin
from mlir.dialects import func
from mlir.dialects.linalg.opdsl import lang as dsl

_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_SCRIPT_PATH)
from tools import sparse_compiler

@dsl.linalg_structured_op
def sddmm_dsl(
    A=dsl.TensorDef(dsl.T, dsl.S.M, dsl.S.K),
    B=dsl.TensorDef(dsl.T, dsl.S.K, dsl.S.N),
    S=dsl.TensorDef(dsl.T, dsl.S.M, dsl.S.N),
    C=dsl.TensorDef(dsl.T, dsl.S.M, dsl.S.N, output=True)):
  C[dsl.D.m,
    dsl.D.n] += S[dsl.D.m, dsl.D.n] * A[dsl.D.m, dsl.D.k] * B[dsl.D.k, dsl.D.n]


def build_SDDMM(attr: st.EncodingAttr):
  """Build SDDMM kernel.

  This method generates a linalg op with for matrix multiplication using
  just the Python API. Effectively, a generic linalg op is constructed
  that computes C(i,j) += S(i,j) SUM_k A(i,k) B(k,j) for sparse S.
  """
  module = ir.Module.create()
  f64 = ir.F64Type.get()
  a = ir.RankedTensorType.get([8, 8], f64)
  b = ir.RankedTensorType.get([8, 8], f64)
  c = ir.RankedTensorType.get([8, 8], f64)
  s = ir.RankedTensorType.get([8, 8], f64, attr)
  arguments = [a, b, s, c]
  with ir.InsertionPoint(module.body):

    @func.FuncOp.from_py_func(*arguments)
    def sddmm(*args):
      return sddmm_dsl(args[0], args[1], args[2], outs=[args[3]])

  return module


def boilerplate(attr: st.EncodingAttr):
  """Returns boilerplate code for main driver."""
  return f"""
func @main(%a: tensor<8x8xf64>,
           %b: tensor<8x8xf64>,
           %c: tensor<8x8xf64>) -> tensor<8x8xf64> attributes {{ llvm.emit_c_interface }} {{
  %t = arith.constant sparse<[[0,0], [0,2], [4,1]], [1.0, 2.0, 3.0]> : tensor<8x8xf64>
  %s = sparse_tensor.convert %t : tensor<8x8xf64> to tensor<8x8xf64, {attr}>
  %0 = call @sddmm(%a, %b, %s, %c) : (tensor<8x8xf64>,
                                      tensor<8x8xf64>,
                                      tensor<8x8xf64, {attr}>,
                                      tensor<8x8xf64>) -> tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}}
"""


def build_compile_and_run_SDDMMM(attr: st.EncodingAttr, compiler):
  # Build.
  module = build_SDDMM(attr)
  func = str(module.operation.regions[0].blocks[0].operations[0].operation)
  module = ir.Module.parse(func + boilerplate(attr))

  # Compile.
  engine = compiler.compile_and_jit(module)

  # Set up numpy input and buffer for output.
  a = np.array([[1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1],
                [1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2],
                [1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3],
                [1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4],
                [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
                [1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6],
                [1.7, 2.7, 3.7, 4.7, 5.7, 6.7, 7.7, 8.7],
                [1.8, 2.8, 3.8, 4.8, 5.8, 6.8, 7.8, 8.8]], np.float64)
  b = np.ones((8, 8), np.float64)
  c = np.zeros((8, 8), np.float64)

  mem_a = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(a)))
  mem_b = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(b)))
  mem_c = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(c)))

  # Allocate a MemRefDescriptor to receive the output tensor.
  # The buffer itself is allocated inside the MLIR code generation.
  ref_out = rt.make_nd_memref_descriptor(2, ctypes.c_double)()
  mem_out = ctypes.pointer(ctypes.pointer(ref_out))

  # Invoke the kernel and get numpy output.
  # Built-in bufferization uses in-out buffers.
  # TODO: replace with inplace comprehensive bufferization.
  engine.invoke('main', mem_out, mem_a, mem_b, mem_c)

  # Sanity check on computed result. Only a few elements
  # are sampled from the full dense matrix multiplication.
  full_matmul = np.matmul(a, b)
  expected = np.zeros((8, 8), np.float64)
  expected[0, 0] = 1.0 * full_matmul[0, 0]
  expected[0, 2] = 2.0 * full_matmul[0, 2]
  expected[4, 1] = 3.0 * full_matmul[4, 1]
  c = rt.ranked_memref_to_numpy(mem_out[0])
  if np.allclose(c, expected):
    pass
  else:
    quit(f'FAILURE')


def main():
  support_lib = os.getenv('SUPPORT_LIB')
  assert support_lib is not None, 'SUPPORT_LIB is undefined'
  if not os.path.exists(support_lib):
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                            support_lib)

  # CHECK-LABEL: TEST: testSDDMMM
  print('\nTEST: testSDDMMM')
  with ir.Context() as ctx, ir.Location.unknown():
    count = 0
    # Loop over various ways to compile and annotate the SDDMM kernel with
    # a *single* sparse tensor. Note that we deliberate do not exhaustively
    # search the full state space to reduce runtime of the test. It is
    # straightforward to adapt the code below to explore more combinations.
    levels = [[st.DimLevelType.dense, st.DimLevelType.dense],
              [st.DimLevelType.dense, st.DimLevelType.compressed],
              [st.DimLevelType.compressed, st.DimLevelType.dense],
              [st.DimLevelType.compressed, st.DimLevelType.compressed]]
    orderings = [
        ir.AffineMap.get_permutation([0, 1]),
        ir.AffineMap.get_permutation([1, 0])
    ]
    for level in levels:
      for ordering in orderings:
        for pwidth in [32]:
          for iwidth in [32]:
            for par in [0]:
              for vec in [0, 1]:
                for e in [True]:
                  vl = 1 if vec == 0 else 16
                  attr = st.EncodingAttr.get(level, ordering, pwidth, iwidth)
                  opt = (f'parallelization-strategy={par} '
                         f'vectorization-strategy={vec} '
                         f'vl={vl} enable-simd-index32={e}')
                  compiler = sparse_compiler.SparseCompiler(
                      options=opt, opt_level=0, shared_libs=[support_lib])
                  build_compile_and_run_SDDMMM(attr, compiler)
                  count = count + 1
  # CHECK: Passed 16 tests
  print('Passed ', count, 'tests')


if __name__ == '__main__':
  main()
