# RUN: SUPPORT_LIB=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext %PYTHON %s | FileCheck %s

import os
import ctypes
import mlir.all_passes_registration
import numpy as np

from mlir.dialects import builtin
from mlir.dialects.linalg.opdsl.lang import *
from mlir.dialects.sparse_tensor import *
from mlir.execution_engine import *
from mlir.ir import *
from mlir.passmanager import *
from mlir.runtime import *


def run(f):
  print('\nTEST:', f.__name__)
  f()
  return f


@linalg_structured_op
def matmul_dsl(
    A=TensorDef(T, S.M, S.K),
    B=TensorDef(T, S.K, S.N),
    C=TensorDef(T, S.M, S.N, output=True)):
  C[D.m, D.n] += A[D.m, D.k] * B[D.k, D.n]


def build_SpMM(attr: EncodingAttr):
  """Build SpMM kernel.

  This method generates a linalg op with for matrix multiplication using
  just the Python API. Effectively, a generic linalg op is constructed
  that computes C(i,j) += A(i,k) * B(k,j) for annotated matrix A.
  """
  module = Module.create()
  f64 = ir.F64Type.get()
  a = RankedTensorType.get([3, 4], f64, attr)
  b = RankedTensorType.get([4, 2], f64)
  c = RankedTensorType.get([3, 2], f64)
  arguments = [a, b, c]
  with InsertionPoint(module.body):

    @builtin.FuncOp.from_py_func(*arguments)
    def spMxM(*args):
      return matmul_dsl(args[0], args[1], outs=[args[2]])

  return module


def boilerplate(attr: EncodingAttr):
  """Returns boilerplate main method.

  This method sets up a boilerplate main method that calls the generated
  sparse kernel. For convenience, this part is purely done as string input.
  """
  return f"""
func @main(%c: tensor<3x2xf64>) -> tensor<3x2xf64>
  attributes {{ llvm.emit_c_interface }} {{
  %0 = constant dense<[ [ 1.1,  0.0,  0.0,  1.4 ],
                        [ 0.0,  0.0,  0.0,  0.0 ],
                        [ 0.0,  0.0,  3.3,  0.0 ]]> : tensor<3x4xf64>
  %a = sparse_tensor.convert %0 : tensor<3x4xf64> to tensor<3x4xf64, {attr}>
  %b = constant dense<[ [ 1.0,  2.0 ],
                        [ 4.0,  3.0 ],
                        [ 5.0,  6.0 ],
                        [ 8.0,  7.0 ]]> : tensor<4x2xf64>
  %1 = call @spMxM(%a, %b, %c) : (tensor<3x4xf64, {attr}>,
                                  tensor<4x2xf64>,
                                  tensor<3x2xf64>) -> tensor<3x2xf64>
  return %1 : tensor<3x2xf64>
}}
"""


def build_compile_and_run_SpMM(attr: EncodingAttr, support_lib: str, compiler):
  # Build.
  module = build_SpMM(attr)
  func = str(module.operation.regions[0].blocks[0].operations[0].operation)
  module = Module.parse(func + boilerplate(attr))
  # Compile.
  compiler(module)
  execution_engine = ExecutionEngine(
      module, opt_level=0, shared_libs=[support_lib])
  # Set up numpy input, invoke the kernel, and get numpy output.
  # Built-in bufferization uses in-out buffers.
  # TODO: replace with inplace comprehensive bufferization.
  Cin = np.zeros((3, 2), np.double)
  Cout = np.zeros((3, 2), np.double)
  Cin_memref_ptr = ctypes.pointer(
      ctypes.pointer(get_ranked_memref_descriptor(Cin)))
  Cout_memref_ptr = ctypes.pointer(
      ctypes.pointer(get_ranked_memref_descriptor(Cout)))
  execution_engine.invoke('main', Cout_memref_ptr, Cin_memref_ptr)
  Cresult = ranked_memref_to_numpy(Cout_memref_ptr[0])

  # Sanity check on computed result.
  expected = [[12.3, 12.0], [0.0, 0.0], [16.5, 19.8]]
  if np.allclose(Cresult, expected):
    pass
  else:
    quit(f'FAILURE')


class SparseCompiler:
  """Sparse compiler passes."""

  def __init__(self, options: str):
    pipeline = (
        f'sparsification{{{options}}},'
        f'sparse-tensor-conversion,'
        f'builtin.func(convert-linalg-to-loops,convert-vector-to-scf),'
        f'convert-scf-to-std,'
        f'func-bufferize,'
        f'tensor-constant-bufferize,'
        f'builtin.func(tensor-bufferize,std-bufferize,finalizing-bufferize),'
        f'convert-vector-to-llvm{{reassociate-fp-reductions=1 enable-index-optimizations=1}},'
        f'convert-memref-to-llvm,'
        f'convert-std-to-llvm')
    self.pipeline = pipeline

  def __call__(self, module: Module):
    PassManager.parse(self.pipeline).run(module)


# CHECK-LABEL: TEST: testSpMM
# CHECK: Passed 72 tests
@run
def testSpMM():
  support_lib = os.getenv('SUPPORT_LIB')
  with Context() as ctx, Location.unknown():
    count = 0
    # Fixed compiler optimization strategy.
    # TODO: explore state space here too
    par = 0
    vec = 0
    vl = 1
    e = False
    opt = (f'parallelization-strategy={par} '
           f'vectorization-strategy={vec} '
           f'vl={vl} enable-simd-index32={e}')
    # Exhaustive loop over various ways to annotate a kernel with
    # a *single* sparse tensor. Even this subset already gives
    # quite a large state space!
    levels = [[DimLevelType.dense, DimLevelType.dense],
              [DimLevelType.dense, DimLevelType.compressed],
              [DimLevelType.compressed, DimLevelType.dense],
              [DimLevelType.compressed, DimLevelType.compressed]]
    orderings = [
        AffineMap.get_permutation([0, 1]),
        AffineMap.get_permutation([1, 0])
    ]
    bitwidths = [0, 8, 32]
    for levels in levels:
      for ordering in orderings:
        for pwidth in bitwidths:
          for iwidth in bitwidths:
            attr = EncodingAttr.get(levels, ordering, pwidth, iwidth)
            compiler = SparseCompiler(options=opt)
            build_compile_and_run_SpMM(attr, support_lib, compiler)
            count = count + 1
    print('Passed ', count, 'tests')
