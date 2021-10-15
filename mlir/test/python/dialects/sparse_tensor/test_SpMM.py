# RUN: SUPPORT_LIB=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext %PYTHON %s | FileCheck %s

import ctypes
import numpy as np
import os

import mlir.all_passes_registration

from mlir import ir
from mlir import runtime as rt
from mlir import execution_engine
from mlir import passmanager

from mlir.dialects import sparse_tensor as st
from mlir.dialects import builtin
from mlir.dialects.linalg.opdsl import lang as dsl


def run(f):
  print('\nTEST:', f.__name__)
  f()
  return f


@dsl.linalg_structured_op
def matmul_dsl(
    A=dsl.TensorDef(dsl.T, dsl.S.M, dsl.S.K),
    B=dsl.TensorDef(dsl.T, dsl.S.K, dsl.S.N),
    C=dsl.TensorDef(dsl.T, dsl.S.M, dsl.S.N, output=True)):
  C[dsl.D.m, dsl.D.n] += A[dsl.D.m, dsl.D.k] * B[dsl.D.k, dsl.D.n]


def build_SpMM(attr: st.EncodingAttr):
  """Build SpMM kernel.

  This method generates a linalg op with for matrix multiplication using
  just the Python API. Effectively, a generic linalg op is constructed
  that computes C(i,j) += A(i,k) * B(k,j) for annotated matrix A.
  """
  module = ir.Module.create()
  f64 = ir.F64Type.get()
  a = ir.RankedTensorType.get([3, 4], f64, attr)
  b = ir.RankedTensorType.get([4, 2], f64)
  c = ir.RankedTensorType.get([3, 2], f64)
  arguments = [a, b, c]
  with ir.InsertionPoint(module.body):

    @builtin.FuncOp.from_py_func(*arguments)
    def spMxM(*args):
      return matmul_dsl(args[0], args[1], outs=[args[2]])

  return module


def boilerplate(attr: st.EncodingAttr):
  """Returns boilerplate main method.

  This method sets up a boilerplate main method that takes three tensors
  (a, b, c), converts the first tensor a into s sparse tensor, and then
  calls the sparse kernel for matrix multiplication. For convenience,
  this part is purely done as string input.
  """
  return f"""
func @main(%ad: tensor<3x4xf64>, %b: tensor<4x2xf64>, %c: tensor<3x2xf64>) -> tensor<3x2xf64>
  attributes {{ llvm.emit_c_interface }} {{
  %a = sparse_tensor.convert %ad : tensor<3x4xf64> to tensor<3x4xf64, {attr}>
  %0 = call @spMxM(%a, %b, %c) : (tensor<3x4xf64, {attr}>,
                                  tensor<4x2xf64>,
                                  tensor<3x2xf64>) -> tensor<3x2xf64>
  return %0 : tensor<3x2xf64>
}}
"""


def build_compile_and_run_SpMM(attr: st.EncodingAttr, support_lib: str,
                               compiler):
  # Build.
  module = build_SpMM(attr)
  func = str(module.operation.regions[0].blocks[0].operations[0].operation)
  module = ir.Module.parse(func + boilerplate(attr))

  # Compile.
  compiler(module)
  engine = execution_engine.ExecutionEngine(
      module, opt_level=0, shared_libs=[support_lib])

  # Set up numpy input and buffer for output.
  a = np.array(
      [[1.1, 0.0, 0.0, 1.4], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 3.3, 0.0]],
      np.float64)
  b = np.array([[1.0, 2.0], [4.0, 3.0], [5.0, 6.0], [8.0, 7.0]], np.float64)
  c = np.zeros((3, 2), np.float64)
  out = np.zeros((3, 2), np.float64)

  mem_a = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(a)))
  mem_b = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(b)))
  mem_c = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(c)))
  mem_out = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(out)))

  # Invoke the kernel and get numpy output.
  # Built-in bufferization uses in-out buffers.
  # TODO: replace with inplace comprehensive bufferization.
  engine.invoke('main', mem_out, mem_a, mem_b, mem_c)

  # Sanity check on computed result.
  expected = np.matmul(a, b);
  c = rt.ranked_memref_to_numpy(mem_out[0])
  if np.allclose(c, expected):
    pass
  else:
    quit(f'FAILURE')


class SparseCompiler:
  """Sparse compiler passes."""

  def __init__(self, options: str):
    pipeline = (
        f'builtin.func(linalg-generalize-named-ops,linalg-fuse-elementwise-ops),'
        f'sparsification{{{options}}},'
        f'sparse-tensor-conversion,'
        f'builtin.func(linalg-bufferize,convert-linalg-to-loops,convert-vector-to-scf),'
        f'convert-scf-to-std,'
        f'func-bufferize,'
        f'tensor-constant-bufferize,'
        f'builtin.func(tensor-bufferize,std-bufferize,finalizing-bufferize),'
        f'convert-vector-to-llvm{{reassociate-fp-reductions=1 enable-index-optimizations=1}},'
        f'lower-affine,'
        f'convert-memref-to-llvm,'
        f'convert-std-to-llvm,'
        f'reconcile-unrealized-casts')
    self.pipeline = pipeline

  def __call__(self, module: ir.Module):
    passmanager.PassManager.parse(self.pipeline).run(module)


# CHECK-LABEL: TEST: testSpMM
# CHECK: Passed 8 tests
@run
def testSpMM():
  # Obtain path to runtime support library.
  support_lib = os.getenv('SUPPORT_LIB')
  assert os.path.exists(support_lib), f'{support_lib} does not exist'

  with ir.Context() as ctx, ir.Location.unknown():
    count = 0
    # Loop over various ways to compile and annotate the SpMM kernel with
    # a *single* sparse tensor. Note that we deliberate do not exhaustively
    # search the full state space to reduce runtime of the test. It is
    # straightforward to adapt the code below to explore more combinations.
    par = 0
    vec = 0
    vl = 1
    e = False
    opt = (f'parallelization-strategy={par} '
           f'vectorization-strategy={vec} '
           f'vl={vl} enable-simd-index32={e}')
    levels = [[st.DimLevelType.dense, st.DimLevelType.dense],
              [st.DimLevelType.dense, st.DimLevelType.compressed],
              [st.DimLevelType.compressed, st.DimLevelType.dense],
              [st.DimLevelType.compressed, st.DimLevelType.compressed]]
    orderings = [
        ir.AffineMap.get_permutation([0, 1]),
        ir.AffineMap.get_permutation([1, 0])
    ]
    bitwidths = [0]
    for level in levels:
      for ordering in orderings:
        for pwidth in bitwidths:
          for iwidth in bitwidths:
            attr = st.EncodingAttr.get(level, ordering, pwidth, iwidth)
            compiler = SparseCompiler(options=opt)
            build_compile_and_run_SpMM(attr, support_lib, compiler)
            count = count + 1
    print('Passed ', count, 'tests')
