# RUN: SUPPORT_LIB=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
# RUN:   %PYTHON %s | FileCheck %s

import ctypes
import os
import tempfile

import mlir.all_passes_registration

from mlir import execution_engine
from mlir import ir
from mlir import passmanager
from mlir import runtime as rt

from mlir.dialects import builtin
from mlir.dialects import sparse_tensor as st


# TODO: move more into actual IR building.
def boilerplate(attr: st.EncodingAttr):
  """Returns boilerplate main method."""
  return f"""
func @main(%p : !llvm.ptr<i8>) -> () attributes {{ llvm.emit_c_interface }} {{
  %d = arith.constant sparse<[[0, 0], [1, 1], [0, 9], [9, 0], [4, 4]],
                             [1.0, 2.0, 3.0, 4.0, 5.0]> : tensor<10x10xf64>
  %a = sparse_tensor.convert %d : tensor<10x10xf64> to tensor<10x10xf64, {attr}>
  sparse_tensor.out %a, %p : tensor<10x10xf64, {attr}>, !llvm.ptr<i8>
  return
}}
"""


def expected():
  """Returns expected contents of output.

  Regardless of the dimension ordering, compression, and bitwidths that are
  used in the sparse tensor, the output is always lexicographically sorted
  by natural index order.
  """
  return f"""; extended FROSTT format
2 5
10 10
1 1 1
1 10 3
2 2 2
5 5 5
10 1 4
"""


def build_compile_and_run_output(attr: st.EncodingAttr, support_lib: str,
                                 compiler):
  # Build and Compile.
  module = ir.Module.parse(boilerplate(attr))
  compiler(module)
  engine = execution_engine.ExecutionEngine(
      module, opt_level=0, shared_libs=[support_lib])

  # Invoke the kernel and compare output.
  with tempfile.TemporaryDirectory() as test_dir:
    out = os.path.join(test_dir, 'out.tns')
    buf = out.encode('utf-8')
    mem_a = ctypes.pointer(ctypes.pointer(ctypes.create_string_buffer(buf)))
    engine.invoke('main', mem_a)

    actual = open(out).read()
    if actual != expected():
      quit('FAILURE')


class SparseCompiler:
  """Sparse compiler passes."""

  def __init__(self):
    pipeline = (
        f'builtin.func(linalg-generalize-named-ops,linalg-fuse-elementwise-ops),'
        f'sparsification,'
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


def main():
  support_lib = os.getenv('SUPPORT_LIB')
  assert support_lib is not None, 'SUPPORT_LIB is undefined'
  if not os.path.exists(support_lib):
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                            support_lib)

  # CHECK-LABEL: TEST: test_output
  print('\nTEST: test_output')
  count = 0
  with ir.Context() as ctx, ir.Location.unknown():
    # Loop over various sparse types: CSR, DCSR, CSC, DCSC.
    levels = [[st.DimLevelType.dense, st.DimLevelType.compressed],
              [st.DimLevelType.compressed, st.DimLevelType.compressed]]
    orderings = [
        ir.AffineMap.get_permutation([0, 1]),
        ir.AffineMap.get_permutation([1, 0])
    ]
    bitwidths = [8, 16, 32, 64]
    for level in levels:
      for ordering in orderings:
        for bwidth in bitwidths:
          attr = st.EncodingAttr.get(level, ordering, bwidth, bwidth)
          compiler = SparseCompiler()
          build_compile_and_run_output(attr, support_lib, compiler)
          count = count + 1

  # CHECK: Passed 16 tests
  print('Passed', count, 'tests')


if __name__ == '__main__':
  main()
