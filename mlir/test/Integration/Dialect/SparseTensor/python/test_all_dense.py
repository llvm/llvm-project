# RUN: env SUPPORT_LIB=%mlir_c_runner_utils \
# RUN:   %PYTHON %s | FileCheck %s

import ctypes
import os
import sys
import tempfile

from mlir import ir
from mlir import runtime as rt
from mlir.dialects import builtin
from mlir.dialects import sparse_tensor as st
import numpy as np

_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_SCRIPT_PATH)
from tools import sparsifier


def boilerplate():
    """Returns boilerplate main method."""
    return """
#Dense = #sparse_tensor.encoding<{
  map = (i, j) -> (i: dense, j: dense)
}>

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @add(%st_0 : tensor<3x4xf64, #Dense>,
               %st_1 : tensor<3x4xf64, #Dense>) attributes { llvm.emit_c_interface } {
  %out_st = tensor.empty() : tensor<3x4xf64, #Dense>
  %res = linalg.generic {indexing_maps = [#map, #map, #map],
                         iterator_types = ["parallel", "parallel"]}
                         ins(%st_0, %st_1 : tensor<3x4xf64, #Dense>, tensor<3x4xf64, #Dense>)
                         outs(%out_st : tensor<3x4xf64, #Dense>) {
  ^bb0(%in_0: f64, %in_1: f64, %out: f64):
    %2 = sparse_tensor.binary %in_0, %in_1 : f64, f64 to f64
    overlap = {
      ^bb0(%arg1: f64, %arg2: f64):
        %3 = arith.addf %arg1, %arg2 : f64
        sparse_tensor.yield %3 : f64
    }
    left = {
      ^bb0(%arg1: f64):
        sparse_tensor.yield %arg1 : f64
    }
    right = {
      ^bb0(%arg1: f64):
        sparse_tensor.yield %arg1 : f64
    }
    linalg.yield %2 : f64
  } -> tensor<3x4xf64, #Dense>
  sparse_tensor.print %res : tensor<3x4xf64, #Dense>
  return
}
"""


def main():
    support_lib = os.getenv("SUPPORT_LIB")
    assert support_lib is not None, "SUPPORT_LIB is undefined"
    if not os.path.exists(support_lib):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), support_lib)

    # CHECK-LABEL: TEST: all dense
    # CHECK: ---- Sparse Tensor ----
    # CHECK: nse = 12
    # CHECK: dim = ( 3, 4 )
    # CHECK: lvl = ( 3, 4 )
    # CHECK: values : ( 1, 1, 0, 1, 0, 6, 2, 3, 0, 0, 0, 2 )
    # CHECK: ----
    print("\nTEST: all dense")
    with ir.Context() as ctx, ir.Location.unknown():
        compiler = sparsifier.Sparsifier(
            extras="sparse-assembler,",
            options="enable-runtime-library=false",
            opt_level=2,
            shared_libs=[support_lib],
        )
        module = ir.Module.parse(boilerplate())
        engine = compiler.compile_and_jit(module)
        print(module)

        a = np.array([1, 0, 0, 1, 0, 2, 2, 0, 0, 0, 0, 1], dtype=np.float64)
        b = np.array([0, 1, 0, 0, 0, 4, 0, 3, 0, 0, 0, 1], dtype=np.float64)
        mem_a = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(a)))
        mem_b = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(b)))

        # Invoke the kernel and get numpy output.
        # Built-in bufferization uses in-out buffers.
        engine.invoke("add", mem_a, mem_b)


if __name__ == "__main__":
    main()
