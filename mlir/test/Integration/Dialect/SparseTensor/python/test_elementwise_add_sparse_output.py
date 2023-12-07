# RUN: env SUPPORT_LIB=%mlir_c_runner_utils %PYTHON %s | FileCheck %s

import ctypes
import numpy as np
import os
import sys

from mlir import ir
from mlir import runtime as rt
from mlir.dialects import sparse_tensor as st
from mlir.dialects import builtin
from mlir.dialects.linalg.opdsl import lang as dsl

_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_SCRIPT_PATH)
from tools import np_to_sparse_tensor as test_tools
from tools import sparse_compiler

# TODO: Use linalg_structured_op to generate the kernel after making it to
# handle sparse tensor outputs.
_KERNEL_STR = """
#DCSR = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed", "compressed" ]
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

func.func @sparse_add_elt(
    %arga: tensor<3x4xf64, #DCSR>, %argb: tensor<3x4xf64, #DCSR>) -> tensor<3x4xf64, #DCSR> {
  %argx = bufferization.alloc_tensor() : tensor<3x4xf64, #DCSR>
  %0 = linalg.generic #trait_add_elt
    ins(%arga, %argb: tensor<3x4xf64, #DCSR>, tensor<3x4xf64, #DCSR>)
    outs(%argx: tensor<3x4xf64, #DCSR>) {
      ^bb(%a: f64, %b: f64, %x: f64):
        %1 = arith.addf %a, %b : f64
        linalg.yield %1 : f64
  } -> tensor<3x4xf64, #DCSR>
  return %0 : tensor<3x4xf64, #DCSR>
}

func.func @main(%ad: tensor<3x4xf64>, %bd: tensor<3x4xf64>) -> tensor<3x4xf64, #DCSR>
  attributes { llvm.emit_c_interface } {
  %a = sparse_tensor.convert %ad : tensor<3x4xf64> to tensor<3x4xf64, #DCSR>
  %b = sparse_tensor.convert %bd : tensor<3x4xf64> to tensor<3x4xf64, #DCSR>
  %0 = call @sparse_add_elt(%a, %b) : (tensor<3x4xf64, #DCSR>, tensor<3x4xf64, #DCSR>) -> tensor<3x4xf64, #DCSR>
  return %0 : tensor<3x4xf64, #DCSR>
}
"""


def _run_test(support_lib, kernel):
    """Compiles, runs and checks results."""
    compiler = sparse_compiler.SparseCompiler(
        options="", opt_level=2, shared_libs=[support_lib]
    )
    module = ir.Module.parse(kernel)
    engine = compiler.compile_and_jit(module)

    # Set up numpy inputs and buffer for output.
    a = np.array(
        [[1.1, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 6.6, 0.0]], np.float64
    )
    b = np.array(
        [[1.1, 0.0, 0.0, 2.8], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], np.float64
    )

    mem_a = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(a)))
    mem_b = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(b)))

    # The sparse tensor output is a pointer to pointer of char.
    out = ctypes.c_char(0)
    mem_out = ctypes.pointer(ctypes.pointer(out))

    # Invoke the kernel.
    engine.invoke("main", mem_a, mem_b, mem_out)

    # Retrieve and check the result.
    rank, nse, shape, values, indices = test_tools.sparse_tensor_to_coo_tensor(
        support_lib, mem_out[0], np.float64
    )

    # CHECK: PASSED
    if np.allclose(values, [2.2, 2.8, 6.6]) and np.allclose(
        indices, [[0, 0], [0, 3], [2, 2]]
    ):
        print("PASSED")
    else:
        quit("FAILURE")


def test_elementwise_add():
    # Obtain path to runtime support library.
    support_lib = os.getenv("SUPPORT_LIB")
    assert support_lib is not None, "SUPPORT_LIB is undefined"
    assert os.path.exists(support_lib), f"{support_lib} does not exist"
    with ir.Context() as ctx, ir.Location.unknown():
        _run_test(support_lib, _KERNEL_STR)


test_elementwise_add()
