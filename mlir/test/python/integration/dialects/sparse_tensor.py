# RUN: %PYTHON %s | FileCheck %s

import ctypes
import ctypes.util

import numpy as np

from mlir.ir import Context, Module
import mlir.runtime as rt
from mlir import execution_engine, passmanager


with Context():
    module_coo = Module.parse(
    """
    #COO = #sparse_tensor.encoding<{
        map = (i, j) -> (i : compressed(nonunique), j : singleton), posWidth = 64, crdWidth = 64
    }>
    func.func @build(%a1 : tensor<?xi64>, %a2 : tensor<?x2xi64>, %a3 : tensor<?xf64>) -> tensor<3x4xf64, #COO> attributes { llvm.emit_c_interface } {
        %st = sparse_tensor.assemble (%a1, %a2), %a3 : (tensor<?xi64>, tensor<?x2xi64>), tensor<?xf64> to tensor<3x4xf64, #COO>
        return %st : tensor<3x4xf64, #COO>
    }
    """
    )

    module_add = Module.parse(
    """
    #COO = #sparse_tensor.encoding<{
        map = (i, j) -> (i : compressed(nonunique), j : singleton), posWidth = 64, crdWidth = 64
    }>
    #map = affine_map<(d0, d1) -> (d0, d1)>
    func.func @add(%st_0 : tensor<3x4xf64, #COO>, %st_1 : tensor<3x4xf64, #COO>) -> tensor<3x4xf64, #COO> attributes { llvm.emit_c_interface } {
        %out_st = tensor.empty() : tensor<3x4xf64, #COO>
        %res = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%st_0, %st_1 : tensor<3x4xf64, #COO>, tensor<3x4xf64, #COO>) outs(%out_st : tensor<3x4xf64, #COO>) {
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
        } -> tensor<3x4xf64, #COO>
        return %res : tensor<3x4xf64, #COO>
    }
    """
    )

    pm = passmanager.PassManager.parse("builtin.module(sparsifier{create-sparse-deallocs=1})")
    pm.run(module_add.operation)
    pm.run(module_coo.operation)

    import sys

    if sys.platform == "win32":
        shared_libs = [
            "../../../../../../bin/mlir_c_runner_utils.dll",
        ]
    elif sys.platform == "darwin":
        shared_libs = [
            "../../../../../../lib/libmlir_c_runner_utils.dylib",
        ]
    else:
        shared_libs = [
            "../../../../../../lib/libmlir_c_runner_utils.so",
        ]

    ee_build = execution_engine.ExecutionEngine(module_coo, opt_level=2, shared_libs=shared_libs)
    ee_add = execution_engine.ExecutionEngine(module_add, opt_level=2, shared_libs=shared_libs)

    pos = np.array([0, 5], dtype=np.int64)
    idx = np.array([[0, 0], [0, 1], [1, 1], [2, 2], [1, 0]], dtype=np.int64)
    data = np.array([1.1, 2.2, 3.3, 5.5, 6.6], dtype=np.float64)

    p_pos = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(pos)))
    p_idx = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(idx)))
    p_data = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(data)))

    out = ctypes.c_void_p()
    ret = ee_build.invoke("build", p_pos, p_idx, p_data, ctypes.pointer(out))

    add_out = ctypes.c_void_p()

    ret = ee_add.invoke("add", ctypes.pointer(out), ctypes.pointer(out), ctypes.pointer(add_out))

    # CHECK: RESULT: -1
    print(f"RESULT: -1")
