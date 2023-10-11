// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(scf-parallel-loop-tiling{parallel-loop-tile-sizes=0}))' -split-input-file | FileCheck %s

func.func @parallel_loop(%arg0 : index, %arg1 : index, %arg2 : index,
                    %arg3 : index, %arg4 : index, %arg5 : index,
		    %A: memref<?x?xf32>, %B: memref<?x?xf32>,
                    %C: memref<?x?xf32>, %result: memref<?x?xf32>) {
  scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3) step (%arg4, %arg5) {
    %B_elem = memref.load %B[%i0, %i1] : memref<?x?xf32>
    %C_elem = memref.load %C[%i0, %i1] : memref<?x?xf32>
    %sum_elem = arith.addf %B_elem, %C_elem : f32
    memref.store %sum_elem, %result[%i0, %i1] : memref<?x?xf32>
  }
  return
}

// CHECK-LABEL:   func @parallel_loop(
// CHECK-SAME:        [[ARG1:%.*]]: index, [[ARG2:%.*]]: index, [[ARG3:%.*]]: index, [[ARG4:%.*]]: index, [[ARG5:%.*]]: index, [[ARG6:%.*]]: index, [[ARG7:%.*]]: memref<?x?xf32>, [[ARG8:%.*]]: memref<?x?xf32>, [[ARG9:%.*]]: memref<?x?xf32>, [[ARG10:%.*]]: memref<?x?xf32>) {
// CHECK:           scf.parallel ([[V1:%.*]], [[V2:%.*]]) = ([[ARG1]], [[ARG2]]) to ([[ARG3]], [[ARG4]]) step ([[ARG5]]
// CHECK:           }
// CHECK:           return