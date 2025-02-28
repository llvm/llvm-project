// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(func.func(remove-single-iteration-loop))' -split-input-file | FileCheck %s

// -----

// CHECK-LABEL: func @loop_once(
func.func @loop_once(%arg : index) -> index{
  %0 = affine.for %iv = 2 to 3 step 1 iter_args(%arg1 = %arg) -> index {
    %sum = arith.addi %arg1, %iv : index
    affine.yield %sum : index
  }
  return %0 : index
}
// CHECK-SAME: %[[ARG:.*]]: index)
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[SUM:.*]] = arith.addi %[[ARG]], %[[C2]] : index
// CHECK: return %[[SUM]] : index

// -----

// CHECK-LABEL: func @invalid_loop(
func.func @invalid_loop(%arg : index) -> index{
  %0 = affine.for %iv = 4 to 3 step 1 iter_args(%arg1 = %arg) -> index {
    %sum = arith.addi %arg1, %iv : index
    affine.yield %sum : index
  }
  return %0 : index
}
// CHECK-SAME: %[[ARG:.*]]: index)
// CHECK: return %[[ARG]] : index

// -----

// CHECK-LABEL: func @gpu_invalid_loop
func.func @gpu_invalid_loop() {
  %0 = arith.constant 0 :index
  %1 = arith.constant 2 : index
  gpu.launch blocks(%bx, %by, %bz) in (%sz_bx = %1, %sz_by = %1, %sz_bz = %1)
             threads(%tx, %ty, %tz) in (%sz_tx = %1, %sz_ty = %1, %sz_tz = %1) {
    %threadid = gpu.thread_id x
    affine.for %iv = %tx to 0 step 2 iter_args(%arg = %0) -> index {
      %3 = arith.addi %arg, %0 : index
      affine.yield %3 : index
    }
    gpu.terminator
  }
  // CHECK-NEXT: return
  return
}
