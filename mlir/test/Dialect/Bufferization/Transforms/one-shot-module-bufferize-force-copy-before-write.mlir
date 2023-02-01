// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries=1 no-analysis-func-filter=contains_to_memref_op" -drop-equivalent-buffer-results --split-input-file | FileCheck %s

// ToMemref ops do not pass analysis step. CopyBeforeWrite will be true only for the
// FuncOp "contains_to_memref_op" since it is specified in no-analysis-func-filter.

module {
  // CHECK-LABEL:   func.func @foo(
  // CHECK-SAME:                   %[[arg0:.*]]: memref<?xf32, strided<[?], offset: ?>>) {
  func.func @foo(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-NEXT:      %[[c0:.*]] = arith.constant 0 : index
    %cst = arith.constant 1.000000e+00 : f32

  // CHECK-NEXT:      %[[c1:.*]] = arith.constant 1.000000e+00 : f32
    %c0 = arith.constant 0 : index

  // CHECK-NEXT:      memref.store %[[c1]], %[[arg0]]{{\[}}%[[c0]]] : memref<?xf32, strided<[?], offset: ?>>
    %inserted = tensor.insert %cst into %arg0[%c0] : tensor<?xf32>

    return %inserted : tensor<?xf32>
  }

  // CHECK-LABEL:   func.func @contains_to_memref_op(
  // CHECK-SAME:                                     %[[arg0:.*]]: memref<?xf32, strided<[?], offset: ?>>,
  // CHECK-SAME:                                     %[[arg1:.*]]: index) -> vector<5xf32> {
  func.func @contains_to_memref_op(%arg0: tensor<?xf32> {bufferization.writable = true}, %arg1: index) -> vector<5xf32> {

    %0 = bufferization.to_memref %arg0 : memref<?xf32>

    // CHECK:           %[[c0:.*]] = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32

    // CHECK:           %[[dim:.*]] = memref.dim %[[arg0]], %[[c0]] : memref<?xf32, strided<[?], offset: ?>>
    // CHECK:           %[[alloc:.*]] = memref.alloc(%[[dim]]) : memref<?xf32>
    // CHECK:           memref.copy %[[arg0]], %[[alloc]] : memref<?xf32, strided<[?], offset: ?>> to memref<?xf32>
    // CHECK:           vector.transfer_read
    %1 = vector.transfer_read %0[%arg1], %cst : memref<?xf32>, vector<5xf32>
    return %1 : vector<5xf32>
  }
}