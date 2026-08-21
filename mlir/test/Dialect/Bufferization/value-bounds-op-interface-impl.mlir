// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(test-affine-reify-value-bounds))' -verify-diagnostics \
// RUN:     -split-input-file | FileCheck %s

// CHECK-LABEL: func @to_buffer(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x4xf32>
//       CHECK:   %[[c0:.*]] = arith.constant 0 : index
//       CHECK:   %[[dim:.*]] = tensor.dim %[[t]], %[[c0]]
//       CHECK:   %[[c4:.*]] = arith.constant 4 : index
//       CHECK:   return %[[dim]], %[[c4]]
func.func @to_buffer(%t: tensor<?x4xf32>) -> (index, index) {
  %0 = bufferization.to_buffer %t : tensor<?x4xf32> to memref<?x4xf32>
  %1 = "test.reify_bound"(%0) {dim = 0} : (memref<?x4xf32>) -> (index)
  %2 = "test.reify_bound"(%0) {dim = 1} : (memref<?x4xf32>) -> (index)
  return %1, %2 : index, index
}

// -----

// The query goes through the op and reaches the ops that define the tensor.

// CHECK-LABEL: func @to_buffer_constant(
//       CHECK:   %[[c5:.*]] = arith.constant 5 : index
//       CHECK:   return %[[c5]]
func.func @to_buffer_constant() -> index {
  %0 = tensor.empty() : tensor<5x4xf32>
  %1 = tensor.cast %0 : tensor<5x4xf32> to tensor<?x4xf32>
  %2 = bufferization.to_buffer %1 : tensor<?x4xf32> to memref<?x4xf32>
  %3 = "test.reify_bound"(%2) {dim = 0, constant} : (memref<?x4xf32>) -> (index)
  return %3 : index
}

// -----

// CHECK-LABEL: func @to_tensor(
//  CHECK-SAME:     %[[m:.*]]: memref<?x4xf32>
//       CHECK:   %[[c0:.*]] = arith.constant 0 : index
//       CHECK:   %[[dim:.*]] = memref.dim %[[m]], %[[c0]]
//       CHECK:   %[[c4:.*]] = arith.constant 4 : index
//       CHECK:   return %[[dim]], %[[c4]]
func.func @to_tensor(%m: memref<?x4xf32>) -> (index, index) {
  %0 = bufferization.to_tensor %m : memref<?x4xf32> to tensor<?x4xf32>
  %1 = "test.reify_bound"(%0) {dim = 0} : (tensor<?x4xf32>) -> (index)
  %2 = "test.reify_bound"(%0) {dim = 1} : (tensor<?x4xf32>) -> (index)
  return %1, %2 : index, index
}

// -----

// The query goes through the op and reaches the ops that define the buffer.

// CHECK-LABEL: func @to_tensor_constant(
//       CHECK:   %[[c5:.*]] = arith.constant 5 : index
//       CHECK:   return %[[c5]]
func.func @to_tensor_constant() -> index {
  %0 = memref.alloc() : memref<5x4xf32>
  %1 = memref.cast %0 : memref<5x4xf32> to memref<?x4xf32>
  %2 = bufferization.to_tensor %1 : memref<?x4xf32> to tensor<?x4xf32>
  %3 = "test.reify_bound"(%2) {dim = 0, constant} : (tensor<?x4xf32>) -> (index)
  return %3 : index
}
