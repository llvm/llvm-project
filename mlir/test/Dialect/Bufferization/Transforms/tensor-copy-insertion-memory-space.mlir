// RUN: mlir-opt %s -tensor-copy-insertion="must-infer-memory-space" -split-input-file | FileCheck %s

// CHECK-LABEL: func @unknown_op_copy
func.func @unknown_op_copy() -> (tensor<10xf32>, tensor<10xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  // CHECK: %[[dummy:.*]] = "test.dummy_op"() : () -> tensor<10xf32>
  %t = "test.dummy_op"() : () -> tensor<10xf32>
  // CHECK: %[[copy:.*]] = bufferization.alloc_tensor() copy(%[[dummy]]) {bufferization.escape = [false]} : tensor<10xf32>
  %s = tensor.insert %cst into %t[%c0] : tensor<10xf32>
  return %s, %t : tensor<10xf32>, tensor<10xf32>
}

// -----

// CHECK-LABEL: func @alloc_tensor_copy
func.func @alloc_tensor_copy() -> (tensor<10xf32>, tensor<10xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  // CHECK: bufferization.alloc_tensor() {bufferization.escape = [false], memory_space = 1 : ui64} : tensor<10xf32>
  %t = bufferization.alloc_tensor() {memory_space = 1 : ui64} : tensor<10xf32>
  // CHECK: bufferization.alloc_tensor() {bufferization.escape = [false], memory_space = 1 : ui64} : tensor<10xf32>
  %s = tensor.insert %cst into %t[%c0] : tensor<10xf32>
  return %s, %t : tensor<10xf32>, tensor<10xf32>
}
