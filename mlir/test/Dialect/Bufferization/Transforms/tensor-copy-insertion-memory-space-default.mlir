// RUN: mlir-opt %s -test-tensor-copy-insertion -split-input-file | FileCheck %s

// -----

// CHECK-LABEL: func @alloc_tensor_default_memory_space
func.func @alloc_tensor_default_memory_space() -> (tensor<10xf32>, tensor<10xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  // CHECK: bufferization.alloc_tensor() : tensor<10xf32>
  %t = bufferization.alloc_tensor() : tensor<10xf32>
  // CHECK: bufferization.alloc_tensor() : tensor<10xf32>
  %s = tensor.insert %cst into %t[%c0] : tensor<10xf32>
  return %s, %t : tensor<10xf32>, tensor<10xf32>
}
