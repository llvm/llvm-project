// RUN: mlir-opt %s -test-one-shot-module-bufferize="test-tensor-to-dynamic-memref" -split-input-file -verify-diagnostics | FileCheck %s

// The buffer type of a test tensor has dynamic extents; a copy takes the
// static shape of the buffer it copies.

// CHECK-LABEL: func.func @custom_types_copy_static_buffer(
// CHECK-SAME:    %[[m:[^:]+]]: memref<4xf64>
// CHECK: %[[alloc:.+]] = memref.alloc() {{.*}}: memref<4xf64>
// CHECK: memref.copy %[[m]], %[[alloc]]
func.func @custom_types_copy_static_buffer(%m: memref<4xf64>)
    -> (!test.test_tensor<[4], f64>, !test.test_tensor<[4], f64>) {
  %t = bufferization.to_tensor %m restrict writable
    : memref<4xf64> to !test.test_tensor<[4], f64>
  %0 = "test.dummy_tensor_op"(%t)
    : (!test.test_tensor<[4], f64>) -> !test.test_tensor<[4], f64>
  %1 = "test.dummy_tensor_op"(%t)
    : (!test.test_tensor<[4], f64>) -> !test.test_tensor<[4], f64>
  return %0, %1 : !test.test_tensor<[4], f64>, !test.test_tensor<[4], f64>
}

// -----

func.func @custom_types_alloc_dynamic_buffer() -> !test.test_tensor<[4], f64> {
  // expected-error @+2 {{allocating 'memref<?xf64>' needs a size for each dynamic dimension, which the op takes only for a ranked tensor result}}
  // expected-error @+1 {{failed to bufferize op}}
  %e = bufferization.alloc_tensor() : !test.test_tensor<[4], f64>
  %0 = "test.dummy_tensor_op"(%e)
    : (!test.test_tensor<[4], f64>) -> !test.test_tensor<[4], f64>
  return %0 : !test.test_tensor<[4], f64>
}
