// RUN: mlir-opt %s -test-one-shot-module-bufferize="test-tensor-to-dynamic-memref" -verify-diagnostics

// A copy into a memref with a dynamic extent whose source buffer is not a
// memref has no extent to be allocated with.

func.func @custom_buffer_copied_to_dynamic_memref(
    %m: !test.test_memref<[-9223372036854775808], f64>)
    -> !test.test_tensor<[-9223372036854775808], f64> {
  %t = bufferization.to_tensor %m restrict writable
    : !test.test_memref<[-9223372036854775808], f64>
      to !test.test_tensor<[-9223372036854775808], f64>
  // expected-error @+2 {{allocating 'memref<?xf64>' needs a size for each dynamic dimension, which the op takes only for a ranked tensor result}}
  // expected-error @+1 {{failed to bufferize op}}
  %0 = bufferization.alloc_tensor() copy(%t)
    : !test.test_tensor<[-9223372036854775808], f64>
  return %0 : !test.test_tensor<[-9223372036854775808], f64>
}
