// RUN: mlir-opt %s -allow-unregistered-dialect -one-shot-bufferize="must-infer-memory-space" -split-input-file -verify-diagnostics

func.func @alloc_tensor_without_memory_space() -> tensor<10xf32> {
  // expected-error @+2 {{could not infer memory space}}
  // expected-error @+1 {{failed to bufferize op}}
  %0 = bufferization.alloc_tensor() : tensor<10xf32>
  return %0 : tensor<10xf32>
}

// -----

func.func @memory_space_of_unknown_op() -> f32 {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{could not infer memory space}}
  %t = "test.dummy_op"() : () -> (tensor<10xf32>)
  // expected-error @+1 {{failed to bufferize op}}
  %s = tensor.extract %t[%c0] : tensor<10xf32>
  return %s : f32
}
