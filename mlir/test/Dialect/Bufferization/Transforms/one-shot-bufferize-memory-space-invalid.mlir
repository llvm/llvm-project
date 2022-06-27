// RUN: mlir-opt %s -allow-unregistered-dialect -one-shot-bufferize="must-infer-memory-space" -split-input-file -verify-diagnostics

func.func @alloc_tensor_without_memory_space() -> tensor<10xf32> {
  // expected-error @+2 {{could not infer memory space}}
  // expected-error @+1 {{failed to bufferize op}}
  %0 = bufferization.alloc_tensor() : tensor<10xf32>
  return %0 : tensor<10xf32>
}
