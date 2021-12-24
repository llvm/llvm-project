// RUN: mlir-opt %s -finalizing-bufferize -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func @eliminate_materializations(
// CHECK-SAME:                                     %[[ARG:.*]]: memref<f32>) -> memref<f32> {
// CHECK:           return %[[ARG]] : memref<f32>
func @eliminate_materializations(%arg0: memref<f32>) -> memref<f32> {
  %0 = bufferization.to_tensor %arg0 : memref<f32>
  %1 = bufferization.to_memref %0 : memref<f32>
  return %1 : memref<f32>
}

// -----

func @unable_to_convert_lone_buffer_cast() -> memref<f32> {
  // expected-error @+1 {{failed to legalize operation 'test.source'}}
  %0 = "test.source"() : () -> tensor<f32>
  %1 = bufferization.to_memref %0 : memref<f32>
  return %1 : memref<f32>
}

// -----

func @unable_to_convert_lone_tensor_load(%arg0: memref<f32>) {
  %0 = bufferization.to_tensor %arg0 : memref<f32>
  // expected-error @+1 {{failed to legalize operation 'test.sink'}}
  "test.sink"(%0) : (tensor<f32>) -> ()
  return
}
