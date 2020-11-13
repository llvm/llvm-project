// RUN: mlir-opt %s -func-bufferize -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func @identity(
// CHECK-SAME:        %[[ARG:.*]]: memref<f32>) -> memref<f32> {
// CHECK:           return %[[ARG]] : memref<f32>
func @identity(%arg0: tensor<f32>) -> tensor<f32> {
  return %arg0 : tensor<f32>
}

// CHECK-LABEL:   func @block_arguments(
// CHECK-SAME:        %[[ARG:.*]]: memref<f32>) -> memref<f32> {
// CHECK:           br ^bb1(%[[ARG]] : memref<f32>)
// CHECK:         ^bb1(%[[BBARG:.*]]: memref<f32>):
// CHECK:           return %[[BBARG]] : memref<f32>
func @block_arguments(%arg0: tensor<f32>) -> tensor<f32> {
  br ^bb1(%arg0: tensor<f32>)
^bb1(%bbarg: tensor<f32>):
  return %bbarg : tensor<f32>
}

// CHECK-LABEL:   func @eliminate_target_materialization(
// CHECK-SAME:        %[[ARG:.*]]: memref<f32>) -> memref<f32> {
// CHECK:           return %[[ARG]] : memref<f32>
func @eliminate_target_materialization(%arg0: tensor<f32>) -> memref<f32> {
  %0 = tensor_to_memref %arg0 : memref<f32>
  return %0 : memref<f32>
}

// CHECK-LABEL:   func @eliminate_source_materialization(
// CHECK-SAME:        %[[ARG:.*]]: memref<f32>) -> memref<f32> {
// CHECK:           return %[[ARG]] : memref<f32>
func @eliminate_source_materialization(%arg0: memref<f32>) -> tensor<f32> {
  %0 = tensor_load %arg0 : memref<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL:   func @source() -> memref<f32>
// CHECK-LABEL:   func @call_source() -> memref<f32> {
// CHECK:           %[[RET:.*]] = call @source() : () -> memref<f32>
// CHECK:           return %[[RET]] : memref<f32>
func @source() -> tensor<f32>
func @call_source() -> tensor<f32> {
  %0 = call @source() : () -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL:   func @sink(memref<f32>)
// CHECK-LABEL:   func @call_sink(
// CHECK-SAME:        %[[ARG:.*]]: memref<f32>) {
// CHECK:           call @sink(%[[ARG]]) : (memref<f32>) -> ()
// CHECK:           return
func @sink(tensor<f32>)
func @call_sink(%arg0: tensor<f32>) {
  call @sink(%arg0) : (tensor<f32>) -> ()
  return
}

// -----

func @failed_to_legalize() -> tensor<f32> {
  // expected-error @+1 {{failed to legalize operation 'test.source'}}
  %0 = "test.source"() : () -> (tensor<f32>)
  return %0 : tensor<f32>
}
