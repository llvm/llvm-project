// RUN: mlir-opt %s -func-bufferize -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func @identity(
// CHECK-SAME:                   %[[ARG:.*]]: memref<f32>) -> memref<f32> {
// CHECK:           %[[TENSOR:.*]] = tensor_load %[[ARG]] : memref<f32>
// CHECK:           %[[MEMREF:.*]] = tensor_to_memref %[[TENSOR]] : memref<f32>
// CHECK:           return %[[MEMREF]] : memref<f32>
func @identity(%arg0: tensor<f32>) -> tensor<f32> {
  return %arg0 : tensor<f32>
}

// CHECK-LABEL:   func @block_arguments(
// CHECK-SAME:        %[[ARG:.*]]: memref<f32>) -> memref<f32> {
// CHECK:           %[[T1:.*]] = tensor_load %[[ARG]] : memref<f32>
// CHECK:           %[[M1:.*]] = tensor_to_memref %[[T1]] : memref<f32>
// CHECK:           br ^bb1(%[[M1]] : memref<f32>)
// CHECK:         ^bb1(%[[BBARG:.*]]: memref<f32>):
// CHECK:           %[[T2:.*]] = tensor_load %[[BBARG]] : memref<f32>
// CHECK:           %[[M2:.*]] = tensor_to_memref %[[T2]] : memref<f32>
// CHECK:           return %[[M2]] : memref<f32>
func @block_arguments(%arg0: tensor<f32>) -> tensor<f32> {
  br ^bb1(%arg0: tensor<f32>)
^bb1(%bbarg: tensor<f32>):
  return %bbarg : tensor<f32>
}

// CHECK-LABEL:   func private @source() -> memref<f32>
// CHECK-LABEL:   func @call_source() -> memref<f32> {
// CHECK:           %[[RET:.*]] = call @source() : () -> memref<f32>
// CHECK:           return %[[RET]] : memref<f32>
func private @source() -> tensor<f32>
func @call_source() -> tensor<f32> {
  %0 = call @source() : () -> tensor<f32>
  return %0 : tensor<f32>
}
// CHECK-LABEL:   func @call_sink(
// CHECK-SAME:                    %[[ARG:.*]]: memref<f32>) {
// CHECK:           %[[TENSOR:.*]] = tensor_load %[[ARG]] : memref<f32>
// CHECK:           %[[MEMREF:.*]] = tensor_to_memref %[[TENSOR]] : memref<f32>
// CHECK:           call @sink(%[[MEMREF]]) : (memref<f32>) -> ()
// CHECK:           return
func private @sink(tensor<f32>)
func @call_sink(%arg0: tensor<f32>) {
  call @sink(%arg0) : (tensor<f32>) -> ()
  return
}

// CHECK-LABEL:   func @unconverted_op_in_body() -> memref<f32> {
// CHECK:           %[[TENSOR:.*]] = "test.source"() : () -> tensor<f32>
// CHECK:           %[[MEMREF:.*]] = tensor_to_memref %[[TENSOR]] : memref<f32>
// CHECK:           return %[[MEMREF]] : memref<f32>
func @unconverted_op_in_body() -> tensor<f32> {
  %0 = "test.source"() : () -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// Because this pass updates block arguments, it needs to also atomically
// update all terminators and issue an error if that is not possible.
func @unable_to_update_terminator(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = constant true
    cond_br %0, ^bb1(%arg0: tensor<f32>), ^bb2(%arg0: tensor<f32>)
  ^bb1(%bbarg0: tensor<f32>):
    // expected-error @+1 {{failed to legalize operation 'test.terminator'}}
    "test.terminator"() : () -> ()
  ^bb2(%bbarg1: tensor<f32>):
    return %bbarg1 : tensor<f32>
}
