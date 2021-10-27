// RUN: mlir-opt %s -func-bufferize -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func @identity(
// CHECK-SAME:                   %[[ARG:.*]]: memref<f32>) -> memref<f32> {
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
// CHECK:           call @sink(%[[ARG]]) : (memref<f32>) -> ()
// CHECK:           return
func private @sink(tensor<f32>)
func @call_sink(%arg0: tensor<f32>) {
  call @sink(%arg0) : (tensor<f32>) -> ()
  return
}

// CHECK-LABEL:   func @unconverted_op_in_body() -> memref<f32> {
// CHECK:           %[[TENSOR:.*]] = "test.source"() : () -> tensor<f32>
// CHECK:           %[[MEMREF:.*]] = memref.buffer_cast %[[TENSOR]] : memref<f32>
// CHECK:           return %[[MEMREF]] : memref<f32>
func @unconverted_op_in_body() -> tensor<f32> {
  %0 = "test.source"() : () -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// Because this pass updates block arguments, it needs to also atomically
// update all terminators and issue an error if that is not possible.
func @unable_to_update_terminator(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = arith.constant true
    cond_br %0, ^bb1(%arg0: tensor<f32>), ^bb2(%arg0: tensor<f32>)
  ^bb1(%bbarg0: tensor<f32>):
    // expected-error @+1 {{failed to legalize operation 'test.terminator'}}
    "test.terminator"() : () -> ()
  ^bb2(%bbarg1: tensor<f32>):
    return %bbarg1 : tensor<f32>
}

// -----

// There was a bug in func-bufferize pass which caused terminators without
// ReturnLike and BranchOpInterface traits (e.g. scf.condition) to always
// fail to legalize even if bufferization doesn't needed.
// Check the pass succedeed.
// CHECK: bufferize_while
// CHECK: scf.while
// CHECK: scf.condition
func @bufferize_while(%arg0: i64, %arg1: i64) -> i64 {
  %c2_i64 = arith.constant 2 : i64
  %0:2 = scf.while (%arg2 = %arg0) : (i64) -> (i64, i64) {
    %1 = arith.cmpi slt, %arg2, %arg1 : i64
    scf.condition(%1) %arg2, %arg2 : i64, i64
  } do {
  ^bb0(%arg2: i64, %arg3: i64):
    %1 = arith.muli %arg3, %c2_i64 : i64
    scf.yield %1 : i64
  }
  return %0#1 : i64
}
