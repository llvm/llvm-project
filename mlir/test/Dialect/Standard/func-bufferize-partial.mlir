// RUN: mlir-opt %s -func-bufferize -split-input-file -verify-diagnostics | FileCheck %s

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

// CHECK-LABEL: func @partial()
// CHECK-SAME: memref<f32>
func @partial() -> tensor<f32> {
  // CHECK-NEXT: %[[SRC:.*]] = "test.source"() : () -> tensor<f32>
  // CHECK-NEXT: %[[MEM:.*]] = tensor_to_memref %[[SRC]] : memref<f32>
  %0 = "test.source"() : () -> tensor<f32>
  // CHECK-NEXT: return %[[MEM]] : memref<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: func @region_op
// CHECK-SAME: (%[[ARG0:.*]]: i1) -> memref<f32>
func @region_op(%arg0: i1) -> tensor<f32> {
  // CHECK-NEXT: %[[IF:.*]] = scf.if %[[ARG0]] -> (tensor<f32>)
  %0 = scf.if %arg0 -> (tensor<f32>) {
    // CHECK-NEXT: %[[SRC:.*]] = "test.source"() : () -> tensor<f32>
    %1 = "test.source"() : () -> tensor<f32>
    // CHECK-NEXT: scf.yield %[[SRC]] : tensor<f32>
    scf.yield %1 : tensor<f32>
  // CHECK-NEXT: else
  } else {
    // CHECK-NEXT: %[[OSRC:.*]] = "test.other_source"() : () -> tensor<f32>
    %1 = "test.other_source"() : () -> tensor<f32>
    // CHECK-NEXT: scf.yield %[[OSRC]] : tensor<f32>
    scf.yield %1 : tensor<f32>
  }
  // CHECK: %[[MEM:.*]] = tensor_to_memref %[[IF]] : memref<f32>
  // CHECK: return %[[MEM]] : memref<f32>
  return %0 : tensor<f32>
}

// -----

func @failed_to_legalize(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = constant true
    cond_br %0, ^bb1(%arg0: tensor<f32>), ^bb2(%arg0: tensor<f32>)
  ^bb1(%bbarg0: tensor<f32>):
    // expected-error @+1 {{failed to legalize operation 'test.terminator'}}
    "test.terminator"() : () -> ()
  ^bb2(%bbarg1: tensor<f32>):
    return %bbarg1 : tensor<f32>
}
