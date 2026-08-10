// RUN: mlir-opt -transform-interpreter %s --split-input-file --allow-unregistered-dialect -verify-diagnostics | FileCheck %s

// CHECK:  func.func @linalg_copy_to_memref_copy(%[[INPUT:.*]]: memref<128x64xf32>, %[[OUTPUT:.*]]: memref<128x64xf32>) {
// CHECK:    memref.copy %[[INPUT]], %[[OUTPUT]] : memref<128x64xf32> to memref<128x64xf32>
// CHECK:    return
// CHECK:  }

func.func @linalg_copy_to_memref_copy(%input : memref<128x64xf32>, %output : memref<128x64xf32>) {
  linalg.copy ins(%input : memref<128x64xf32>) outs(%output : memref<128x64xf32>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.copy"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.linalg_copy_to_memref %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK:  func.func @linalg_copy_to_memref_copy_strides(%[[INPUT:.*]]: memref<128x32xf32>, %[[OUTPUT:.*]]: memref<128x64xf32>) {
// CHECK:    %[[ALLOC:.*]] = memref.alloc() {alignment = 64 : i64} : memref<128x64xf32>
// CHECK:    %[[SUBVIEW:.*]] = memref.subview %[[ALLOC]][0, 32] [128, 32] [1, 1] : memref<128x64xf32> to memref<128x32xf32, strided<[64, 1], offset: 32>>
// CHECK:    memref.copy %[[INPUT]], %[[SUBVIEW]] : memref<128x32xf32> to memref<128x32xf32, strided<[64, 1], offset: 32>>
// CHECK:    return
// CHECK:  }

func.func @linalg_copy_to_memref_copy_strides(%input : memref<128x32xf32>, %output : memref<128x64xf32>) {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x64xf32>
  %subview = memref.subview %alloc[0, 32] [128, 32] [1, 1] : memref<128x64xf32> to memref<128x32xf32, strided<[64, 1], offset: 32>>
  linalg.copy ins(%input : memref<128x32xf32>) outs(%subview : memref<128x32xf32, strided<[64, 1], offset: 32>>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.copy"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.linalg_copy_to_memref %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @linalg_copy_to_memref_copy_tensors(%input : tensor<128x64xf32>, %output : tensor<128x64xf32>) -> tensor<128x64xf32> {
  // expected-note @below {{target op}}
  %0 = linalg.copy ins(%input : tensor<128x64xf32>) outs(%output : tensor<128x64xf32>) -> tensor<128x64xf32>
  return %0 : tensor<128x64xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.copy"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{cannot transform a linalg.copy on tensors into a memref.copy}}
    %1 = transform.structured.linalg_copy_to_memref %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @linalg_copy_to_memref_copy_different_element(%input : memref<128x64xf32>, %output : memref<128x64xf64>) {
  // expected-note @below {{target op}}
  linalg.copy ins(%input : memref<128x64xf32>) outs(%output : memref<128x64xf64>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.copy"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{cannot transform a linalg.copy with different source and destination element types}}
    %1 = transform.structured.linalg_copy_to_memref %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @linalg_copy_to_memref_copy_scalar(%input : f64, %output : memref<128x64xf64>) {
  // expected-note @below {{target op}}
  linalg.copy ins(%input : f64) outs(%output : memref<128x64xf64>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.copy"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{cannot transform a linalg.copy which input has no shape}}
    %1 = transform.structured.linalg_copy_to_memref %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
