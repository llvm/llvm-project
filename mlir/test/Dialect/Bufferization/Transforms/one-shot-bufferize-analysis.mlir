// RUN: mlir-opt %s -one-shot-bufferize="test-analysis-only" -allow-unregistered-dialect -split-input-file | FileCheck %s

// CHECK-LABEL: func @unknown_op_aliasing(
func.func @unknown_op_aliasing(%f: f32, %f2: f32, %pos: index) -> f32 {
  %0 = tensor.empty() : tensor<10xf32>
  // CHECK: linalg.fill {__inplace_operands_attr__ = ["none", "true"]}
  %1 = linalg.fill ins(%f : f32) outs(%0 : tensor<10xf32>) -> tensor<10xf32>

  // Something must bufferize out-of-place because the op may return an alias
  // of %1.
  // CHECK: "dummy.dummy_op"(%{{.*}}) {__inplace_operands_attr__ = ["false"]}
  %alias = "dummy.dummy_op"(%1) : (tensor<10xf32>) -> (tensor<10xf32>)

  // CHECK: linalg.fill {__inplace_operands_attr__ = ["none", "true"]}
  %2 = linalg.fill ins(%f2 : f32) outs(%1 : tensor<10xf32>) -> tensor<10xf32>
  %3 = tensor.extract %alias[%pos] : tensor<10xf32>
  return %3 : f32
}

// -----

// CHECK-LABEL: func @unknown_op_writing(
func.func @unknown_op_writing(%f: f32, %f2: f32, %pos: index) -> f32 {
  %0 = tensor.empty() : tensor<10xf32>
  // CHECK: linalg.fill {__inplace_operands_attr__ = ["none", "true"]}
  %1 = linalg.fill ins(%f : f32) outs(%0 : tensor<10xf32>) -> tensor<10xf32>

  // The op may bufferize to a memory write, so it must bufferize out-of-place.
  // CHECK: "dummy.dummy_op"(%{{.*}}) {__inplace_operands_attr__ = ["false"]}
  "dummy.dummy_op"(%1) : (tensor<10xf32>) -> ()

  %3 = tensor.extract %1[%pos] : tensor<10xf32>
  return %3 : f32
}
