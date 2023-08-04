// RUN: mlir-opt %s -one-shot-bufferize="allow-return-allocs bufferize-function-boundaries test-analysis-only" -split-input-file | FileCheck %s

// CHECK-LABEL: @elementwise_no_conflict
func.func @elementwise_no_conflict(%a: tensor<5xf32>,
                                   %b: tensor<5xf32>) -> tensor<5xf32> {
  // CHECK: linalg.elemwise_binary
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "true"], fun = #linalg.binary_fn<add>}
  %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
      ins(%a, %b : tensor<5xf32>, tensor<5xf32>)
      outs(%a : tensor<5xf32>) -> tensor<5xf32>
  return %0 : tensor<5xf32>
}

// -----

// CHECK-LABEL: @elementwise_no_conflict_2
func.func @elementwise_no_conflict_2(%a: tensor<5xf32>) -> tensor<5xf32> {
  // CHECK: linalg.elemwise_binary
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "true"], fun = #linalg.binary_fn<add>}
  %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
      ins(%a, %a : tensor<5xf32>, tensor<5xf32>)
      outs(%a : tensor<5xf32>) -> tensor<5xf32>
  return %0 : tensor<5xf32>
}

// -----

// CHECK-LABEL: @elementwise_no_conflict_3
func.func @elementwise_no_conflict_3(%a: tensor<5xf32>) -> tensor<5xf32> {
  %c0f = arith.constant 1.0 : f32
  // CHECK: linalg.elemwise_binary
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "none", "true"], fun = #linalg.binary_fn<add>}
  %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
      ins(%a, %c0f : tensor<5xf32>, f32)
      outs(%a : tensor<5xf32>) -> tensor<5xf32>
  return %0 : tensor<5xf32>
}

// -----

func.func @not_elementwise(%a: tensor<5x6xf32>) -> tensor<5x6xf32> {
  %cst = arith.constant 5.0 : f32
  // CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["false"]}
  %b = tensor.extract_slice %a[0, 0] [1, 6] [1, 1]
      : tensor<5x6xf32> to tensor<6xf32>
  // CHECK: linalg.generic
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true"]}
  %0 = linalg.generic 
    { iterator_types = ["parallel", "parallel"],
      indexing_maps = [ affine_map<(d0, d1) -> (d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>] }
    ins(%b: tensor<6xf32>) outs(%a: tensor<5x6xf32>) {
    ^bb0(%arg0: f32, %arg1: f32):
      %r = arith.addf %arg0, %arg1 : f32
      linalg.yield %r : f32
    } -> tensor<5x6xf32>
  return %0 : tensor<5x6xf32>
}
