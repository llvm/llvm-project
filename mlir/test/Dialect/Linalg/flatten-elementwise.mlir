// RUN: mlir-opt %s -transform-interpreter -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @fill(
// CHECK-SAME:                  %[[ARG0:.*]]: f32,
// CHECK-SAME:                  %[[ARG1:.*]]: memref<32x7xf32>
// CHECK-NEXT:    %[[FLATTENED:.*]] = memref.collapse_shape %[[ARG1]] {{\[}}[0, 1]]
// CHECK-NEXT:    linalg.fill ins(%[[ARG0]] : f32) outs(%[[FLATTENED]] : memref<224xf32>)
func.func @fill(%cst: f32, %arg: memref<32x7xf32>) {
    linalg.fill ins(%cst: f32) outs(%arg: memref<32x7xf32>)
    return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match interface{LinalgOp} in %arg1 : (!transform.any_op) -> !transform.any_op
    %flattened = transform.structured.flatten_elementwise %0
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @fill_tensor(
// CHECK-SAME:                         %[[ARG0:.*]]: f32,
// CHECK-SAME:                         %[[ARG1:.*]]: tensor<32x7xf32>
// CHECK-NEXT:    %[[FLATTENED:.*]] = tensor.collapse_shape %[[ARG1]] {{\[}}[0, 1]]
// CHECK-NEXT:    %[[FLATTENED_RESULT:.*]] = linalg.fill ins(%[[ARG0]] : f32) outs(%[[FLATTENED]] : tensor<224xf32>)
// CHECK-NEXT:    %[[RESULT:.*]] = tensor.expand_shape %[[FLATTENED_RESULT]] {{\[}}[0, 1]]
func.func @fill_tensor(%cst: f32, %arg: tensor<32x7xf32>) -> tensor<32x7xf32> {
    %0 = linalg.fill ins(%cst: f32) outs(%arg: tensor<32x7xf32>) ->  tensor<32x7xf32>
    return %0 :  tensor<32x7xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match interface{LinalgOp} in %arg1 : (!transform.any_op) -> !transform.any_op
    %flattened = transform.structured.flatten_elementwise %0
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @map(
// CHECK-SAME:                 %[[ARG0:[a-zA-Z0-9_]*]]: memref<32x7xf32>
// CHECK-SAME:                 %[[ARG1:[a-zA-Z0-9_]*]]: memref<32x7xf32>
// CHECK-SAME:                 %[[ARG2:[a-zA-Z0-9_]*]]: memref<32x7xf32>
// CHECK-NEXT:    %[[FLATTENED_0:.*]] = memref.collapse_shape %[[ARG0]] {{\[}}[0, 1]]
// CHECK-NEXT:    %[[FLATTENED_1:.*]] = memref.collapse_shape %[[ARG1]] {{\[}}[0, 1]]
// CHECK-NEXT:    %[[FLATTENED_2:.*]] = memref.collapse_shape %[[ARG2]] {{\[}}[0, 1]]
// CHECK-NEXT:    linalg.map { arith.addf } ins(%[[FLATTENED_0]], %[[FLATTENED_1]] : memref<224xf32>, memref<224xf32>) outs(%[[FLATTENED_2]] : memref<224xf32>)
func.func @map(%arg0: memref<32x7xf32>, %arg1: memref<32x7xf32>, %arg2: memref<32x7xf32>) {
    linalg.map {arith.addf} ins(%arg0, %arg1: memref<32x7xf32>, memref<32x7xf32>) outs(%arg2: memref<32x7xf32>)
    return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match interface{LinalgOp} in %arg1 : (!transform.any_op) -> !transform.any_op
    %flattened = transform.structured.flatten_elementwise %0
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @generic
// CHECK-SAME:                 %[[ARG0:[a-zA-Z0-9_]*]]: memref<32x7xf32>
// CHECK-SAME:                 %[[ARG1:[a-zA-Z0-9_]*]]: memref<32x7xf32>
// CHECK-SAME:                 %[[ARG2:[a-zA-Z0-9_]*]]: memref<32x7xf32>
// CHECK-NEXT:    %[[FLATTENED_0:.*]] = memref.collapse_shape %[[ARG0]] {{\[}}[0, 1]]
// CHECK-NEXT:    %[[FLATTENED_1:.*]] = memref.collapse_shape %[[ARG1]] {{\[}}[0, 1]]
// CHECK-NEXT:    %[[FLATTENED_2:.*]] = memref.collapse_shape %[[ARG2]] {{\[}}[0, 1]]
// CHECK-NEXT:    linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]], #[[$MAP0]]], iterator_types = ["parallel"]} ins(%[[FLATTENED_0]], %[[FLATTENED_1]] : memref<224xf32>, memref<224xf32>) outs(%[[FLATTENED_2]] : memref<224xf32>)
// CHECK-NEXT:       ^bb0(%[[A:.*]]: f32, %[[B:.*]]: f32, %[[C:.*]]: f32)
// CHECK-NEXT:         %[[SUM:.*]] = arith.addf %[[A]], %[[B]]
// CHECK-NEXT:         linalg.yield %[[SUM]]
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @generic( %arg0: memref<32x7xf32>, %arg1: memref<32x7xf32>, %arg2: memref<32x7xf32>) {
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1: memref<32x7xf32>, memref<32x7xf32>) outs(%arg2: memref<32x7xf32>) {
        ^bb0(%a: f32, %b: f32, %c: f32):
            %0 = arith.addf %a, %b : f32
            linalg.yield %0 : f32
    }
    return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match interface{LinalgOp} in %arg1 : (!transform.any_op) -> !transform.any_op
    %flattened = transform.structured.flatten_elementwise %0
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
