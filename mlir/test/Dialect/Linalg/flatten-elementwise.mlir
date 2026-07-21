// RUN: mlir-opt %s -transform-interpreter -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @fill_memref(
// CHECK-SAME:                  %[[ARG0:.*]]: f32,
// CHECK-SAME:                  %[[ARG1:.*]]: memref<32x7xf32>
// CHECK-NEXT:    %[[FLATTENED:.*]] = memref.collapse_shape %[[ARG1]] {{\[}}[0, 1]]
// CHECK-NEXT:    linalg.fill ins(%[[ARG0]] : f32) outs(%[[FLATTENED]] : memref<224xf32>)
func.func @fill_memref(%cst: f32, %arg: memref<32x7xf32>) {
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
// CHECK-NEXT:    %[[RESULT:.*]] = tensor.expand_shape %[[FLATTENED_RESULT]] {{\[}}[0, 1]] output_shape [32, 7] : tensor<224xf32> into tensor<32x7xf32>
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

// CHECK-LABEL: func.func @broadcast_rank0_tensor(
// CHECK-SAME:                         %[[ARG0:.*]]: tensor<i32>,
// CHECK-SAME:                         %[[ARG1:.*]]: tensor<32x2xi32>
// CHECK-NEXT:    %[[FLATTENED:.*]] = tensor.collapse_shape %[[ARG1]] {{\[}}[0, 1]]
// CHECK-NEXT:    %[[FLATTENED_RESULT:.*]] = linalg.generic {{.*}} ins(%[[ARG0]] : tensor<i32>) outs(%[[FLATTENED]] : tensor<64xi32>)
// CHECK:         %[[RESULT:.*]] = tensor.expand_shape %[[FLATTENED_RESULT]] {{\[}}[0, 1]] output_shape [32, 2] : tensor<64xi32> into tensor<32x2xi32>
#map0 = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @broadcast_rank0_tensor(%arg0: tensor<i32>, %arg1: tensor<32x2xi32>) -> tensor<32x2xi32> {
  %0 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<i32>) outs(%arg1 : tensor<32x2xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
  } -> tensor<32x2xi32>
  return %0 : tensor<32x2xi32>
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

// CHECK-LABEL: func.func @map_memref(
// CHECK-SAME:                 %[[ARG0:[a-zA-Z0-9_]*]]: memref<32x7xf32>
// CHECK-SAME:                 %[[ARG1:[a-zA-Z0-9_]*]]: memref<32x7xf32>
// CHECK-SAME:                 %[[ARG2:[a-zA-Z0-9_]*]]: memref<32x7xf32>
// CHECK-NEXT:    %[[FLATTENED_0:.*]] = memref.collapse_shape %[[ARG0]] {{\[}}[0, 1]]
// CHECK-NEXT:    %[[FLATTENED_1:.*]] = memref.collapse_shape %[[ARG1]] {{\[}}[0, 1]]
// CHECK-NEXT:    %[[FLATTENED_2:.*]] = memref.collapse_shape %[[ARG2]] {{\[}}[0, 1]]
// CHECK-NEXT:    linalg.map { arith.addf } ins(%[[FLATTENED_0]], %[[FLATTENED_1]] : memref<224xf32>, memref<224xf32>) outs(%[[FLATTENED_2]] : memref<224xf32>)
func.func @map_memref(%arg0: memref<32x7xf32>, %arg1: memref<32x7xf32>, %arg2: memref<32x7xf32>) {
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

// CHECK-LABEL: func.func @map_already_flat_memref(
// CHECK-SAME:                 %[[ARG0:[a-zA-Z0-9_]*]]: memref<32xf32>
// CHECK-SAME:                 %[[ARG1:[a-zA-Z0-9_]*]]: memref<32xf32>
// CHECK-SAME:                 %[[ARG2:[a-zA-Z0-9_]*]]: memref<32xf32>
// CHECK-NEXT:    linalg.map { arith.addf } ins(%[[ARG0]], %[[ARG1]] : memref<32xf32>, memref<32xf32>) outs(%[[ARG2]] : memref<32xf32>)
func.func @map_already_flat_memref(%arg0: memref<32xf32>, %arg1: memref<32xf32>, %arg2: memref<32xf32>) {
    linalg.map {arith.addf} ins(%arg0, %arg1: memref<32xf32>, memref<32xf32>) outs(%arg2: memref<32xf32>)
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
// CHECK-LABEL: func.func @elementwise_memref
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
func.func @elementwise_memref( %arg0: memref<32x7xf32>, %arg1: memref<32x7xf32>, %arg2: memref<32x7xf32>) {
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
