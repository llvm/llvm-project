// RUN: mlir-opt -test-linalg-elementwise-fusion-patterns=control-fusion-by-expansion %s -split-input-file | FileCheck %s

func @control_producer_reshape_fusion(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<?x?x?xf32> into tensor<?x?xf32>
  %d0 = tensor.dim %0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %0, %c1 : tensor<?x?xf32>
  %init = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%0, %arg1 : tensor<?x?xf32>, tensor<?xf32>)
      outs(%init : tensor<?x?xf32>) {
      ^bb0(%arg2 : f32, %arg3:f32, %arg4 : f32):
        %2 = arith.addf %arg2, %arg3 : f32
        linalg.yield %2 : f32
      } -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d1)>
//      CHECK: func @control_producer_reshape_fusion
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//      CHECK:   %[[RESHAPE:.+]] = tensor.collapse_shape %[[ARG0]]
// CHECK-SAME:       {{\[}}[0, 1], [2]{{\]}} : tensor<?x?x?xf32> into tensor<?x?xf32>
//      CHECK:   %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:       indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP0]]]
// CHECK-SAME:       ins(%[[RESHAPE]], %[[ARG1]] : tensor<?x?xf32>, tensor<?xf32>)
//      CHECK:   return %[[RESULT]]

// -----

func @control_consumer_reshape_fusion(%arg0 : tensor<1x?x?xf32>, %arg1 : tensor<1x?x?xf32>) -> tensor<1x?x?xf32> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %arg0, %c1 : tensor<1x?x?xf32>
  %d1 = tensor.dim %arg1, %c2 : tensor<1x?x?xf32>
  %init = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %fill = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      outs(%init : tensor<?x?xf32>) {
      ^bb0(%arg2: f32):
        linalg.yield %cst : f32
      } -> tensor<?x?xf32>
  %0 = tensor.expand_shape %fill [[0, 1], [2]] : tensor<?x?xf32> into tensor<1x?x?xf32>
  %1 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<1x?x?xf32>, tensor<1x?x?xf32>)
      outs(%0 : tensor<1x?x?xf32>) -> tensor<1x?x?xf32>
  return %1 : tensor<1x?x?xf32>
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)
//      CHECK: func @control_consumer_reshape_fusion
//      CHECK:   %[[FILL:.+]] = linalg.generic
// CHECK-SAME:       indexing_maps = [#[[MAP]]]
// CHECK-SAME:       outs(%{{.+}} : tensor<1x?x?xf32>)
//      CHECK:   linalg.batch_matmul
// CHECK-SAME:       outs(%[[FILL]] : tensor<1x?x?xf32>)
