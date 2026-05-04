// RUN: mlir-opt %s -test-linalg-elementwise-fusion-patterns=fuse-generic-ops-control -split-input-file | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @drop_unused_producer_result(%arg0 : tensor<?x?xf32>,
    %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0:2 = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<?x?xf32>) outs(%arg0, %arg0  : tensor<?x?xf32>, tensor<?x?xf32>) {
    ^bb0(%b0: f32, %b1: f32, %b2: f32):
      %1 = arith.addf %b0, %b0 : f32
      %2 = arith.mulf %b0, %b0 : f32
      linalg.yield %1, %2 : f32, f32
    } -> (tensor<?x?xf32>, tensor<?x?xf32>)
  %3 = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]}
      ins(%0#0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg0  : tensor<?x?xf32>) {
    ^bb0(%b0: f32, %b1: f32, %b2: f32):
      %4 = arith.subf %b0, %b1 : f32
      linalg.yield %4 : f32
    } -> tensor<?x?xf32>
  return %3 : tensor<?x?xf32>
}
// CHECK-LABEL: func @drop_unused_producer_result
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//       CHECK:   %[[FUSED_OP:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
//       CHECK:   return %[[FUSED_OP]]

// -----

#map = affine_map<(d0) -> (d0)>
func.func @handle_unused_operands(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant 1.000000e+00 : f32
  %0:2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} outs(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>) {
  ^bb0(%out: f32, %out_2: f32):
    %1 = linalg.index 0 : index
    %2 = arith.index_cast %1 : index to i64
    %3 = arith.sitofp %2 : i64 to f32
    %4 = arith.divf %3, %cst_0 : f32
    linalg.yield %3, %4 : f32, f32
  } -> (tensor<8xf32>, tensor<8xf32>)
  linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} ins(%0#1 : tensor<8xf32>) {
  ^bb0(%in: f32):
    %2 = arith.cmpf one, %in, %cst_1 : f32
    cf.assert %2, "Side effect op"
    linalg.yield
  }
  func.return %arg1 : tensor<8xf32>
}

// CHECK-LABEL: func @handle_unused_operands
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<8xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: tensor<8xf32>
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<8xf32>
//       CHECK:   %[[FUSED_OP:.+]] = linalg.generic
//  CHECK-SAME:       outs(%[[EMPTY]] :
//   CHECK-NOT:   linalg.generic
