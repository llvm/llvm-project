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
