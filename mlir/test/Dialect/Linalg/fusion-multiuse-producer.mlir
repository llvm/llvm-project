// RUN: mlir-opt -test-linalg-elementwise-fusion-patterns=fuse-multiuse-producer -split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @multi_use_producer(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : tensor<?x?xf32>, %arg3 : tensor<?x?xf32>, %arg4 : tensor<?x?xf32>)
    -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) {
  %0:2 = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<?x?xf32>)
      outs(%arg1, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>) {
  ^bb0(%b0: f32, %b1 : f32, %b2 : f32):
    %1 = arith.addf %b0, %b1 : f32
    linalg.yield %1, %1 : f32, f32
  } -> (tensor<?x?xf32>, tensor<?x?xf32>)
  %2 = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]}
      ins(%0#1, %arg3 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg4 : tensor<?x?xf32>) {
  ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
    %3 = arith.mulf %b0, %b1 : f32
    linalg.yield %3 : f32
  } -> tensor<?x?xf32>
  return %0#0, %0#1, %2 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
}
//      CHECK: func @multi_use_producer(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG4:[a-zA-Z0-9]+]]: tensor<?x?xf32>)
//      CHECK:   %[[RESULT:.+]]:3 = linalg.generic
//      CHECK:   return %[[RESULT]]#0, %[[RESULT]]#1, %[[RESULT]]#2
