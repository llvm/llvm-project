// RUN: mlir-opt %s -test-linalg-elementwise-fusion-patterns=blacklist-ops-for-reduction -split-input-file | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (0, d0)>
#map2 = affine_map<(d0) -> (0)>
func.func @consumer_with_reduction_blacklist(%arg0: tensor<1x10xf32>,
                              %arg1: tensor<1x10xf32>,
                              %arg2: tensor<1xf32>) -> tensor<1xf32> {
  %init = tensor.empty() : tensor<1x10xf32>
  %0 = linalg.generic
    {indexing_maps = [#map0, #map0, #map0],
     iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1 : tensor<1x10xf32>, tensor<1x10xf32>)
    outs(%init : tensor<1x10xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %2 = arith.addf %arg3, %arg4 : f32
    linalg.yield %2 : f32
  } -> tensor<1x10xf32>
  %1 = linalg.generic
    {indexing_maps = [#map1, #map2],
     iterator_types = ["reduction"]}
    ins(%0 : tensor<1x10xf32>)
    outs(%arg2 : tensor<1xf32>)  {
  ^bb0(%arg3: f32, %arg4: f32):
    %2 = arith.addf %arg3, %arg4 : f32
    linalg.yield %2 : f32
  } -> tensor<1xf32>
  return %1 : tensor<1xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0) -> (0, d0)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0) -> (0)>
//      CHECK: func @consumer_with_reduction_blacklist(%[[ARG0:.+]]: tensor<1x10xf32>, %[[ARG1:.+]]: tensor<1x10xf32>, %[[ARG2:.+]]: tensor<1xf32>)
//      CHECK:   %[[RES0:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP0]], #[[MAP0]], #[[MAP0]]]
// CHECK-SAME:     iterator_types = ["parallel", "parallel"]
// CHECK-SAME:     ins(%[[ARG0]], %[[ARG1]] : tensor<1x10xf32>, tensor<1x10xf32>)
//      CHECK:   ^{{.+}}(%[[T0:.+]]: f32, %[[T1:.+]]: f32, %[[T2:.+]]: f32)
//      CHECK:     %[[T3:.+]] = arith.addf %[[T0]], %[[T1]] : f32
//      CHECK:     linalg.yield %[[T3]]
//      CHECK:   %[[RES1:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP1]], #[[MAP2]]]
// CHECK-SAME:     iterator_types = ["reduction"]
// CHECK-SAME:     ins(%[[RES0]] : tensor<1x10xf32>)
//      CHECK:   ^{{.+}}(%[[T0:.+]]: f32, %[[T1:.+]]: f32)
//      CHECK:     %[[T2:.+]] = arith.addf %[[T0]], %[[T1]] : f32
//      CHECK:     linalg.yield %[[T2]]
//      CHECK:   return %[[RES1]]

