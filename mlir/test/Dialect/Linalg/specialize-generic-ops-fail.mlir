// RUN: mlir-opt %s -split-input-file --linalg-specialize-generic-ops | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d1, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// This test checks that linalg.generic does not get incorrectly specialized to transform or broadcast.
// CHECK-LABEL: @transpose_and_broadcast
// CHECK: linalg.generic
func.func @transpose_and_broadcast(%arg0: tensor<7x8xf32>, %arg1: tensor<8x7x9xf32>) -> tensor<8x7x9xf32> {
  %0 = linalg.generic
        {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%arg0 : tensor<7x8xf32>) outs(%arg1 : tensor<8x7x9xf32>) {
        ^bb0(%in: f32, %out: f32):
           linalg.yield %in : f32
  } -> tensor<8x7x9xf32>
  return %0 : tensor<8x7x9xf32>
}
