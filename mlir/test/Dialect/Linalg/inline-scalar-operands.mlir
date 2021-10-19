// RUN: mlir-opt %s -linalg-inline-scalar-operands -split-input-file | FileCheck %s

// CHECK: #[[MAP:.*]] = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0) -> ()>

// CHECK: func @inline_zerod(%[[ARG:.*]]: tensor<4xf32>, %[[SCALAR:.*]]: tensor<f32>)
func @inline_zerod(%arg0: tensor<4xf32>, %scalar: tensor<f32>) -> tensor<4xf32> {
    %0 = linalg.init_tensor [4] : tensor<4xf32>
    // CHECK: linalg.generic {indexing_maps = [#[[MAP]], #[[MAP]]],
    // CHECK-SAME: iterator_types = ["parallel"]} ins(%[[ARG]] : tensor<4xf32>)
    %1 = linalg.generic {indexing_maps = [#map2, #map3, #map2],
                         iterator_types = ["parallel"]}
                         ins(%arg0, %scalar : tensor<4xf32>, tensor<f32>)
                         outs(%0 : tensor<4xf32>) {
    // CHECK: ^bb0(%{{.*}}: f32, %{{.*}}: f32)
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):  // no predecessors
      // CHECK: tensor.extract %[[SCALAR]][]
      %2 = arith.divf %arg1, %arg2 : f32
      linalg.yield %2 : f32
    } -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// CHECK: #[[MAP:.*]] = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0) -> (0)>

// CHECK: func @inline_oned(%[[ARG:.*]]: tensor<4xf32>, %[[SCALAR:.*]]: tensor<1xf32>)
func @inline_oned(%arg0: tensor<4xf32>, %scalar: tensor<1xf32>) -> tensor<4xf32> {
    // CHECK: %[[ZERO:.*]] = arith.constant 0 : index
    %0 = linalg.init_tensor [4] : tensor<4xf32>
    // CHECK: linalg.generic {indexing_maps = [#[[MAP]], #[[MAP]]],
    // CHECK-SAME: iterator_types = ["parallel"]} ins(%[[ARG]] : tensor<4xf32>)
    %1 = linalg.generic {indexing_maps = [#map2, #map3, #map2],
                         iterator_types = ["parallel"]}
                         ins(%arg0, %scalar : tensor<4xf32>, tensor<1xf32>)
                         outs(%0 : tensor<4xf32>) {
    // CHECK: ^bb0(%{{.*}}: f32, %{{.*}}: f32)
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):  // no predecessors
      // CHECK: tensor.extract %[[SCALAR]][%[[ZERO]]]
      %2 = arith.divf %arg1, %arg2 : f32
      linalg.yield %2 : f32
    } -> tensor<4xf32>
  return %1 : tensor<4xf32>
}
