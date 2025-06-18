// RUN: mlir-opt --transform-interpreter -canonicalize -split-input-file --verify-diagnostics %s | FileCheck %s

//     CHECK-LABEL: pad_lhs
func.func @pad_lhs(
  %arg0: tensor<24x12xf32>, %arg1: tensor<12x25xf32>, %arg2: tensor<24x25xf32>)
     -> tensor<24x25xf32>
{
  //      CHECK: scf.for %{{.*}} -> (tensor<24x25xf32>)
  //      CHECK:   tensor.pad %{{.*}} 
  //      CHECK:     : tensor<?x12xf32> to tensor<8x12xf32>
  //      CHECK:   tensor.pad %{{.*}} 
  //      CHECK:     : tensor<?x25xf32> to tensor<8x25xf32>
  //      CHECK:   linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<8x12xf32>, tensor<12x25xf32>) outs(%{{.*}} : tensor<8x25xf32>) -> tensor<8x25xf32>
  //      CHECK:   tensor.extract_slice %{{.*}}[0, 0] [%{{.*}}, 25] [1, 1]
  //      CHECK:     : tensor<8x25xf32> to tensor<?x25xf32>
  //      CHECK:   tensor.insert_slice %{{.*}} into %{{.*}}[%{{.*}}, 0] [%{{.*}}, 25] [1, 1]
  // CHECK-SAME:     : tensor<?x25xf32> into tensor<24x25xf32>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<24x12xf32>, tensor<12x25xf32>) outs(%arg2 : tensor<24x25xf32>) -> tensor<24x25xf32>
  func.return %0 : tensor<24x25xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1
      : (!transform.any_op) -> !transform.any_op

    // Tile to 5 then pad to 8 (supposedly to better hit vector ops).
    %matmul_l1, %loops_l1 = transform.structured.tile_using_for %matmul tile_sizes [5] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %matmul_padded, %_ = transform.structured.pad_tiling_interface %matmul_l1 to padding_sizes [8] {
      padding_values=[0.0: f32, 0.0 : f32, 0.0 : f32],
      padding_dimensions=[0]
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2, d0 + d1)>
module {

// CHECK-LABEL: @generic
// CHECK-SAME:      %[[T0:.*]]: tensor<7x5xf32>,
// CHECK-SAME:      %[[T1:.*]]: tensor<7x11x12xf32>)
  func.func @generic(%arg0: tensor<7x5xf32>, %arg1: tensor<7x11x12xf32>) -> tensor<7x11x12xf32> {

  //  CHECK-DAG: %[[CST:.*]] = arith.constant 0.

  //      CHECK: %[[PAD0:.*]] = tensor.pad %[[T0]] low[0, 0] high[1, 0]
  //      CHECK:   : tensor<7x5xf32> to tensor<8x5xf32>
  //      CHECK: %[[PAD1:.*]] = tensor.pad %[[T1]] low[0, 0, 0] high[1, 3, 1] {
  //      CHECK:   : tensor<7x11x12xf32> to tensor<8x14x13xf32>
  // CHECK-NEXT: linalg.generic
  //      CHECK: tensor.extract_slice %{{.*}}[0, 0, 0] [7, 11, 12] [1, 1, 1] : tensor<8x14x13xf32> to tensor<7x11x12xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0 : tensor<7x5xf32>) outs(%arg1 : tensor<7x11x12xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<7x11x12xf32>
    return %0 : tensor<7x11x12xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %padded, %pad = transform.structured.pad_tiling_interface %0 to padding_sizes [8, 14] {
        padding_dimensions = [0, 2], 
        padding_values = [0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32]
      } : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.yield 
    }
  }
}
