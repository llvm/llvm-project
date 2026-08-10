// RUN: mlir-opt --transform-interpreter -canonicalize -split-input-file --verify-diagnostics %s | FileCheck %s

//     CHECK-LABEL: pad_fill
//           CHECK:   linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : tensor<8x25xf32>) -> tensor<8x25xf32>
func.func @pad_fill(%value: f32, %output: tensor<24x25xf32>) -> tensor<24x25xf32>
{
  %0 = linalg.fill ins(%value : f32) outs(%output : tensor<24x25xf32>) -> tensor<24x25xf32>
  func.return %0 : tensor<24x25xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %fill = transform.structured.match ops{["linalg.fill"]} in %arg1
      : (!transform.any_op) -> !transform.any_op

    // Tile to 5 then pad to 8
    %fill_l1, %loops_l1 = transform.structured.tile_using_for %fill tile_sizes [5]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %fill_padded, %_ = transform.structured.pad_tiling_interface %fill_l1 to padding_sizes [8] {
      padding_values= [#ub.poison, 0.0 : f32]
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.yield
  }
}

// -----

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
      padding_values=[0.0: f32, 0.0 : f32, 0.0 : f32]
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
// CHECK-SAME:      %[[T1:.*]]: tensor<7x11x11xf32>)
  func.func @generic(%arg0: tensor<7x5xf32>, %arg1: tensor<7x11x11xf32>) -> tensor<7x11x11xf32> {

  //  CHECK-DAG: %[[CST:.*]] = arith.constant 0.

  //      CHECK: %[[PAD0:.*]] = tensor.pad %[[T0]] low[0, 0] high[1, 0]
  //      CHECK:   : tensor<7x5xf32> to tensor<8x5xf32>
  //      CHECK: %[[PAD1:.*]] = tensor.pad %[[T1]] low[0, 0, 0] high[1, 3, 1] {
  //      CHECK:   : tensor<7x11x11xf32> to tensor<8x14x12xf32>
  // CHECK-NEXT: linalg.generic
  //      CHECK: tensor.extract_slice %{{.*}}[0, 0, 0] [7, 11, 11] [1, 1, 1] : tensor<8x14x12xf32> to tensor<7x11x11xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0 : tensor<7x5xf32>) outs(%arg1 : tensor<7x11x11xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<7x11x11xf32>
    return %0 : tensor<7x11x11xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %padded, %pad = transform.structured.pad_tiling_interface %0 to padding_sizes [8, 0, 14] {
        padding_values = [0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32]
      } : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.yield
    }
  }
}


// -----


// CHECK-DAG: #[[$MAP0:.*]] = affine_map<()[s0] -> (-s0 + 8)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<()[s0] -> (-s0 + 12)>
// CHECK-DAG: #[[$MAP2:.*]] = affine_map<()[s0] -> (s0 + 5)>

#map = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2, d0 + d1)>
module {

// CHECK-LABEL: @generic
// CHECK-SAME:      %[[T0:.*]]: tensor<?x5xf32>,
// CHECK-SAME:      %[[T1:.*]]: tensor<?x11x?xf32>)
  func.func @generic(%arg0: tensor<?x5xf32>, %arg1: tensor<?x11x?xf32>) -> tensor<?x11x?xf32> {

  //  CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  //  CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  //
  //      CHECK: %[[D0_0:.*]] = tensor.dim %{{.*}}, %[[C0]] : tensor<?x5xf32>
  //      CHECK: %[[H0:.*]] = affine.apply #[[$MAP0]]()[%[[D0_0]]]
  //      CHECK: tensor.pad %{{.*}} low[0, 0] high[%[[H0]], 0] {
  //      CHECK:   : tensor<?x5xf32> to tensor<8x5xf32>
  //
  //      CHECK: %[[D0_1:.*]] = tensor.dim %{{.*}}, %[[C0]] : tensor<?x11x?xf32>
  //      CHECK: %[[H1:.*]] = affine.apply #[[$MAP0]]()[%[[D0_1]]]
  //      CHECK: %[[D2_0:.*]] = tensor.dim %{{.*}}, %[[C2]] : tensor<?x11x?xf32>
  //      CHECK: %[[H2:.*]] = affine.apply #[[$MAP1]]()[%[[D2_0]]]
  //      CHECK: tensor.pad %{{.*}} low[0, 0, 0] high[%[[H1]], 3, %[[H2]]] {
  //      CHECK:   : tensor<?x11x?xf32> to tensor<8x14x12xf32>
  //
  //      CHECK: %[[D0_2:.*]] = tensor.dim %{{.*}}, %[[C0]] : tensor<?x5xf32>
  //      CHECK: %[[D2_1:.*]] = affine.apply #[[$MAP2]]()[%[[D0_2]]]
  //      CHECK: linalg.generic {{.*}} ins(%{{.*}} : tensor<8x5xf32>) outs(%{{.*}} : tensor<8x14x12xf32>) {
  //      CHECK: } -> tensor<8x14x12xf32>
  //      CHECK: tensor.extract_slice %{{.*}}[0, 0, 0] [%[[D0_2]], 11, %[[D2_1]]] [1, 1, 1] : tensor<8x14x12xf32> to tensor<?x11x?xf32>
  //
  %0 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0 : tensor<?x5xf32>) outs(%arg1 : tensor<?x11x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x11x?xf32>
    return %0 : tensor<?x11x?xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %padded, %pad = transform.structured.pad_tiling_interface %0 to padding_sizes [8, 0, 14] {
        padding_values = [0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32]
      } : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.yield
    }
  }
}
