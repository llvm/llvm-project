// RUN: mlir-opt --transform-interpreter -canonicalize -split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0) -> (-d0 + 24, 5)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0) -> (-d0 + (d0 ceildiv 8) * 8)>

//     CHECK-LABEL: pad_lhs
func.func @pad_lhs(
  %arg0: tensor<24x12xf32>, %arg1: tensor<12x25xf32>, %arg2: tensor<24x25xf32>)
     -> tensor<24x25xf32>
{
  //      CHECK: scf.for %{{.*}} -> (tensor<24x25xf32>)
  //      CHECK:   %[[MIN:.*]] = affine.min #[[$MAP0]](%{{.*}})
  //      CHECK:   %[[H0:.*]] = affine.apply #[[$MAP1]](%[[MIN]])
  //      CHECK:   tensor.pad %{{.*}} low[0, 0] high[%[[H0]], 0]
  //      CHECK:     : tensor<?x12xf32> to tensor<?x12xf32>

  //      CHECK:   %[[H1:.*]] = affine.apply #[[$MAP1]](%[[MIN]])
  //      CHECK:   tensor.pad %{{.*}} low[0, 0] high[%[[H1]], 0]
  //      CHECK:     : tensor<?x25xf32> to tensor<?x25xf32>

  //      CHECK:   linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<?x12xf32>, tensor<12x25xf32>) outs(%{{.*}} : tensor<?x25xf32>) -> tensor<?x25xf32>

  //      CHECK:   tensor.extract_slice %{{.*}}[0, 0] [%{{.*}}, 25] [1, 1]
  //      CHECK:     : tensor<?x25xf32> to tensor<?x25xf32>
  //      CHECK:   tensor.insert_slice %{{.*}} into %{{.*}}[%{{.*}}, 0] [%{{.*}}, 25] [1, 1]
  // CHECK-SAME:     : tensor<?x25xf32> into tensor<24x25xf32>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<24x12xf32>, tensor<12x25xf32>) outs(%arg2 : tensor<24x25xf32>) -> tensor<24x25xf32>
  func.return %0 : tensor<24x25xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %module_op
      : (!transform.any_op) -> !transform.any_op

    // Tile to 5 then pad to 8 (supposedly to better hit vector ops).
    %matmul_l1, %loops_l1 = transform.structured.tile_using_for %matmul tile_sizes [5] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %matmul_padded, %_ = transform.structured.pad_tiling_interface %matmul_l1 to padding_sizes [8] pad_to_multiple_of {
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
// CHECK-SAME:      %[[T1:.*]]: tensor<7x11x12xf32>)
  func.func @generic(%arg0: tensor<7x5xf32>, %arg1: tensor<7x11x12xf32>) -> tensor<7x11x12xf32> {

  //  CHECK-DAG: %[[CST:.*]] = arith.constant 0.

  //      CHECK: %[[PAD0:.*]] = tensor.pad %[[T0]] low[0, 0] high[2, 0]
  //      CHECK:   : tensor<7x5xf32> to tensor<9x5xf32>
  //      CHECK: %[[PAD1:.*]] = tensor.pad %[[T1]] low[0, 0, 0] high[2, 4, 2] {
  //      CHECK:   : tensor<7x11x12xf32> to tensor<9x15x14xf32>
  // CHECK-NEXT: linalg.generic
  //      CHECK: tensor.extract_slice %{{.*}}[0, 0, 0] [7, 11, 12] [1, 1, 1] : tensor<9x15x14xf32> to tensor<7x11x12xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0 : tensor<7x5xf32>) outs(%arg1 : tensor<7x11x12xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<7x11x12xf32>
    return %0 : tensor<7x11x12xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %padded, %pad = transform.structured.pad_tiling_interface %0 to padding_sizes [3, 0, 5] pad_to_multiple_of {
        padding_values = [0.0 : f32, 0.0 : f32, 0.0 : f32]
      } : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.yield
    }
  }
}

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<()[s0, s1] -> (-s1 + (s0 ceildiv 3) * 3)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<()[s0, s1] -> (-s1 + (s0 ceildiv 3) * 3 + 5)>
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
  //      CHECK: %[[D0_1:.*]] = tensor.dim %{{.*}}, %[[C0]] : tensor<?x5xf32>
  //      CHECK: %[[H0:.*]] = affine.apply #[[$MAP0]]()[%[[D0_0]], %[[D0_1]]]
  //      CHECK: tensor.pad %{{.*}} low[0, 0] high[%[[H0]], 0] {
  //      CHECK:   : tensor<?x5xf32> to tensor<?x5xf32>
  //
  //      CHECK: %[[D0_2:.*]] = tensor.dim %{{.*}}, %[[C0]] : tensor<?x11x?xf32>
  //      CHECK: %[[H1:.*]] = affine.apply #[[$MAP0]]()[%[[D0_0]], %[[D0_2]]]
  //      CHECK: %[[D2_0:.*]] = tensor.dim %{{.*}}, %[[C2]] : tensor<?x11x?xf32>
  //      CHECK: %[[H2:.*]] = affine.apply #[[$MAP1]]()[%[[D0_0]], %[[D2_0]]]
  //      CHECK: tensor.pad %{{.*}} low[0, 0, 0] high[%[[H1]], 4, %[[H2]]] {
  //      CHECK:   : tensor<?x11x?xf32> to tensor<?x15x?xf32>
  //
  //      CHECK: %[[D0_3:.*]] = tensor.dim %{{.*}}, %[[C0]] : tensor<?x5xf32>
  //      CHECK: %[[D2_1:.*]] = affine.apply #[[$MAP2]]()[%[[D0_3]]]
  //      CHECK: linalg.generic {{.*}} ins(%{{.*}} : tensor<?x5xf32>) outs(%{{.*}} : tensor<?x15x?xf32>) {
  //      CHECK: } -> tensor<?x15x?xf32>
  //      CHECK: tensor.extract_slice %{{.*}}[0, 0, 0] [%[[D0_3]], 11, %[[D2_1]]] [1, 1, 1] : tensor<?x15x?xf32> to tensor<?x11x?xf32>
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
      %padded, %pad = transform.structured.pad_tiling_interface %0 to padding_sizes [3, 0, 5] pad_to_multiple_of {
        padding_values = [0.0 : f32, 0.0 : f32, 0.0 : f32]
      } : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.yield
    }
  }
}

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<()[s0] -> (-s0 + (s0 ceildiv 16) * 16)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<()[s0, s1] -> (-s1 + (s0 ceildiv 16) * 16)>
// CHECK-DAG: #[[$MAP2:.*]] = affine_map<()[s0] -> ((s0 ceildiv 16) * 16)>
//     CHECK-LABEL: pad_lhs
func.func @pad_lhs(
  %arg0: tensor<24x?xf32>, %arg1: tensor<?x25xf32>, %arg2: tensor<24x25xf32>)
     -> tensor<24x25xf32>
{
  //      CHECK: %[[D0_0:.*]] = tensor.dim
  //      CHECK: %[[H0:.*]] = affine.apply #[[$MAP0]]()[%[[D0_0]]]
  //      CHECK: tensor.pad %{{.*}} low[0, 0] high[0, %[[H0]]]
  //      CHECK:   : tensor<24x?xf32> to tensor<24x?xf32>

  //      CHECK: %[[D0_2:.*]] = tensor.dim
  //      CHECK: %[[H1:.*]] = affine.apply #[[$MAP1]]()[%[[D0_0]], %[[D0_2]]]
  //      CHECK: tensor.pad %{{.*}} low[0, 0] high[%[[H1]], 0]
  //      CHECK:   : tensor<?x25xf32> to tensor<?x25xf32>
  //      CHECK: scf.for %{{.*}} -> (tensor<24x25xf32>)

  //      CHECK:    linalg.matmul ins(%{{.*}}, %{{.*}}: tensor<8x16xf32>, tensor<16x25xf32>) outs(%{{.*}} : tensor<8x25xf32>) -> tensor<8x25xf32>

  //      CHECK:   tensor.insert_slice %{{.*}} into %{{.*}}[%{{.*}}, 0] [8, 25] [1, 1]
  // CHECK-SAME:     : tensor<8x25xf32> into tensor<24x25xf32>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<24x?xf32>, tensor<?x25xf32>) outs(%arg2 : tensor<24x25xf32>) -> tensor<24x25xf32>
  func.return %0 : tensor<24x25xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %module_op
      : (!transform.any_op) -> !transform.any_op

    // Pad then tile should produce static shapes.
    %matmul_padded, %_ = transform.structured.pad_tiling_interface %matmul to padding_sizes [8, 0, 16] pad_to_multiple_of {
      padding_values=[0.0: f32, 0.0 : f32, 0.0 : f32]
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %m, %l0, %l1 = transform.structured.tile_using_for %matmul_padded tile_sizes [8, 0, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    %func = transform.structured.match ops{["func.func"]} in %module_op
      : (!transform.any_op) -> !transform.any_op
    %func2 = transform.apply_registered_pass "resolve-shaped-type-result-dims" to %func
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func2 {
      transform.apply_patterns.canonicalization
    } {apply_cse} : !transform.any_op
    %minmax = transform.structured.match ops{["affine.min", "affine.max"]} in %module_op
      : (!transform.any_op) -> !transform.any_op
    transform.affine.simplify_min_max_affine_ops %minmax : !transform.any_op
    transform.apply_patterns to %func2 {
      transform.apply_patterns.canonicalization
    } {apply_cse} : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 16)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0) -> (-d0 + 16)>

//     CHECK-LABEL: pad_lhs
func.func @pad_lhs(
  %arg0: tensor<24x?xf32>, %arg1: tensor<?x25xf32>, %arg2: tensor<24x25xf32>)
     -> tensor<24x25xf32>
{
  //      CHECK: scf.for %{{.*}} -> (tensor<24x25xf32>)
  //      CHECK:   %[[MIN:.*]] = affine.min #[[$MAP0]](%{{.*}})
  //      CHECK:   %[[H0:.*]] = affine.apply #[[$MAP1]](%[[MIN]])
  //      CHECK:   tensor.pad %{{.*}} low[0, 0] high[0, %[[H0]]]
  //      CHECK:     : tensor<8x?xf32> to tensor<8x16xf32>

  //      CHECK:   %[[H1:.*]] = affine.apply #[[$MAP1]](%[[MIN]])
  //      CHECK:   tensor.pad %{{.*}} low[0, 0] high[%[[H1]], 0]
  //      CHECK:     : tensor<?x25xf32> to tensor<16x25xf32>

  //      CHECK:   linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<8x16xf32>, tensor<16x25xf32>) outs(%{{.*}} : tensor<8x25xf32>) -> tensor<8x25xf32>

  //      CHECK:   tensor.insert_slice %{{.*}} into %{{.*}}[%{{.*}}, 0] [8, 25] [1, 1]
  // CHECK-SAME:     : tensor<8x25xf32> into tensor<24x25xf32>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<24x?xf32>, tensor<?x25xf32>) outs(%arg2 : tensor<24x25xf32>) -> tensor<24x25xf32>
  func.return %0 : tensor<24x25xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %module_op
      : (!transform.any_op) -> !transform.any_op

    // Tile then pad should produce static shapes.
    %m, %l0, %l1 = transform.structured.tile_using_for %matmul tile_sizes [8, 0, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    %matmul_padded, %_ = transform.structured.pad_tiling_interface %m to padding_sizes [8, 0, 16] pad_to_multiple_of {
      padding_values=[0.0: f32, 0.0 : f32, 0.0 : f32]
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.yield
  }
}

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0) -> (-d0 + 20, 8)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 16)>
// CHECK-DAG: #[[$MAP2:.*]] = affine_map<(d0) -> (-d0 + 8)>
// CHECK-DAG: #[[$MAP3:.*]] = affine_map<(d0) -> (-d0 + 16)>

//     CHECK-LABEL: pad_lhs
func.func @pad_lhs(
  %arg0: tensor<20x?xf32>, %arg1: tensor<?x25xf32>, %arg2: tensor<20x25xf32>)
     -> tensor<20x25xf32>
{
  //      CHECK:   linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<8x16xf32>, tensor<16x25xf32>) outs(%{{.*}} : tensor<8x25xf32>) -> tensor<8x25xf32>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<20x?xf32>, tensor<?x25xf32>) outs(%arg2 : tensor<20x25xf32>) -> tensor<20x25xf32>
  func.return %0 : tensor<20x25xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %module_op
      : (!transform.any_op) -> !transform.any_op

    // Tile then pad should produce static shapes.
    %m, %l0, %l1 = transform.structured.tile_using_for %matmul tile_sizes [8, 0, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    %matmul_padded, %_ = transform.structured.pad_tiling_interface %m to padding_sizes [8, 0, 16] pad_to_multiple_of {
      padding_values=[0.0: f32, 0.0 : f32, 0.0 : f32]
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.yield
  }
}

