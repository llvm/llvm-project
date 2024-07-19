// RUN: mlir-opt -split-input-file -test-tensor-transform-patterns=test-fold-into-pack-and-unpack  %s | FileCheck %s

func.func @fold_unpack_slice(%arg0 : tensor<?x?x8x4xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : index, %arg3 : index) -> tensor<?x?xf32> {
  %0 = tensor.unpack %arg0 inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %arg1
      : tensor<?x?x8x4xf32> -> tensor<?x?xf32>
  %1 = tensor.extract_slice %0[0, 0] [%arg2, %arg3] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
//      CHECK: func @fold_unpack_slice(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?x8x4xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: index
//      CHECK:   %[[INIT:.+]] = tensor.empty(%[[ARG2]], %[[ARG3]]) : tensor<?x?xf32>
//      CHECK:   %[[UNPACK:.+]] = tensor.unpack %[[ARG0]] inner_dims_pos = [0, 1] inner_tiles = [8, 4]
// CHECK-SAME:       into %[[INIT]]
//      CHECK:   return %[[UNPACK]]

// -----

func.func @nofold_unpack_slice_non_zero_offset(%arg0 : tensor<?x?x8x4xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : index, %arg3 : index, %arg4 : index) -> tensor<?x?xf32> {
  %0 = tensor.unpack %arg0 inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %arg1
      : tensor<?x?x8x4xf32> -> tensor<?x?xf32>
  %1 = tensor.extract_slice %0[0, %arg4] [%arg2, %arg3] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: func @nofold_unpack_slice_non_zero_offset(
//       CHECK:   %[[UNPACK:.+]] = tensor.unpack
//       CHECK:   tensor.extract_slice %[[UNPACK]]

// -----

func.func @nofold_unpack_slice_non_unit_stride(%arg0 : tensor<?x?x8x4xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : index, %arg3 : index, %arg4 : index) -> tensor<?x?xf32> {
  %0 = tensor.unpack %arg0 inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %arg1
      : tensor<?x?x8x4xf32> -> tensor<?x?xf32>
  %1 = tensor.extract_slice %0[0, 0] [%arg2, %arg3] [%arg4, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: func @nofold_unpack_slice_non_unit_stride(
//       CHECK:   %[[UNPACK:.+]] = tensor.unpack
//       CHECK:   tensor.extract_slice %[[UNPACK]]

// -----

func.func @nofold_unpack_slice_rank_reduced(%arg0 : tensor<?x?x8x4xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : index, %arg3 : index) -> tensor<f32> {
  %0 = tensor.unpack %arg0 inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %arg1
      : tensor<?x?x8x4xf32> -> tensor<?x?xf32>
  %1 = tensor.extract_slice %0[0, 0] [1, 1] [1, 1] : tensor<?x?xf32> to tensor<f32>
  return %1 : tensor<f32>
}
// CHECK-LABEL: func @nofold_unpack_slice_rank_reduced(
//       CHECK:   %[[UNPACK:.+]] = tensor.unpack
//       CHECK:   tensor.extract_slice %[[UNPACK]]

// -----

func.func @pad_pack(%src: tensor<16641x16xf32>) -> tensor<2082x1x8x32xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %padded = tensor.pad %src low[0, 0] high[15, 0] {
  ^bb0(%arg0: index, %arg1: index):
    tensor.yield %cst : f32
  } : tensor<16641x16xf32> to tensor<16656x16xf32>
  %empty = tensor.empty() : tensor<2082x1x8x32xf32>
  %pack = tensor.pack %padded padding_value(%cst : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 32] into %empty
      : tensor<16656x16xf32> -> tensor<2082x1x8x32xf32>
  return %pack : tensor<2082x1x8x32xf32>
}
// CHECK-LABEL: func.func @pad_pack
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK:         %[[PAD_VAL:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[DEST:.+]] = tensor.empty() : tensor<2082x1x8x32xf32>
// CHECK:         %[[PACK:.+]] = tensor.pack %[[SRC]]
// CHECK-SAME:      padding_value(%[[PAD_VAL]] : f32)
// CHECK-SAME:      inner_dims_pos = [0, 1] inner_tiles = [8, 32] into %[[DEST]]

// -----

func.func @nofold_pad_pack(%src: tensor<16641x16xf32>) -> tensor<2082x1x8x32xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %padded = tensor.pad %src nofold low[0, 0] high[15, 0] {
  ^bb0(%arg0: index, %arg1: index):
    tensor.yield %cst : f32
  } : tensor<16641x16xf32> to tensor<16656x16xf32>
  %empty = tensor.empty() : tensor<2082x1x8x32xf32>
  %pack = tensor.pack %padded padding_value(%cst : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 32] into %empty
      : tensor<16656x16xf32> -> tensor<2082x1x8x32xf32>
  return %pack : tensor<2082x1x8x32xf32>
}
// CHECK-LABEL: func.func @nofold_pad_pack
// CHECK:         tensor.pad
// CHECK:         tensor.pack

// -----

func.func @pad_pack_different_padding_value(%src: tensor<16641x16xf32>) -> tensor<2082x1x8x32xf32> {
  %c0 = arith.constant 0 : index
  %cst0 = arith.constant 0.000000e+00 : f32
  %cst1 = arith.constant 1.000000e+00 : f32
  %padded = tensor.pad %src low[0, 0] high[15, 0] {
  ^bb0(%arg0: index, %arg1: index):
    tensor.yield %cst0 : f32
  } : tensor<16641x16xf32> to tensor<16656x16xf32>
  %empty = tensor.empty() : tensor<2082x1x8x32xf32>
  %pack = tensor.pack %padded padding_value(%cst1 : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 32] into %empty
      : tensor<16656x16xf32> -> tensor<2082x1x8x32xf32>
  return %pack : tensor<2082x1x8x32xf32>
}
// CHECK-LABEL: func.func @pad_pack_different_padding_value
// CHECK:         tensor.pad
// CHECK:         tensor.pack

// -----

func.func @tensor_pack_linalg_transpose_fold(%arg0: tensor<56x57x1x64xf32>) -> tensor<1x57x56x2x32xf32> {
  %0 = tensor.empty() : tensor<56x2x1x57x32xf32>
  %pack = tensor.pack %arg0
    outer_dims_perm = [0, 3, 2, 1]
    inner_dims_pos = [3]
    inner_tiles = [32]
    into %0 : tensor<56x57x1x64xf32> -> tensor<56x2x1x57x32xf32>

  %1 = tensor.empty() : tensor<1x57x56x2x32xf32>
  %transposed = linalg.transpose
    ins(%pack : tensor<56x2x1x57x32xf32>)
    outs(%1 : tensor<1x57x56x2x32xf32>)
    permutation = [2, 3, 0, 1, 4]
  return %transposed : tensor<1x57x56x2x32xf32>
}
//      CHECK: func @tensor_pack_linalg_transpose_fold(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<56x57x1x64xf32>)
//      CHECK:   %[[INIT:.+]] = tensor.empty() : tensor<1x57x56x2x32xf32>
//      CHECK:   %[[PACK:.+]] = tensor.pack %[[ARG0]]
// CHECK-SAME:      outer_dims_perm = [2, 1, 0, 3]
// CHECK-SAME:      inner_dims_pos = [3] inner_tiles = [32]
// CHECK-SAME:       into %[[INIT]]
//      CHECK:   return %[[PACK]]

// -----

func.func @tensor_pack_linalg_transpose_fold_with_padding(%arg0: tensor<56x57x1x55xf32>, %padding: f32) -> tensor<1x57x56x2x32xf32> {
  %0 = tensor.empty() : tensor<56x2x1x57x32xf32>
  %pack = tensor.pack %arg0 padding_value(%padding : f32)
    outer_dims_perm = [0, 3, 2, 1]
    inner_dims_pos = [3]
    inner_tiles = [32]
    into %0 : tensor<56x57x1x55xf32> -> tensor<56x2x1x57x32xf32>

  %1 = tensor.empty() : tensor<1x57x56x2x32xf32>
  %transposed = linalg.transpose
    ins(%pack : tensor<56x2x1x57x32xf32>)
    outs(%1 : tensor<1x57x56x2x32xf32>)
    permutation = [2, 3, 0, 1, 4]
  return %transposed : tensor<1x57x56x2x32xf32>
}
//      CHECK: func @tensor_pack_linalg_transpose_fold_with_padding(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<56x57x1x55xf32>, %[[PADDING:.+]]: f32)
//      CHECK:   %[[INIT:.+]] = tensor.empty() : tensor<1x57x56x2x32xf32>
//      CHECK:   %[[PACK:.+]] = tensor.pack %[[ARG0]] padding_value(%[[PADDING]] : f32)
// CHECK-SAME:      outer_dims_perm = [2, 1, 0, 3]
// CHECK-SAME:      inner_dims_pos = [3] inner_tiles = [32]
// CHECK-SAME:       into %[[INIT]]
//      CHECK:   return %[[PACK]]

// -----

func.func @tensor_pack_linalg_transpose_fold_no_outer_dims_perm(%arg0: tensor<56x57x1x64xf32>) -> tensor<1x2x56x57x32xf32> {
  %0 = tensor.empty() : tensor<56x57x1x2x32xf32>
  %pack = tensor.pack %arg0
    inner_dims_pos = [3]
    inner_tiles = [32]
    into %0 : tensor<56x57x1x64xf32> -> tensor<56x57x1x2x32xf32>

  %1 = tensor.empty() : tensor<1x2x56x57x32xf32>
  %transposed = linalg.transpose
    ins(%pack : tensor<56x57x1x2x32xf32>)
    outs(%1 : tensor<1x2x56x57x32xf32>)
    permutation = [2, 3, 0, 1, 4]
  return %transposed : tensor<1x2x56x57x32xf32>
}
//      CHECK: func @tensor_pack_linalg_transpose_fold_no_outer_dims_perm(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<56x57x1x64xf32>)
//      CHECK:   %[[INIT:.+]] = tensor.empty() : tensor<1x2x56x57x32xf32>
//      CHECK:   %[[PACK:.+]] = tensor.pack %[[ARG0]]
// CHECK-SAME:      outer_dims_perm = [2, 3, 0, 1]
// CHECK-SAME:      inner_dims_pos = [3] inner_tiles = [32]
// CHECK-SAME:       into %[[INIT]]
//      CHECK:   return %[[PACK]]

// -----

func.func @tensor_pack_linalg_transpose_fold_tile_dims_transpose(%arg0: tensor<56x72x24x128xf32>) -> tensor<12x56x4x9x32x8x2xf32> {
  %0 = tensor.empty() : tensor<4x9x12x56x8x2x32xf32>
  %pack = tensor.pack %arg0
    outer_dims_perm = [3, 1, 2, 0]
    inner_dims_pos = [1, 2, 3]
    inner_tiles = [8, 2, 32]
    into %0 : tensor<56x72x24x128xf32> -> tensor<4x9x12x56x8x2x32xf32>

  %1 = tensor.empty() : tensor<12x56x4x9x32x8x2xf32>
  %transposed = linalg.transpose
    ins(%pack : tensor<4x9x12x56x8x2x32xf32>)
    outs(%1 : tensor<12x56x4x9x32x8x2xf32>)
    permutation = [2, 3, 0, 1, 6, 4, 5]
  return %transposed : tensor<12x56x4x9x32x8x2xf32>
}
//      CHECK: func @tensor_pack_linalg_transpose_fold_tile_dims_transpose(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<56x72x24x128xf32>)
//      CHECK:   %[[INIT:.+]] = tensor.empty() : tensor<12x56x4x9x32x8x2xf32>
//      CHECK:   %[[PACK:.+]] = tensor.pack %[[ARG0]]
// CHECK-SAME:      outer_dims_perm = [2, 0, 3, 1]
// CHECK-SAME:      inner_dims_pos = [3, 1, 2] inner_tiles = [32, 8, 2]
// CHECK-SAME:       into %[[INIT]]
//      CHECK:   return %[[PACK]]

// -----

func.func @tensor_pack_linalg_transpose_fold_tile_dims_outer_dims_transpose(%arg0: tensor<56x72x24x128xf32>) -> tensor<9x56x2x12x32x8x4xf32> {
  %0 = tensor.empty() : tensor<4x12x9x56x8x2x32xf32>
  %pack = tensor.pack %arg0
    outer_dims_perm = [3, 2, 1, 0]
    inner_dims_pos = [1, 2, 3]
    inner_tiles = [8, 2, 32]
    into %0 : tensor<56x72x24x128xf32> -> tensor<4x12x9x56x8x2x32xf32>

  %1 = tensor.empty() : tensor<9x56x2x12x32x8x4xf32>
  %transposed = linalg.transpose
    ins(%pack : tensor<4x12x9x56x8x2x32xf32>)
    outs(%1 : tensor<9x56x2x12x32x8x4xf32>)
    permutation = [2, 3, 5, 1, 6, 4, 0]
  return %transposed : tensor<9x56x2x12x32x8x4xf32>
}
//      CHECK: func @tensor_pack_linalg_transpose_fold_tile_dims_outer_dims_transpose(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<56x72x24x128xf32>)
//      CHECK:   tensor.pack
//      CHECK:   linalg.transpose

// -----

func.func @tensor_pack_linalg_transpose_fold_dynamic_outer_dims(%arg0: tensor<56x?x?x64xf32>) -> tensor<?x?x56x2x32xf32> {
  %0 = tensor.empty() : tensor<56x2x1x57x32xf32>
  %pack = tensor.pack %arg0
    outer_dims_perm = [0, 3, 2, 1]
    inner_dims_pos = [3]
    inner_tiles = [32]
    into %0 : tensor<56x?x?x64xf32> -> tensor<56x2x1x57x32xf32>

  %1 = tensor.empty() : tensor<1x57x56x2x32xf32>
  %transposed = linalg.transpose
    ins(%pack : tensor<56x2x1x57x32xf32>)
    outs(%1 : tensor<1x57x56x2x32xf32>)
    permutation = [2, 3, 0, 1, 4]

  %return_value = tensor.cast %transposed : tensor<1x57x56x2x32xf32> to tensor<?x?x56x2x32xf32>
  return %return_value : tensor<?x?x56x2x32xf32>
}
//      CHECK: func @tensor_pack_linalg_transpose_fold_dynamic_outer_dims(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<56x?x?x64xf32>)
//  CHECK-DAG:   %[[c1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[c2:.+]] = arith.constant 2 : index
//      CHECK:   %[[dim:.+]] = tensor.dim %[[ARG0]], %[[c1]] : tensor<56x?x?x64xf32>
//      CHECK:   %[[dim_0:.+]] = tensor.dim %[[ARG0]], %[[c2]] : tensor<56x?x?x64xf32>
//      CHECK:   %[[INIT:.+]] = tensor.empty(%[[dim_0]], %[[dim]]) : tensor<?x?x56x2x32xf32>
//      CHECK:   %[[PACK:.+]] = tensor.pack %[[ARG0]]
// CHECK-SAME:      outer_dims_perm = [2, 1, 0, 3]
// CHECK-SAME:      inner_dims_pos = [3] inner_tiles = [32]
// CHECK-SAME:       into %[[INIT]]
//      CHECK:   return %[[PACK]]

// -----

func.func @tensor_pack_linalg_transpose_fold_dynamic_outer_and_tile_dims(%arg0: tensor<56x?x?x128xf32>) -> tensor<?x?x56x9x32x8x2xf32> {
  %0 = tensor.empty() : tensor<56x9x12x4x8x2x32xf32>
  %pack = tensor.pack %arg0
    inner_dims_pos = [1, 2, 3]
    inner_tiles = [8, 2, 32]
    into %0 : tensor<56x?x?x128xf32> -> tensor<56x9x12x4x8x2x32xf32>

  %1 = tensor.empty() : tensor<12x4x56x9x32x8x2xf32>
  %transposed = linalg.transpose
    ins(%pack : tensor<56x9x12x4x8x2x32xf32>)
    outs(%1 : tensor<12x4x56x9x32x8x2xf32>)
    permutation = [2, 3, 0, 1, 6, 4, 5]

  %return_value = tensor.cast %transposed : tensor<12x4x56x9x32x8x2xf32> to tensor<?x?x56x9x32x8x2xf32>
  return %return_value : tensor<?x?x56x9x32x8x2xf32>
}
//   CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
// CHECK-LABEL:   func.func @tensor_pack_linalg_transpose_fold_dynamic_outer_and_tile_dims(
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<56x?x?x128xf32>)
//   CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[c2:.+]] = arith.constant 2 : index
//       CHECK:     %[[dim:.+]] = tensor.dim %[[ARG0]], %[[c1]] : tensor<56x?x?x128xf32>
//       CHECK:     %[[dim_0:.+]] = tensor.dim %[[ARG0]], %[[c2]] : tensor<56x?x?x128xf32>
//       CHECK:     %[[mapped_dim1:.+]] = affine.apply #[[$MAP0]]()[%[[dim]]]
//       CHECK:     %[[mapped_dim2:.+]] = affine.apply #[[$MAP1]]()[%[[dim_0]]]
//       CHECK:     %[[INIT:.+]] = tensor.empty(%[[mapped_dim2]], %[[mapped_dim1]]) : tensor<?x4x56x?x32x8x2xf32>
//       CHECK:     %[[PACK:.+]] = tensor.pack %[[ARG0]] outer_dims_perm = [2, 3, 0, 1] inner_dims_pos = [3, 1, 2] inner_tiles = [32, 8, 2] into %[[INIT]] : tensor<56x?x?x128xf32> -> tensor<?x4x56x?x32x8x2xf32>
//       CHECK:     %[[CAST:.+]] = tensor.cast %[[PACK]] : tensor<?x4x56x?x32x8x2xf32> to tensor<?x?x56x9x32x8x2xf32>
//       CHECK:     return %[[CAST]] : tensor<?x?x56x9x32x8x2xf32>
//       CHECK:   }

// -----

func.func @tensor_pack_linalg_transpose_fold_dynamic_outer_dims_tile_dims_tile_sizes(%arg0: tensor<?x?x?x?xf32>, %pack_dest: tensor<?x?x?x?x?x?x?xf32>, %transpose_dest: tensor<?x?x?x?x?x?x?xf32>, %tile_p : index, %tile_q : index, %tile_r : index) -> tensor<?x?x?x?x?x?x?xf32> {
  %pack = tensor.pack %arg0
    outer_dims_perm = [3, 0, 2, 1]
    inner_dims_pos = [1, 2, 3]
    inner_tiles = [%tile_p, %tile_q, %tile_r]
    into %pack_dest : tensor<?x?x?x?xf32> -> tensor<?x?x?x?x?x?x?xf32>

  %transposed = linalg.transpose
    ins(%pack : tensor<?x?x?x?x?x?x?xf32>)
    outs(%transpose_dest : tensor<?x?x?x?x?x?x?xf32>)
    permutation = [2, 3, 0, 1, 6, 4, 5]

  return %transposed : tensor<?x?x?x?x?x?x?xf32>
}
//      CHECK: #[[$MAP:.+]] = affine_map<()[s0, s1] -> (s0 ceildiv s1)>
//      CHECK: module {
//      CHECK:   func.func @tensor_pack_linalg_transpose_fold_dynamic_outer_dims_tile_dims_tile_sizes(
// CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?x?x?xf32>,
// CHECK-SAME:   %[[PACK_DEST:.+]]: tensor<?x?x?x?x?x?x?xf32>, %[[TRANSPOSE_DEST:.+]]: tensor<?x?x?x?x?x?x?xf32>,
// CHECK-SAME:   %[[ARG1:.+]]: index, %[[ARG2:.+]]: index,
// CHECK-SAME:   %[[ARG3:.+]]: index)
//  CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//  CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//  CHECK-DAG:     %[[c2:.+]] = arith.constant 2 : index
//  CHECK-DAG:     %[[c3:.+]] = arith.constant 3 : index
//      CHECK:     %[[dim:.+]] = tensor.dim %[[ARG0]], %[[c0]] : tensor<?x?x?x?xf32>
//      CHECK:     %[[dim_0:.+]] = tensor.dim %[[ARG0]], %[[c1]] : tensor<?x?x?x?xf32>
//      CHECK:     %[[dim_1:.+]] = tensor.dim %[[ARG0]], %[[c2]] : tensor<?x?x?x?xf32>
//      CHECK:     %[[dim_2:.+]] = tensor.dim %[[ARG0]], %[[c3]] : tensor<?x?x?x?xf32>
//      CHECK:     %[[mapped_dim0:.+]] = affine.apply #[[$MAP]]()[%[[dim_2]], %[[ARG3]]]
//      CHECK:     %[[mapped_dim1:.+]] = affine.apply #[[$MAP]]()[%[[dim_0]], %[[ARG1]]]
//      CHECK:     %[[mapped_dim2:.+]] = affine.apply #[[$MAP]]()[%[[dim_1]], %[[ARG2]]]
//      CHECK:     %[[INIT:.+]] = tensor.empty(%[[mapped_dim2]], %[[mapped_dim1]], %[[mapped_dim0]], %[[dim]], %[[ARG3]], %[[ARG1]], %[[ARG2]]) : tensor<?x?x?x?x?x?x?xf32>
//      CHECK:     %[[PACK:.+]] = tensor.pack %[[ARG0]] outer_dims_perm = [2, 1, 3, 0] inner_dims_pos = [3, 1, 2] inner_tiles = [%[[ARG3]], %[[ARG1]], %[[ARG2]]] into %[[INIT]] : tensor<?x?x?x?xf32> -> tensor<?x?x?x?x?x?x?xf32>
//      CHECK:     return %[[PACK]] : tensor<?x?x?x?x?x?x?xf32>
//      CHECK:   }

// -----

func.func @linalg_transpose_tensor_pack_fold(%arg0: tensor<56x57x1x64xf32>) -> tensor<1x57x56x2x32xf32> {
  %0 = tensor.empty() : tensor<1x56x57x64xf32>
  %transposed = linalg.transpose
    ins(%arg0 : tensor<56x57x1x64xf32>)
    outs(%0 : tensor<1x56x57x64xf32>)
    permutation = [2, 0, 1, 3]

  %1 = tensor.empty() : tensor<1x57x56x2x32xf32>
  %pack = tensor.pack %transposed
    outer_dims_perm = [0, 2, 1, 3]
    inner_dims_pos = [3]
    inner_tiles = [32]
    into %1 : tensor<1x56x57x64xf32> -> tensor<1x57x56x2x32xf32>
  return %pack : tensor<1x57x56x2x32xf32>
}
//CHECK-LABEL: func @linalg_transpose_tensor_pack_fold(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<56x57x1x64xf32>)
//      CHECK:   %[[INIT:.+]] = tensor.empty() : tensor<1x57x56x2x32xf32>
//      CHECK:   %[[PACK:.+]] = tensor.pack %[[ARG0]]
// CHECK-SAME:      outer_dims_perm = [2, 1, 0, 3]
// CHECK-SAME:      inner_dims_pos = [3] inner_tiles = [32]
// CHECK-SAME:       into %[[INIT]]
//      CHECK:   return %[[PACK]]

// -----

func.func @linalg_transpose_tensor_pack_fold_with_padding(%arg0: tensor<56x57x1x55xf32>, %padding: f32) -> tensor<1x57x56x2x32xf32> {
  %0 = tensor.empty() : tensor<1x56x57x55xf32>
  %transpose = linalg.transpose
    ins(%arg0 : tensor<56x57x1x55xf32>)
    outs(%0 : tensor<1x56x57x55xf32>)
    permutation = [2, 0, 1, 3]

  %1 = tensor.empty() : tensor<1x57x56x2x32xf32>
  %pack = tensor.pack %transpose padding_value(%padding : f32)
    outer_dims_perm = [0, 2, 1, 3]
    inner_dims_pos = [3]
    inner_tiles = [32]
    into %1 : tensor<1x56x57x55xf32> -> tensor<1x57x56x2x32xf32>
  return %pack : tensor<1x57x56x2x32xf32>
}
//CHECK-LABEL: func @linalg_transpose_tensor_pack_fold_with_padding(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<56x57x1x55xf32>, %[[PADDING:.+]]: f32)
//      CHECK:   %[[INIT:.+]] = tensor.empty() : tensor<1x57x56x2x32xf32>
//      CHECK:   %[[PACK:.+]] = tensor.pack %[[ARG0]] padding_value(%[[PADDING]] : f32)
// CHECK-SAME:      outer_dims_perm = [2, 1, 0, 3]
// CHECK-SAME:      inner_dims_pos = [3] inner_tiles = [32]
// CHECK-SAME:       into %[[INIT]]
//      CHECK:   return %[[PACK]]

// -----

func.func @linalg_transpose_tensor_pack_fold_no_outer_dims_perm(%arg0: tensor<56x57x1x64xf32>) -> tensor<1x56x57x2x32xf32> {
  %0 = tensor.empty() : tensor<1x56x57x64xf32>
  %transposed = linalg.transpose
    ins(%arg0 : tensor<56x57x1x64xf32>)
    outs(%0 : tensor<1x56x57x64xf32>)
    permutation = [2, 0, 1, 3]

  %1 = tensor.empty() : tensor<1x56x57x2x32xf32>
  %pack = tensor.pack %transposed
    inner_dims_pos = [3]
    inner_tiles = [32]
    into %1 : tensor<1x56x57x64xf32> -> tensor<1x56x57x2x32xf32>
  return %pack : tensor<1x56x57x2x32xf32>
}
//CHECK-LABEL: func @linalg_transpose_tensor_pack_fold_no_outer_dims_perm(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<56x57x1x64xf32>)
//      CHECK:   %[[INIT:.+]] = tensor.empty() : tensor<1x56x57x2x32xf32>
//      CHECK:   %[[PACK:.+]] = tensor.pack %[[ARG0]]
// CHECK-SAME:      outer_dims_perm = [2, 0, 1, 3]
// CHECK-SAME:      inner_dims_pos = [3] inner_tiles = [32]
// CHECK-SAME:       into %[[INIT]]
//      CHECK:   return %[[PACK]]

// -----

func.func @linalg_transpose_tensor_pack_fold_complex_inner_dims_change(%arg0: tensor<25x30x35x40xf32>, %transpose_dest: tensor<35x40x25x30xf32>, %pack_dest: tensor<3x35x5x8x5x10x5xf32>) -> tensor<3x35x5x8x5x10x5xf32> {
  %transposed = linalg.transpose
    ins(%arg0 : tensor<25x30x35x40xf32>)
    outs(%transpose_dest : tensor<35x40x25x30xf32>)
    permutation = [2, 3, 0, 1]

  %pack = tensor.pack %transposed
    outer_dims_perm = [3, 0, 2, 1]
    inner_dims_pos = [1, 3, 2]
    inner_tiles = [5, 10, 5]
    into %pack_dest : tensor<35x40x25x30xf32> -> tensor<3x35x5x8x5x10x5xf32>
  return %pack : tensor<3x35x5x8x5x10x5xf32>
}
//CHECK-LABEL:   func.func @linalg_transpose_tensor_pack_fold_complex_inner_dims_change(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<25x30x35x40xf32>,
// CHECK-SAME:     %[[ARG1:.+]]: tensor<35x40x25x30xf32>,
// CHECK-SAME:     %[[ARG2:.+]]: tensor<3x35x5x8x5x10x5xf32>) -> tensor<3x35x5x8x5x10x5xf32> {
//      CHECK:     %[[VAL0:.+]] = tensor.empty() : tensor<3x35x5x8x5x10x5xf32>
//      CHECK:     %[[PACK:.+]] = tensor.pack %[[ARG0]]
// CHECK-SAME:        outer_dims_perm = [1, 2, 0, 3]
// CHECK-SAME:        inner_dims_pos = [3, 1, 0]
// CHECK-SAME:        inner_tiles = [5, 10, 5]
// CHECK-SAME:         into %[[VAL0]]
//      CHECK:     return %[[PACK]]

// -----

func.func @linalg_transpose_tensor_pack_fold_dynamic_outer_dims_tile_dims_tile_sizes(%arg0: tensor<?x?x?x?xf32>, %transpose_dest: tensor<?x?x?x?xf32>, %pack_dest: tensor<?x?x?x?x?x?x?xf32>, %tile_p : index, %tile_q : index, %tile_r : index) -> tensor<?x?x?x?x?x?x?xf32> {
  %transposed = linalg.transpose
    ins(%arg0 : tensor<?x?x?x?xf32>)
    outs(%transpose_dest : tensor<?x?x?x?xf32>)
    permutation = [2, 3, 0, 1]

  %pack = tensor.pack %transposed
    outer_dims_perm = [3, 0, 2, 1]
    inner_dims_pos = [1, 3, 2]
    inner_tiles = [%tile_p, %tile_q, %tile_r]
    into %pack_dest : tensor<?x?x?x?xf32> -> tensor<?x?x?x?x?x?x?xf32>
  return %pack : tensor<?x?x?x?x?x?x?xf32>
}
//      CHECK:   #[[$MAP:.+]] = affine_map<()[s0, s1] -> (s0 ceildiv s1)>
//CHECK-LABEL:   func.func @linalg_transpose_tensor_pack_fold_dynamic_outer_dims_tile_dims_tile_sizes(
// CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?x?x?xf32>, %[[ARG1:.+]]: tensor<?x?x?x?xf32>,
// CHECK-SAME:   %[[ARG2:.+]]: tensor<?x?x?x?x?x?x?xf32>, %[[ARG3:.+]]: index, %[[ARG4:.+]]: index, %[[ARG5:.+]]: index) -> tensor<?x?x?x?x?x?x?xf32> {
//      CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
//      CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
//      CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
//      CHECK-DAG:     %[[C3:.+]] = arith.constant 3 : index
//      CHECK:     %[[DIM:.+]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?x?x?xf32>
//      CHECK:     %[[DIM0:.+]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x?x?xf32>
//      CHECK:     %[[DIM1:.+]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x?x?x?xf32>
//      CHECK:     %[[DIM2:.+]] = tensor.dim %[[ARG0]], %[[C3]] : tensor<?x?x?x?xf32>
//      CHECK:     %[[VAL0:.+]] = affine.apply #[[$MAP]]()[%[[DIM2]], %[[ARG3]]]
//      CHECK:     %[[VAL1:.+]] = affine.apply #[[$MAP]]()[%[[DIM0]], %[[ARG4]]]
//      CHECK:     %[[VAL2:.+]] = affine.apply #[[$MAP]]()[%[[DIM]], %[[ARG5]]]
//      CHECK:     %[[VAL3:.+]] = tensor.empty(%[[VAL1]], %[[DIM1]], %[[VAL2]], %[[VAL0]], %[[ARG3]], %[[ARG4]], %[[ARG5]]) : tensor<?x?x?x?x?x?x?xf32>
//      CHECK:     %[[PACK:.+]] = tensor.pack %[[ARG0]] outer_dims_perm = [1, 2, 0, 3] inner_dims_pos = [3, 1, 0] inner_tiles = [%[[ARG3]], %[[ARG4]], %[[ARG5]]] into %[[VAL3]] : tensor<?x?x?x?xf32> -> tensor<?x?x?x?x?x?x?xf32>
//      CHECK:     return %[[PACK]] : tensor<?x?x?x?x?x?x?xf32>

// -----

func.func @linalg_transpose_tensor_pack_multiple_tiles(%arg0: tensor<?x32x128xbf16>) -> tensor<32x?x64x16x2xbf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : bf16
  %dim = tensor.dim %arg0, %c0 : tensor<?x32x128xbf16>

  %0 = tensor.empty(%dim) : tensor<32x128x?xbf16>
  %transposed = linalg.transpose
    ins(%arg0 : tensor<?x32x128xbf16>)
    outs(%0 : tensor<32x128x?xbf16>)
    permutation = [1, 2, 0]

  %2 = tensor.empty(%dim) : tensor<32x?x64x16x2xbf16>
  %pack = tensor.pack %transposed
    padding_value(%cst : bf16)
    outer_dims_perm = [0, 2, 1]
    inner_dims_pos = [2, 1]
    inner_tiles = [16, 2]
    into %2 : tensor<32x128x?xbf16> -> tensor<32x?x64x16x2xbf16>
  return %pack : tensor<32x?x64x16x2xbf16>
}
//      CHECK:   #[[$MAP:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
//CHECK-LABEL:   func.func @linalg_transpose_tensor_pack_multiple_tiles(
// CHECK-SAME:    %[[ARG0:.+]]: tensor<?x32x128xbf16>) -> tensor<32x?x64x16x2xbf16> {
//      CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//      CHECK-DAG:   %[[CST:.+]] = arith.constant 0.000000e+00 : bf16
//      CHECK:   %[[DIM:.+]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x32x128xbf16>
//      CHECK:   %[[VAL0:.+]] = affine.apply #[[$MAP]]()[%[[DIM]]]
//      CHECK:   %[[VAL1:.+]] = tensor.empty(%[[VAL0]]) : tensor<32x?x64x16x2xbf16>
//      CHECK:   %[[PACK:.+]] = tensor.pack %[[ARG0]]
// CHECK-SAME:      padding_value(%[[CST]] : bf16)
// CHECK-SAME:      outer_dims_perm = [1, 0, 2]
// CHECK-SAME:      inner_dims_pos = [0, 2]
// CHECK-SAME:      inner_tiles = [16, 2]
// CHECK-SAME:      into %[[VAL1]] : tensor<?x32x128xbf16> -> tensor<32x?x64x16x2xbf16>
//      CHECK:   return %[[PACK]] : tensor<32x?x64x16x2xbf16>
//      CHECK:  }

// -----

func.func @linalg_transpose_tensor_unpack_fold(%arg0: tensor<1x1x4x16xi32>) -> tensor<16x4xi32> {
  %0 = tensor.empty() : tensor<1x1x16x4xi32>
  %transposed = linalg.transpose ins(%arg0 : tensor<1x1x4x16xi32>)
                outs(%0 : tensor<1x1x16x4xi32>)
                permutation = [1, 0, 3, 2]
  %1 = tensor.empty() : tensor<16x4xi32>
  %unpack = tensor.unpack %transposed
            outer_dims_perm = [0, 1]
            inner_dims_pos = [0, 1]
            inner_tiles = [16, 4] into
            %1 : tensor<1x1x16x4xi32> -> tensor<16x4xi32>
  return %unpack : tensor<16x4xi32>
}
//CHECK-LABEL:  func.func @linalg_transpose_tensor_unpack_fold(
// CHECK-SAME:   %[[ARG0:.+]]: tensor<1x1x4x16xi32>) -> tensor<16x4xi32> {
//      CHECK:     %[[OUT:.+]] = tensor.empty() : tensor<16x4xi32>
//      CHECK:     %[[UNPACK:.+]] = tensor.unpack %[[ARG0]]
// CHECK-SAME:        outer_dims_perm = [1, 0]
// CHECK-SAME:        inner_dims_pos = [1, 0]
// CHECK-SAME:        inner_tiles = [4, 16]
// CHECK-SAME:        into %[[OUT]] : tensor<1x1x4x16xi32> -> tensor<16x4xi32>
//      CHECK:     return %[[UNPACK]] : tensor<16x4xi32>
//      CHECK:   }

// -----

func.func @linalg_transpose_tensor_unpack_fold_dynamic_outer_dims_tile_dims_tile_sizes(%arg0: tensor<?x?x?x?xf32>, %transpose_dest: tensor<?x?x?x?xf32>, %unpack_dest: tensor<?x?xf32>, %tile_p : index, %tile_q : index) -> tensor<?x?xf32> {
  %transposed = linalg.transpose
    ins(%arg0 : tensor<?x?x?x?xf32>)
    outs(%transpose_dest : tensor<?x?x?x?xf32>)
    permutation = [1, 0, 3, 2]

  %unpack = tensor.unpack %transposed
    outer_dims_perm = [1, 0]
    inner_dims_pos = [0, 1]
    inner_tiles = [%tile_p, %tile_q]
    into %unpack_dest : tensor<?x?x?x?xf32> -> tensor<?x?xf32>
  return %unpack : tensor<?x?xf32>
}
//       CHECK:    #[[$MAP:.+]] = affine_map<()[s0, s1] -> (s0 * s1)>
// CHECK-LABEL:   func.func @linalg_transpose_tensor_unpack_fold_dynamic_outer_dims_tile_dims_tile_sizes(
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?x?x?xf32>, %[[ARG1:.+]]: tensor<?x?x?x?xf32>, %[[ARG2:.+]]: tensor<?x?xf32>,
//  CHECK-SAME:     %[[IDX1:.+]]: index, %[[IDX2:.+]]: index) -> tensor<?x?xf32> {
//   CHECK-DAG:       %[[CST1:.+]] = arith.constant 1 : index
//   CHECK-DAG:       %[[CST0:.+]] = arith.constant 0 : index
//   CHECK-DAG:       %[[DIM0:.+]] = tensor.dim %[[ARG0]], %[[CST0]] : tensor<?x?x?x?xf32>
//   CHECK-DAG:       %[[DIM1:.+]] = tensor.dim %[[ARG0]], %[[CST1]] : tensor<?x?x?x?xf32>
//   CHECK-DAG:       %[[AMAP0:.+]] = affine.apply #[[$MAP]]()[%[[DIM1]], %[[IDX2]]]
//   CHECK-DAG:       %[[AMAP1:.+]] = affine.apply #[[$MAP]]()[%[[DIM0]], %[[IDX1]]]
//       CHECK:       %[[OUT:.+]] = tensor.empty(%[[AMAP1]], %[[AMAP0]]) : tensor<?x?xf32>
//       CHECK:       %[[UNPACK:.+]] = tensor.unpack %[[ARG0]]
//  CHECK-SAME:         outer_dims_perm = [0, 1]
//  CHECK-SAME:         inner_dims_pos = [1, 0]
//  CHECK-SAME:         inner_tiles = [%[[IDX2]], %[[IDX1]]]
//  CHECK-SAME:         into %[[OUT]] : tensor<?x?x?x?xf32> -> tensor<?x?xf32>
//       CHECK:       return %[[UNPACK]] : tensor<?x?xf32>
//       CHECK:   }

// -----

func.func @tensor_unpack_linalg_transpose_fold(%arg0: tensor<56x57x1x64xf32>) -> tensor<3648x56xf32> {
  %0 = tensor.empty() : tensor<56x3648xf32>
  %pack = tensor.unpack %arg0
    outer_dims_perm = [0, 1]
    inner_dims_pos = [0, 1]
    inner_tiles = [1, 64]
    into %0 : tensor<56x57x1x64xf32> -> tensor<56x3648xf32>

  %1 = tensor.empty() : tensor<3648x56xf32>
  %transposed = linalg.transpose
    ins(%pack : tensor<56x3648xf32>)
    outs(%1 : tensor<3648x56xf32>)
    permutation = [1,0]
  return %transposed : tensor<3648x56xf32>
}
// CHECK-LABEL:  func.func @tensor_unpack_linalg_transpose_fold(
//  CHECK-SAME:    %[[ARG0:.+]]: tensor<56x57x1x64xf32>) -> tensor<3648x56xf32> {
//       CHECK:        %[[OUT:.+]] = tensor.empty() : tensor<3648x56xf32>
//       CHECK:        %[[UNPACK:.+]] = tensor.unpack %[[ARG0]]
//  CHECK-SAME:        outer_dims_perm = [1, 0]
//  CHECK-SAME:        inner_dims_pos = [1, 0]
//  CHECK-SAME:        inner_tiles = [1, 64]
//  CHECK-SAME:        into %[[OUT:.+]] : tensor<56x57x1x64xf32> -> tensor<3648x56xf32>
//       CHECK:       return %[[UNPACK]] : tensor<3648x56xf32>
//       CHECK:    }

// -----

func.func @tensor_padded_unpack_linalg_transpose_fold(%arg0: tensor<71x7x4x16x16xf32>) -> tensor<100x71x64xf32> {
  %0 = tensor.empty() : tensor<71x100x64xf32>
  %pack = tensor.unpack %arg0
    inner_dims_pos = [1, 2]
    inner_tiles = [16, 16]
    into %0 : tensor<71x7x4x16x16xf32> -> tensor<71x100x64xf32>

  %1 = tensor.empty() : tensor<100x71x64xf32>
  %transposed = linalg.transpose
    ins(%pack : tensor<71x100x64xf32>)
    outs(%1 : tensor<100x71x64xf32>)
    permutation = [1, 0, 2]
  return %transposed : tensor<100x71x64xf32>
}
// CHECK-LABEL:  func.func @tensor_padded_unpack_linalg_transpose_fold(
//  CHECK-SAME:    %[[ARG0:.+]]: tensor<71x7x4x16x16xf32>) -> tensor<100x71x64xf32> {
//       CHECK:        %[[OUT:.+]] = tensor.empty() : tensor<100x71x64xf32>
//       CHECK:        %[[UNPACK:.+]] = tensor.unpack %[[ARG0]]
//  CHECK-SAME:        outer_dims_perm = [1, 0, 2]
//  CHECK-SAME:        inner_dims_pos = [0, 2]
//  CHECK-SAME:        inner_tiles = [16, 16]
//  CHECK-SAME:        into %[[OUT:.+]] : tensor<71x7x4x16x16xf32> -> tensor<100x71x64xf32>
//       CHECK:       return %[[UNPACK]] : tensor<100x71x64xf32>
//       CHECK:    }

// -----

func.func @non_involution_transpose_unpack_fold(%arg0: tensor<2x3x5x4x16xi32>) -> tensor<5x48x8xi32> {
  %0 = tensor.empty() : tensor<5x2x3x16x4xi32>
  %transposed = linalg.transpose ins(%arg0 : tensor<2x3x5x4x16xi32>)
                outs(%0 : tensor<5x2x3x16x4xi32>)
                permutation = [2, 0, 1, 4, 3]
  %1 = tensor.empty() : tensor<5x48x8xi32>
  %unpack = tensor.unpack %transposed
            outer_dims_perm = [0, 2, 1]
            inner_dims_pos = [1, 2]
            inner_tiles = [16, 4] into
            %1 : tensor<5x2x3x16x4xi32> -> tensor<5x48x8xi32>
  return %unpack : tensor<5x48x8xi32>
}
//CHECK-LABEL:  func.func @non_involution_transpose_unpack_fold(
// CHECK-SAME:   %[[ARG0:.+]]: tensor<2x3x5x4x16xi32>) -> tensor<5x48x8xi32> {
//      CHECK:     %[[OUT:.+]] = tensor.empty() : tensor<5x48x8xi32>
//      CHECK:     %[[UNPACK:.+]] = tensor.unpack %[[ARG0]]
// CHECK-SAME:        outer_dims_perm = [2, 1, 0]
// CHECK-SAME:        inner_dims_pos = [2, 1]
// CHECK-SAME:        inner_tiles = [4, 16]
// CHEKC-SAME:        into %[[OUT]] : tensor<2x3x5x4x16xi32> -> tensor<5x48x8xi32>
//      CHECK:     return %[[UNPACK]] : tensor<5x48x8xi32>
//      CHECK:   }

// -----

func.func @unpack_non_involution_transpose_fold(%arg0: tensor<57x3x56x1x64xf32>) -> tensor<3648x3x56xf32> {
  %0 = tensor.empty() : tensor<3x56x3648xf32>
  %unpack = tensor.unpack %arg0
    outer_dims_perm = [2, 0, 1]
    inner_dims_pos = [1, 2]
    inner_tiles = [1, 64]
    into %0 : tensor<57x3x56x1x64xf32> -> tensor<3x56x3648xf32>

  %1 = tensor.empty() : tensor<3648x3x56xf32>
  %transposed = linalg.transpose
    ins(%unpack : tensor<3x56x3648xf32>)
    outs(%1 : tensor<3648x3x56xf32>)
    permutation = [2, 0, 1]
  return %transposed : tensor<3648x3x56xf32>
}
// CHECK-LABEL:  func.func @unpack_non_involution_transpose_fold(
//  CHECK-SAME:    %[[ARG0:.+]]: tensor<57x3x56x1x64xf32>) -> tensor<3648x3x56xf32> {
//       CHECK:        %[[OUT:.+]] = tensor.empty() : tensor<3648x3x56xf32>
//       CHECK:        %[[UNPACK:.+]] = tensor.unpack %[[ARG0]]
//  CHECK-SAME:        outer_dims_perm = [0, 1, 2]
//  CHECK-SAME:        inner_dims_pos = [2, 0]
//  CHECK-SAME:        inner_tiles = [1, 64]
//  CHECK-SAME:        into %[[OUT:.+]] : tensor<57x3x56x1x64xf32> -> tensor<3648x3x56xf32>
//       CHECK:       return %[[UNPACK]] : tensor<3648x3x56xf32>
//       CHECK:    }

// -----

func.func @transpose_unpacked_dims_no_fold(%arg0: tensor<2x16x5x4x3xi32>) -> tensor<5x32x12xi32> {
  %0 = tensor.empty() : tensor<5x2x3x16x4xi32>
  %transposed = linalg.transpose ins(%arg0 : tensor<2x16x5x4x3xi32>)
                outs(%0 : tensor<5x2x3x16x4xi32>)
                permutation = [2, 0, 4, 1, 3]
  %1 = tensor.empty() : tensor<5x32x12xi32>
  %unpack = tensor.unpack %transposed
            inner_dims_pos = [1, 2]
            inner_tiles = [16, 4] into
            %1 : tensor<5x2x3x16x4xi32> -> tensor<5x32x12xi32>
  return %unpack : tensor<5x32x12xi32>
}
//CHECK-LABEL:  func.func @transpose_unpacked_dims_no_fold(
//      CHECK:     linalg.transpose
//      CHECK:     tensor.unpack

// -----

#map = affine_map<(d0, d1, d2, d3, d4)->(d1, d2, d0, d4, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4)->(d0, d1, d2, d3, d4)>
func.func @generic_transpose_unpack_fold(%arg0: tensor<2x3x5x4x16xi32>) -> tensor<5x48x8xi32> {
  %0 = tensor.empty() : tensor<5x2x3x16x4xi32>
  %transposed = linalg.generic {
                iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"],
                indexing_maps = [#map, #map1]}
                ins(%arg0 : tensor<2x3x5x4x16xi32>)
                outs(%0 : tensor<5x2x3x16x4xi32>) {
  ^bb0(%in : i32, %out : i32):
    linalg.yield %in : i32
  } -> tensor<5x2x3x16x4xi32>
  %1 = tensor.empty() : tensor<5x48x8xi32>
  %unpack = tensor.unpack %transposed
            outer_dims_perm = [0, 2, 1]
            inner_dims_pos = [1, 2]
            inner_tiles = [16, 4] into
            %1 : tensor<5x2x3x16x4xi32> -> tensor<5x48x8xi32>
  return %unpack : tensor<5x48x8xi32>
}
//CHECK-LABEL:  func.func @generic_transpose_unpack_fold(
// CHECK-SAME:   %[[ARG0:.+]]: tensor<2x3x5x4x16xi32>) -> tensor<5x48x8xi32> {
//      CHECK:     %[[OUT:.+]] = tensor.empty() : tensor<5x48x8xi32>
//      CHECK:     %[[UNPACK:.+]] = tensor.unpack %[[ARG0]]
// CHECK-SAME:        outer_dims_perm = [2, 1, 0]
// CHECK-SAME:        inner_dims_pos = [2, 1]
// CHECK-SAME:        inner_tiles = [4, 16]
// CHEKC-SAME:        into %[[OUT]] : tensor<2x3x5x4x16xi32> -> tensor<5x48x8xi32>
//      CHECK:     return %[[UNPACK]] : tensor<5x48x8xi32>
//      CHECK:   }

// -----

#map = affine_map<(d0, d1, d2)->(d1, d2, d0)>
#map1 = affine_map<(d0, d1, d2)->(d0, d1, d2)>
func.func @unpack_generic_transpose_fold(%arg0: tensor<57x3x56x1x64xf32>) -> tensor<3648x3x56xf32> {
  %0 = tensor.empty() : tensor<3x56x3648xf32>
  %unpack = tensor.unpack %arg0
    outer_dims_perm = [2, 0, 1]
    inner_dims_pos = [1, 2]
    inner_tiles = [1, 64]
    into %0 : tensor<57x3x56x1x64xf32> -> tensor<3x56x3648xf32>

  %1 = tensor.empty() : tensor<3648x3x56xf32>
  %transposed = linalg.generic {
                iterator_types = ["parallel", "parallel", "parallel"],
                indexing_maps = [#map, #map1]}
                ins(%unpack : tensor<3x56x3648xf32>)
                outs(%1 : tensor<3648x3x56xf32>) {
  ^bb0(%in : f32, %out : f32):
    linalg.yield %in : f32
  } -> tensor<3648x3x56xf32>
  return %transposed : tensor<3648x3x56xf32>
}
// CHECK-LABEL:  func.func @unpack_generic_transpose_fold(
//  CHECK-SAME:    %[[ARG0:.+]]: tensor<57x3x56x1x64xf32>) -> tensor<3648x3x56xf32> {
//       CHECK:        %[[OUT:.+]] = tensor.empty() : tensor<3648x3x56xf32>
//       CHECK:        %[[UNPACK:.+]] = tensor.unpack %[[ARG0]]
//  CHECK-SAME:        outer_dims_perm = [0, 1, 2]
//  CHECK-SAME:        inner_dims_pos = [2, 0]
//  CHECK-SAME:        inner_tiles = [1, 64]
//  CHECK-SAME:        into %[[OUT:.+]] : tensor<57x3x56x1x64xf32> -> tensor<3648x3x56xf32>
//       CHECK:       return %[[UNPACK]] : tensor<3648x3x56xf32>
//       CHECK:    }
