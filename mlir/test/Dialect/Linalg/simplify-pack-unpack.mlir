// RUN: mlir-opt -split-input-file -test-linalg-transform-patterns="test-simplify-pack-unpack-patterns" %s | FileCheck %s

// CHECK-LABEL: func.func @single_dim_packing(
// CHECK-SAME:    %[[ARG0:.+]]: tensor<256xf32>)
// CHECK:         %[[EXPANDED:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1]] output_shape [8, 32] : tensor<256xf32> into tensor<8x32xf32>
// CHECK:         return %[[EXPANDED]] : tensor<8x32xf32>
func.func @single_dim_packing(%arg0: tensor<256xf32>) -> tensor<8x32xf32> {
  %empty = tensor.empty() : tensor<8x32xf32>
  %0 = linalg.pack %arg0 inner_dims_pos = [0] inner_tiles = [32] into %empty : tensor<256xf32> -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// -----

// CHECK-LABEL: func.func @single_dim_packing_with_padding(
// CHECK-SAME:    %[[ARG0:.+]]: tensor<255xf32>)
// CHECK-NOT:     tensor.expand_shape
// CHECK:         linalg.pack
func.func @single_dim_packing_with_padding(%arg0: tensor<255xf32>) -> tensor<8x32xf32> {
  %empty = tensor.empty() : tensor<8x32xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %0 = linalg.pack %arg0 padding_value(%cst : f32) inner_dims_pos = [0] inner_tiles = [32] into %empty : tensor<255xf32> -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// -----

// CHECK-LABEL: func.func @single_last_inner_dim_packing(
// CHECK-SAME:    %[[ARG0:.+]]: tensor<5x256xf32>)
// CHECK:         %[[EXPANDED:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0], [1, 2]] output_shape [5, 8, 32] : tensor<5x256xf32> into tensor<5x8x32xf32>
// CHECK:         return %[[EXPANDED]] : tensor<5x8x32xf32>
func.func @single_last_inner_dim_packing(%arg0: tensor<5x256xf32>) -> tensor<5x8x32xf32> {
  %empty = tensor.empty() : tensor<5x8x32xf32>
  %0 = linalg.pack %arg0 inner_dims_pos = [1] inner_tiles = [32] into %empty : tensor<5x256xf32> -> tensor<5x8x32xf32>
  return %0 : tensor<5x8x32xf32>
}

// -----

// CHECK-LABEL: func.func @pack_1d_with_outer_dims_perm(
// CHECK-SAME:    %[[ARG0:.+]]: tensor<64xf32>)
// CHECK:         %[[EXPANDED:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1]] output_shape [2, 32] : tensor<64xf32> into tensor<2x32xf32>
// CHECK:         return %[[EXPANDED]] : tensor<2x32xf32>
func.func @pack_1d_with_outer_dims_perm(%arg0: tensor<64xf32>) -> tensor<2x32xf32> {
  %empty = tensor.empty() :  tensor<2x32xf32>
  %pack = linalg.pack %arg0 outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [32] into %empty : tensor<64xf32> -> tensor<2x32xf32>
  return %pack : tensor<2x32xf32>
}

// -----

// CHECK-LABEL: func.func @single_last_inner_dim_packing_with_identity_outer_dims_perm(
// CHECK-SAME:    %[[ARG0:.+]]: tensor<5x256xf32>)
// CHECK:         %[[EXPANDED:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0], [1, 2]] output_shape [5, 8, 32] : tensor<5x256xf32> into tensor<5x8x32xf32>
// CHECK:         return %[[EXPANDED]] : tensor<5x8x32xf32>
func.func @single_last_inner_dim_packing_with_identity_outer_dims_perm(%arg0: tensor<5x256xf32>) -> tensor<5x8x32xf32> {
  %empty = tensor.empty() : tensor<5x8x32xf32>
  %0 = linalg.pack %arg0 outer_dims_perm = [0, 1] inner_dims_pos = [1] inner_tiles = [32] into %empty : tensor<5x256xf32> -> tensor<5x8x32xf32>
  return %0 : tensor<5x8x32xf32>
}

// -----

// CHECK-LABEL: func.func @packing_with_outer_dims_perm(
// CHECK-NOT:     tensor.expand_shape
// CHECK:         linalg.pack
func.func @packing_with_outer_dims_perm(%arg0: tensor<5x256xf32>) -> tensor<8x5x32xf32> {
  %empty = tensor.empty() : tensor<8x5x32xf32>
  %0 = linalg.pack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [1] inner_tiles = [32] into %empty : tensor<5x256xf32> -> tensor<8x5x32xf32>
  return %0 : tensor<8x5x32xf32>
}

// -----

// CHECK-LABEL: func.func @single_first_inner_dim_packing(
// CHECK-NOT:     tensor.expand_shape
// CHECK:         linalg.pack
func.func @single_first_inner_dim_packing(%arg0: tensor<256x5xf32>) -> tensor<8x5x32xf32> {
  %empty = tensor.empty() : tensor<8x5x32xf32>
  %0 = linalg.pack %arg0 inner_dims_pos = [0] inner_tiles = [32] into %empty : tensor<256x5xf32> -> tensor<8x5x32xf32>
  return %0 : tensor<8x5x32xf32>
}

// -----

// CHECK-LABEL: func.func @pack_1x32_to_1x32x1x1
// CHECK-SAME:    %[[ARG0:[0-9a-zA-Z]+]]
// CHECK:         %[[EXPANDED:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0], [1, 2, 3]] output_shape [1, 32, 1, 1]
// CHECK:         return %[[EXPANDED]]
func.func @pack_1x32_to_1x32x1x1(%arg0 : tensor<1x32xf32>) -> tensor<1x32x1x1xf32> {
  %empty = tensor.empty() : tensor<1x32x1x1xf32>
  %pack = linalg.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [1, 1] into %empty
    : tensor<1x32xf32> -> tensor<1x32x1x1xf32>
  return %pack : tensor<1x32x1x1xf32>
}

// -----

// CHECK-LABEL: func.func @pack_1x32_to_1x16x1x2
// CHECK-SAME:    %[[ARG0:[0-9a-zA-Z]+]]
// CHECK:         %[[EXPANDED:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0], [1, 2, 3]] output_shape [1, 16, 1, 2]
// CHECK:         return %[[EXPANDED]]
func.func @pack_1x32_to_1x16x1x2(%arg0 : tensor<1x32xf32>) -> tensor<1x16x1x2xf32> {
  %empty = tensor.empty() : tensor<1x16x1x2xf32>
  %pack = linalg.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [1, 2] into %empty
    : tensor<1x32xf32> -> tensor<1x16x1x2xf32>
  return %pack : tensor<1x16x1x2xf32>
}

// -----

// CHECK-LABEL: func.func @pack_32x1_to_16x1x2x1
// CHECK-SAME:    %[[ARG0:[0-9a-zA-Z]+]]
// CHECK:         %[[EXPANDED:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1, 2], [3]] output_shape [1, 16, 2, 1]
// CHECK:         return %[[EXPANDED]]
func.func @pack_32x1_to_16x1x2x1(%arg0 : tensor<32x1xf32>) -> tensor<1x16x2x1xf32> {
  %empty = tensor.empty() : tensor<1x16x2x1xf32>
  %pack = linalg.pack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [2, 1] into %empty
    : tensor<32x1xf32> -> tensor<1x16x2x1xf32>
  return %pack : tensor<1x16x2x1xf32>
}

// -----

// CHECK-LABEL: func.func @pack_32x1_to_16x1x1x2
// CHECK-NOT:     tensor.expand_shape
// CHECK:         linalg.pack
func.func @pack_32x1_to_16x1x1x2(%arg0 : tensor<32x1xf32>) -> tensor<16x1x1x2xf32> {
  %empty = tensor.empty() : tensor<16x1x1x2xf32>
  %pack = linalg.pack %arg0 inner_dims_pos = [1, 0] inner_tiles = [1, 2] into %empty
    : tensor<32x1xf32> -> tensor<16x1x1x2xf32>
  return %pack : tensor<16x1x1x2xf32>
}

// -----

// CHECK-LABEL: func.func @unpack_1d_to_collapse
// CHECK-SAME:    %[[ARG0:.+]]: tensor<8x32xf32>)
// CHECK:         %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1]] : tensor<8x32xf32> into tensor<256xf32>
// CHECK:         return %[[COLLAPSED]]
func.func @unpack_1d_to_collapse(%arg0: tensor<8x32xf32>) -> tensor<256xf32> {
  %empty = tensor.empty() : tensor<256xf32>
  %0 = linalg.unpack %arg0 inner_dims_pos = [0] inner_tiles = [32] into %empty : tensor<8x32xf32> -> tensor<256xf32>
  return %0 : tensor<256xf32>
}

// -----

// CHECK-LABEL: func.func @unpack_to_partial_slice
// CHECK-NOT:     tensor.collapse
// CHECK:         linalg.unpack
func.func @unpack_to_partial_slice(%arg0: tensor<8x32xf32>) -> tensor<255xf32> {
  %empty = tensor.empty() : tensor<255xf32>
  %0 = linalg.unpack %arg0 inner_dims_pos = [0] inner_tiles = [32] into %empty : tensor<8x32xf32> -> tensor<255xf32>
  return %0 : tensor<255xf32>
}

// -----

// CHECK-LABEL: func.func @unpack_dynamic
// CHECK-NOT:     tensor.collapse
// CHECK:         linalg.unpack
func.func @unpack_dynamic(%arg0: tensor<?x32xf32>) -> tensor<?xf32> {
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x32xf32>
  %size = arith.muli %d0, %c32 : index
  %empty = tensor.empty(%size) : tensor<?xf32>
  %0 = linalg.unpack %arg0 inner_dims_pos = [0] inner_tiles = [32] into %empty : tensor<?x32xf32> -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func.func @single_last_inner_dim_unpacking(
// CHECK-SAME:    %[[ARG0:.+]]: tensor<5x8x32xf32>)
// CHECK:         %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0], [1, 2]] : tensor<5x8x32xf32> into tensor<5x256xf32>
// CHECK:         return %[[COLLAPSED]] : tensor<5x256xf32>
func.func @single_last_inner_dim_unpacking(%arg0: tensor<5x8x32xf32>) -> tensor<5x256xf32> {
  %empty = tensor.empty() : tensor<5x256xf32>
  %0 = linalg.unpack %arg0 inner_dims_pos = [1] inner_tiles = [32] into %empty : tensor<5x8x32xf32> -> tensor<5x256xf32>
  return %0 : tensor<5x256xf32>
}

// -----

// CHECK-LABEL: func.func @single_last_inner_dim_unpacking_with_identity_outer_dims_perm(
// CHECK-SAME:    %[[ARG0:.+]]: tensor<5x8x32xf32>)
// CHECK:         %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0], [1, 2]] : tensor<5x8x32xf32> into tensor<5x256xf32>
// CHECK:         return %[[COLLAPSED]] : tensor<5x256xf32>
func.func @single_last_inner_dim_unpacking_with_identity_outer_dims_perm(%arg0: tensor<5x8x32xf32>) -> tensor<5x256xf32> {
  %empty = tensor.empty() : tensor<5x256xf32>
  %0 = linalg.unpack %arg0 outer_dims_perm = [0, 1] inner_dims_pos = [1] inner_tiles = [32] into %empty : tensor<5x8x32xf32> -> tensor<5x256xf32>
  return %0 : tensor<5x256xf32>
}

// -----

// CHECK-LABEL: func.func @unpacking_with_outer_dims_perm(
// CHECK-NOT:     tensor.collpase_shape
// CHECK:         linalg.unpack
func.func @unpacking_with_outer_dims_perm(%arg0: tensor<8x5x32xf32>) -> tensor<5x256xf32> {
  %empty = tensor.empty() : tensor<5x256xf32>
  %0 = linalg.unpack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [1] inner_tiles = [32] into %empty : tensor<8x5x32xf32> -> tensor<5x256xf32>
  return %0 : tensor<5x256xf32>
}

// -----

// CHECK-LABEL: func.func @single_first_inner_dim_unpacking(
// CHECK-NOT:     tensor.collapse_shape
// CHECK:         linalg.unpack
func.func @single_first_inner_dim_unpacking(%arg0: tensor<8x5x32xf32>) -> tensor<256x5xf32> {
  %empty = tensor.empty() : tensor<256x5xf32>
  %0 = linalg.unpack %arg0 inner_dims_pos = [0] inner_tiles = [32] into %empty : tensor<8x5x32xf32> -> tensor<256x5xf32>
  return %0 : tensor<256x5xf32>
}

// -----

// CHECK-LABEL: func.func @unpack_1x32x1x1_to_1x32
// CHECK-SAME:    %[[ARG0:[0-9a-zA-Z]+]]
// CHECK:         %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0], [1, 2, 3]]
// CHECK:         return %[[COLLAPSED]]
func.func @unpack_1x32x1x1_to_1x32(%arg0 : tensor<1x32x1x1xf32>) -> tensor<1x32xf32> {
  %empty = tensor.empty() : tensor<1x32xf32>
  %unpack = linalg.unpack %arg0 inner_dims_pos = [0, 1] inner_tiles = [1, 1] into %empty
    : tensor<1x32x1x1xf32> -> tensor<1x32xf32>
  return %unpack : tensor<1x32xf32>
}

// -----

// CHECK-LABEL: func.func @unpack_1x2x1x16_to_1x32
// CHECK-SAME:    %[[ARG0:[0-9a-zA-Z]+]]
// CHECK:         %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0], [1, 2, 3]]
// CHECK:         return %[[COLLAPSED]]
func.func @unpack_1x2x1x16_to_1x32(%arg0 : tensor<1x2x1x16xf32>) -> tensor<1x32xf32> {
  %empty = tensor.empty() : tensor<1x32xf32>
  %unpack = linalg.unpack %arg0 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [1, 16] into %empty
    : tensor<1x2x1x16xf32> -> tensor<1x32xf32>
  return %unpack : tensor<1x32xf32>
}

// -----

// CHECK-LABEL: func.func @unpack_16x1x2x1_to_32x1
// CHECK-SAME:    %[[ARG0:[0-9a-zA-Z]+]]
// CHECK:         %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1, 2], [3]]
// CHECK:         return %[[COLLAPSED]]
func.func @unpack_16x1x2x1_to_32x1(%arg0 : tensor<1x16x2x1xf32>) -> tensor<32x1xf32> {
  %empty = tensor.empty() : tensor<32x1xf32>
  %unpack = linalg.unpack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [2, 1] into %empty
    : tensor<1x16x2x1xf32> -> tensor<32x1xf32>
  return %unpack : tensor<32x1xf32>
}

// -----

// CHECK-LABEL: func.func @unpack_16x1x1x2_to_32x1
// CHECK-NOT:     tensor.collapse_shape
// CHECK:         linalg.unpack
func.func @unpack_16x1x1x2_to_32x1(%arg0 : tensor<16x1x1x2xf32>) -> tensor<32x1xf32> {
  %empty = tensor.empty() : tensor<32x1xf32>
  %unpack = linalg.unpack %arg0 inner_dims_pos = [1, 0] inner_tiles = [1, 2] into %empty
    : tensor<16x1x1x2xf32> -> tensor<32x1xf32>
  return %unpack : tensor<32x1xf32>
}

// -----

// CHECK-LABEL: func.func @pad_like_pack(
// CHECK-SAME:    %[[ARG0:.+]]: tensor<32x64xf32>)
// CHECK:         %[[EXPANDED:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1, 2], [3]] output_shape [1, 1, 32, 64] : tensor<32x64xf32> into tensor<1x1x32x64xf32>
// CHECK:         return %[[EXPANDED]] : tensor<1x1x32x64xf32>
func.func @pad_like_pack(%arg0: tensor<32x64xf32>) -> tensor<1x1x32x64xf32> {
  %empty = tensor.empty() : tensor<1x1x32x64xf32>
  %0 = linalg.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [32, 64] into %empty : tensor<32x64xf32> -> tensor<1x1x32x64xf32>
  return %0 : tensor<1x1x32x64xf32>
}

// -----

// CHECK-LABEL: func.func @pad_like_pack_with_outer_dims_perm(
// CHECK-SAME:    %[[ARG0:.+]]: tensor<32x64xf32>)
// CHECK:         %[[EXPANDED:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1, 2], [3]] output_shape [1, 1, 32, 64] : tensor<32x64xf32> into tensor<1x1x32x64xf32>
// CHECK:         return %[[EXPANDED]] : tensor<1x1x32x64xf32>
func.func @pad_like_pack_with_outer_dims_perm(%arg0: tensor<32x64xf32>) -> tensor<1x1x32x64xf32> {
  %empty = tensor.empty() : tensor<1x1x32x64xf32>
  %0 = linalg.pack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 64] into %empty : tensor<32x64xf32> -> tensor<1x1x32x64xf32>
  return %0 : tensor<1x1x32x64xf32>
}

// -----

// CHECK-LABEL: func.func @inner_pad_like_pack(
// CHECK-SAME:    %[[ARG0:.+]]: tensor<32x64xf32>)
// CHECK:         %[[EXPANDED:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0], [1, 2]] output_shape [32, 1, 64] : tensor<32x64xf32> into tensor<32x1x64xf32>
// CHECK:         return %[[EXPANDED]] : tensor<32x1x64xf32>
func.func @inner_pad_like_pack(%arg0: tensor<32x64xf32>) -> tensor<32x1x64xf32> {
  %empty = tensor.empty() : tensor<32x1x64xf32>
  %0 = linalg.pack %arg0 inner_dims_pos = [1] inner_tiles = [64] into %empty : tensor<32x64xf32> -> tensor<32x1x64xf32>
  return %0 : tensor<32x1x64xf32>
}

// -----

// Do not simplify pack with inner dimension shuffling.
// CHECK-LABEL: func.func @pad_and_inner_dim_shuffle_pack(
// CHECK-SAME:    %[[ARG0:.+]]: tensor<32x64xf32>)
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<1x1x64x32xf32>
// CHECK:         %[[PACK:.+]] = linalg.pack %[[ARG0]] inner_dims_pos = [1, 0] inner_tiles = [64, 32] into %[[EMPTY]] : tensor<32x64xf32> -> tensor<1x1x64x32xf32>
// CHECK:         return %[[PACK]] : tensor<1x1x64x32xf32>
func.func @pad_and_inner_dim_shuffle_pack(%arg0: tensor<32x64xf32>) -> tensor<1x1x64x32xf32> {
  %empty = tensor.empty() : tensor<1x1x64x32xf32>
  %0 = linalg.pack %arg0 inner_dims_pos = [1, 0] inner_tiles = [64, 32] into %empty : tensor<32x64xf32> -> tensor<1x1x64x32xf32>
  return %0 : tensor<1x1x64x32xf32>
}

// -----

// Do not simplify pack with inner dimension transpose.
// CHECK-LABEL: func.func @pad_like_pack_with_transpose(
// CHECK-SAME:    %[[ARG0:.+]]: tensor<32x64x16xf32>)
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<32x1x16x64xf32>
// CHECK:         %[[PACK:.+]] = linalg.pack %[[ARG0]] inner_dims_pos = [1] inner_tiles = [64] into %[[EMPTY]] : tensor<32x64x16xf32> -> tensor<32x1x16x64xf32>
// CHECK:         return %[[PACK]] : tensor<32x1x16x64xf32>
func.func @pad_like_pack_with_transpose(%arg0: tensor<32x64x16xf32>) -> tensor<32x1x16x64xf32> {
  %empty = tensor.empty() : tensor<32x1x16x64xf32>
  %0 = linalg.pack %arg0 inner_dims_pos = [1] inner_tiles = [64] into %empty : tensor<32x64x16xf32> -> tensor<32x1x16x64xf32>
  return %0 : tensor<32x1x16x64xf32>
}

// -----

// CHECK-LABEL: func.func @unpad_like_unpack(
// CHECK-SAME:    %[[ARG0:.+]]: tensor<1x1x32x64xf32>)
// CHECK:         %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1, 2], [3]] : tensor<1x1x32x64xf32> into tensor<32x64xf32>
// CHECK:         return %[[COLLAPSED]] : tensor<32x64xf32>
func.func @unpad_like_unpack(%arg0: tensor<1x1x32x64xf32>) -> tensor<32x64xf32> {
  %empty = tensor.empty() : tensor<32x64xf32>
  %0 = linalg.unpack %arg0 inner_dims_pos = [0, 1] inner_tiles = [32, 64] into %empty : tensor<1x1x32x64xf32> -> tensor<32x64xf32>
  return %0 : tensor<32x64xf32>
}

// -----

// CHECK-LABEL: func.func @unpad_like_unpack_with_outer_dims_perm(
// CHECK-SAME:    %[[ARG0:.+]]: tensor<1x1x32x64xf32>)
// CHECK:         %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1, 2], [3]] : tensor<1x1x32x64xf32> into tensor<32x64xf32>
// CHECK:         return %[[COLLAPSED]] : tensor<32x64xf32>
func.func @unpad_like_unpack_with_outer_dims_perm(%arg0: tensor<1x1x32x64xf32>) -> tensor<32x64xf32> {
  %empty = tensor.empty() : tensor<32x64xf32>
  %0 = linalg.unpack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 64] into %empty : tensor<1x1x32x64xf32> -> tensor<32x64xf32>
  return %0 : tensor<32x64xf32>
}

// -----

// CHECK-LABEL: func.func @inner_unpad_like_unpack(
// CHECK-SAME:    %[[ARG0:.+]]: tensor<32x1x64xf32>)
// CHECK:         %[[COLLAPSED:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0], [1, 2]] : tensor<32x1x64xf32> into tensor<32x64xf32>
// CHECK:         return %[[COLLAPSED]] : tensor<32x64xf32>
func.func @inner_unpad_like_unpack(%arg0: tensor<32x1x64xf32>) -> tensor<32x64xf32> {
  %empty = tensor.empty() : tensor<32x64xf32>
  %0 = linalg.unpack %arg0 inner_dims_pos = [1] inner_tiles = [64] into %empty : tensor<32x1x64xf32> -> tensor<32x64xf32>
  return %0 : tensor<32x64xf32>
}

// -----

// Do not simplify unpack with inner dimension shuffling.
// CHECK-LABEL: func.func @unpad_and_inner_dim_shuffle_pack(
// CHECK-SAME:    %[[ARG0:.+]]: tensor<1x1x32x64xf32>)
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<64x32xf32>
// CHECK:         %[[UNPACK:.+]] = linalg.unpack %[[ARG0]] inner_dims_pos = [1, 0] inner_tiles = [32, 64] into %[[EMPTY]] : tensor<1x1x32x64xf32> -> tensor<64x32xf32>
// CHECK:         return %[[UNPACK]] : tensor<64x32xf32>
func.func @unpad_and_inner_dim_shuffle_pack(%arg0: tensor<1x1x32x64xf32>) -> tensor<64x32xf32> {
  %empty = tensor.empty() : tensor<64x32xf32>
  %0 = linalg.unpack %arg0 inner_dims_pos = [1, 0] inner_tiles = [32, 64] into %empty : tensor<1x1x32x64xf32> -> tensor<64x32xf32>
  return %0 : tensor<64x32xf32>
}

// -----

// Do not simplify unpack with inner dimension transpose.
// CHECK-LABEL: func.func @unpad_like_unpack_with_transpose(
// CHECK-SAME:    %[[ARG0:.+]]: tensor<32x1x16x64xf32>)
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<32x64x16xf32>
// CHECK:         %[[UNPACK:.+]] = linalg.unpack %[[ARG0]] inner_dims_pos = [1] inner_tiles = [64] into %[[EMPTY]] : tensor<32x1x16x64xf32> -> tensor<32x64x16xf32>
// CHECK:         return %[[UNPACK]] : tensor<32x64x16xf32>
func.func @unpad_like_unpack_with_transpose(%arg0: tensor<32x1x16x64xf32>) -> tensor<32x64x16xf32> {
  %empty = tensor.empty() : tensor<32x64x16xf32>
  %0 = linalg.unpack %arg0 inner_dims_pos = [1] inner_tiles = [64] into %empty : tensor<32x1x16x64xf32> -> tensor<32x64x16xf32>
  return %0 : tensor<32x64x16xf32>
}
