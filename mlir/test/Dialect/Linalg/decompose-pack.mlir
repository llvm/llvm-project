// RUN: mlir-opt -split-input-file \
// RUN: -transform-preload-library='transform-library-paths=%p/td/decompose-pack.mlir' \
// RUN: -transform-interpreter=entry-point=decompose_pack %s | FileCheck %s

func.func @simple_KCRS_to_KCRSsr(%arg0: tensor<?x?xi32>, %arg1: tensor<1x1x?x1xi32>) -> tensor<1x1x?x1xi32> {
  %c8 = arith.constant 8 : index
  %c5 = arith.constant 5 : i32
  %pack = linalg.pack %arg0 padding_value(%c5 : i32) inner_dims_pos = [0, 1] inner_tiles = [%c8, 1] into %arg1 : tensor<?x?xi32> -> tensor<1x1x?x1xi32>
  return %pack : tensor<1x1x?x1xi32>
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<()[s0] -> (-s0 + 8)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<()[s0] -> (-s0 + 1)>

// CHECK-LABEL:   func.func @simple_KCRS_to_KCRSsr(
// CHECK-SAME:      %[[SRC:[a-zA-Z0-9]+]]: tensor<?x?xi32>,
// CHECK-SAME:      %[[DEST:[a-zA-Z0-9]+]]: tensor<1x1x?x1xi32>) -> tensor<1x1x?x1xi32>
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 5 : i32
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = tensor.dim %[[SRC]], %[[VAL_4]] : tensor<?x?xi32>
// CHECK:           %[[VAL_6:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[VAL_5]]]
// CHECK:           %[[VAL_7:.*]] = tensor.dim %[[SRC]], %[[VAL_2]] : tensor<?x?xi32>
// CHECK:           %[[VAL_8:.*]] = affine.apply #[[$ATTR_1]](){{\[}}%[[VAL_7]]]
// CHECK:           %[[PAD:.*]] = tensor.pad %[[SRC]] low[0, 0] high{{\[}}%[[VAL_6]], %[[VAL_8]]] {
// CHECK:           ^bb0(%[[VAL_10:.*]]: index, %[[VAL_11:.*]]: index):
// CHECK:             tensor.yield %[[VAL_3]] : i32
// CHECK:           } : tensor<?x?xi32> to tensor<8x1xi32>
// CHECK:           %[[INSERT:.*]] = tensor.insert_slice %[[PAD:.*]] into %[[DEST]][0, 0, 0, 0] [1, 1, 8, 1] [1, 1, 1, 1] : tensor<8x1xi32> into tensor<1x1x?x1xi32>
// CHECK:           return %[[INSERT]] : tensor<1x1x?x1xi32>

// -----

func.func @simple_pad_and_pack_static_tiles(%input: tensor<5x1xf32>, %output: tensor<1x1x8x2xf32>, %pad: f32) -> tensor<1x1x8x2xf32> {
  %0 = linalg.pack %input padding_value(%pad : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %output : tensor<5x1xf32> -> tensor<1x1x8x2xf32>
  return %0 : tensor<1x1x8x2xf32>
}
// CHECK: #[[$ATTR_0:.+]] = affine_map<()[s0] -> (s0 - 5)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<()[s0] -> (s0 - 1)>

// CHECK-LABEL: func.func @simple_pad_and_pack_static_tiles
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[PAD_VAL:[a-zA-Z0-9]+]]
// CHECK:         %[[PAD:.+]] = tensor.pad %[[SRC]] low[0, 0] high[3, 1]
// CHECK:           tensor.yield %[[PAD_VAL]]
// CHECK-NOT:     linalg.transpose
// CHECK:         %[[INSERT:.+]] = tensor.insert_slice %[[PAD]] into %[[DEST]]
// CHECK-SAME:      [0, 0, 0, 0] [1, 1, 8, 2] [1, 1, 1, 1]
// CHECK:         return %[[INSERT]]

/// Same as example above, but with 1 dynamic tile size.

func.func @simple_pad_and_pack_dynamic_tile(%input: tensor<5x1xf32>, %output: tensor<1x1x?x2xf32>, %pad: f32, %tile_dim_0: index) -> tensor<1x1x?x2xf32> {
  %0 = linalg.pack %input padding_value(%pad : f32) inner_dims_pos = [0, 1] inner_tiles = [%tile_dim_0, 2] into %output : tensor<5x1xf32> -> tensor<1x1x?x2xf32>
  return %0 : tensor<1x1x?x2xf32>
}
// CHECK-LABEL:   func.func @simple_pad_and_pack_dynamic_tile(
// CHECK-SAME:      %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[DEST:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[PAD_VAL:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[TILE_DIM_0:[a-zA-Z0-9]+]]: index) -> tensor<1x1x?x2xf32> {
// CHECK:           %[[PAD_HIGH:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[TILE_DIM_0]]]
// CHECK:           %[[PAD:.*]] = tensor.pad %[[SRC]] low[0, 0] high{{\[}}%[[PAD_HIGH]], 1] {
// CHECK:             tensor.yield %[[PAD_VAL]] : f32
// CHECK-NOT:       linalg.transpose
// CHECK:           %[[RES:.*]] = tensor.insert_slice %[[PAD:.*]] into %[[DEST]][0, 0, 0, 0] [1, 1, %[[TILE_DIM_0]], 2] [1, 1, 1, 1] : tensor<?x2xf32> into tensor<1x1x?x2xf32>
// CHECK:           return %[[RES]] : tensor<1x1x?x2xf32>

/// Same as example above, but the dynamic tile size is a compile-time constant
/// that's folded away.

func.func @simple_pad_and_pack_dynamic_tile_cst(%input: tensor<5x1xf32>, %output: tensor<1x1x?x2xf32>, %pad: f32) -> tensor<1x1x?x2xf32> {
  %tile_dim_0 = arith.constant 8 : index
  %0 = linalg.pack %input padding_value(%pad : f32) inner_dims_pos = [0, 1] inner_tiles = [%tile_dim_0, 2] into %output : tensor<5x1xf32> -> tensor<1x1x?x2xf32>
  return %0 : tensor<1x1x?x2xf32>
}
// CHECK-LABEL:   func.func @simple_pad_and_pack_dynamic_tile_cst(
// CHECK-SAME:      %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[DEST:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[PAD_VAL:[a-zA-Z0-9]+]]: f32) -> tensor<1x1x?x2xf32> {
// CHECK:           %[[PAD:.*]] = tensor.pad %[[SRC]] low[0, 0] high[3, 1] {
// CHECK:             tensor.yield %[[PAD_VAL]] : f32
// CHECK:           } : tensor<5x1xf32> to tensor<8x2xf32>
// CHECK:           %[[RES:.*]] = tensor.insert_slice %[[PAD:.*]] into %[[DEST]][0, 0, 0, 0] [1, 1, 8, 2] [1, 1, 1, 1] : tensor<8x2xf32> into tensor<1x1x?x2xf32>
// CHECK:           return %[[RES]] : tensor<1x1x?x2xf32>

func.func @simple_pad_and_pack_dynamic_tile_transpose(%input: tensor<5x1xf32>, %output: tensor<1x1x2x?xf32>, %pad: f32, %tile_dim_1: index) -> tensor<1x1x2x?xf32> {
  %0 = linalg.pack %input padding_value(%pad : f32) inner_dims_pos = [1, 0] inner_tiles = [2, %tile_dim_1] into %output : tensor<5x1xf32> -> tensor<1x1x2x?xf32>
  return %0 : tensor<1x1x2x?xf32>
}
// CHECK-LABEL:   func.func @simple_pad_and_pack_dynamic_tile_transpose(
// CHECK-SAME:      %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[DEST:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[PAD_VAL:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[TILE_DIM_1:[a-zA-Z0-9]+]]: index) -> tensor<1x1x2x?xf32> {
// CHECK:           %[[PAD_HIGH:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[TILE_DIM_1]]]
// CHECK:           %[[PAD:.*]] = tensor.pad %[[SRC]] low[0, 0] high{{\[}}%[[PAD_HIGH]], 1] {
// CHECK:            tensor.yield %[[PAD_VAL]] : f32
// CHECK-NEXT:      } : tensor<5x1xf32> to tensor<?x2xf32>
// CHECK:           %[[EMPTY:.*]] = tensor.empty(%[[TILE_DIM_1]]) : tensor<2x?xf32>
// CHECK:           %[[TR:.*]] = linalg.transpose
// CHECK-SAME:        ins(%[[PAD:.*]] : tensor<?x2xf32>)
// CHECK-SAME:        outs(%[[EMPTY]] : tensor<2x?xf32>)
// CHECK-SAME:        permutation = [1, 0]
// CHECK:           %[[RES:.*]] = tensor.insert_slice %[[TR]] into %[[DEST]][0, 0, 0, 0] [1, 1, 2, %[[TILE_DIM_1]]] [1, 1, 1, 1] : tensor<2x?xf32> into tensor<1x1x2x?xf32>
// CHECK:           return %[[RES]] : tensor<1x1x2x?xf32>

/// Same as example above, but with 1 scalable tile size.

/// NOTE: For this example to make sense in practice, the "?" in the output shape
///       should effectively be 8 * vector.vscale (and that's what tensor.dim
///       below should return).

func.func @simple_pad_and_pack_scalable_tile(%input: tensor<5x1xf32>, %output: tensor<1x1x?x2xf32>, %pad: f32) -> tensor<1x1x?x2xf32> {
  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %vscale, %c8 : index
  %0 = linalg.pack %input padding_value(%pad : f32) inner_dims_pos = [0, 1] inner_tiles = [%c8_vscale, 2] into %output : tensor<5x1xf32> -> tensor<1x1x?x2xf32>
  return %0 : tensor<1x1x?x2xf32>
}

// CHECK-LABEL:   func.func @simple_pad_and_pack_scalable_tile(
// CHECK-SAME:      %[[SRC:[a-zA-Z0-9]+]]: tensor<5x1xf32>,
// CHECK-SAME:      %[[DEST:[a-zA-Z0-9]+]]: tensor<1x1x?x2xf32>,
// CHECK-SAME:      %[[PAD_VAL:[a-zA-Z0-9]+]]: f32) -> tensor<1x1x?x2xf32> {
// CHECK-DAG:       %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG:       %[[VS:.+]] = vector.vscale
// CHECK:           %[[C8_VS:.+]] = arith.muli %[[VS]], %[[C8]] : index
// CHECK:           %[[PAD_HIGH:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[C8_VS]]]
// CHECK:           %[[PAD:.+]] = tensor.pad %[[SRC]] low[0, 0] high{{\[}}%[[PAD_HIGH]], 1] {
// CHECK:             tensor.yield %[[PAD_VAL]] : f32
// CHECK-NOT:       linalg.transpose
// CHECK:           %[[RES:.*]] = tensor.insert_slice %[[PAD:.*]] into %[[DEST]][0, 0, 0, 0] [1, 1, %[[C8_VS]], 2] [1, 1, 1, 1] : tensor<?x2xf32> into tensor<1x1x?x2xf32>
// CHECK:           return %[[RES]] : tensor<1x1x?x2xf32>


/// Same as example above, but with both tile sizes dynamic.

func.func @simple_pad_and_pack_dynamic_tiles(%input: tensor<5x1xf32>, %output: tensor<1x1x?x?xf32>, %pad: f32, %tile_dim_0: index, %tile_dim_1: index) -> tensor<1x1x?x?xf32> {
  %0 = linalg.pack %input padding_value(%pad : f32) inner_dims_pos = [0, 1] inner_tiles = [%tile_dim_0, %tile_dim_1] into %output : tensor<5x1xf32> -> tensor<1x1x?x?xf32>
  return %0 : tensor<1x1x?x?xf32>
}
// CHECK-LABEL:   func.func @simple_pad_and_pack_dynamic_tiles(
// CHECK-SAME:      %[[SRC:[a-zA-Z0-9]+]]: tensor<5x1xf32>,
// CHECK-SAME:      %[[DEST:[a-zA-Z0-9]+]]: tensor<1x1x?x?xf32>,
// CHECK-SAME:      %[[PAD_VAL:[a-zA-Z0-9]+]]: f32,
// CHECK-SAME:      %[[TILE_DIM_0:[a-zA-Z0-9]+]]: index,
// CHECK-SAME:      %[[TILE_DIM_1:[a-zA-Z0-9]+]]: index) -> tensor<1x1x?x?xf32> {
// CHECK:           %[[PAD_HIGH_1:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[TILE_DIM_0]]]
// CHECK:           %[[PAD_HIGH_2:.*]] = affine.apply #[[$ATTR_1]](){{\[}}%[[TILE_DIM_1]]]
// CHECK:           %[[PAD:.*]] = tensor.pad %[[SRC]] low[0, 0] high{{\[}}%[[PAD_HIGH_1]], %[[PAD_HIGH_2]]] {
// CHECK:             tensor.yield %[[PAD_VAL]] : f32
// CHECK-NOT:       linalg.transpose
// CHECK:           %[[RES:.*]] = tensor.insert_slice %[[PAD]] into %[[DEST]][0, 0, 0, 0] [1, 1, %[[TILE_DIM_0]], %[[TILE_DIM_1]]] [1, 1, 1, 1] : tensor<?x?xf32> into tensor<1x1x?x?xf32>
// CHECK:           return %[[RES]] : tensor<1x1x?x?xf32>

// -----

func.func @simple_pad_and_pack_dynamic_tile_not_all_dims_tiled(%input: tensor<1x1x5x1xf32>, %output: tensor<1x1x1x1x2x?xf32>, %pad: f32, %high: index) -> tensor<1x1x1x1x2x?xf32> {
  %0 = linalg.pack %input padding_value(%pad : f32) outer_dims_perm = [1, 0, 2, 3] inner_dims_pos = [3, 2] inner_tiles = [2, %high] into %output : tensor<1x1x5x1xf32> -> tensor<1x1x1x1x2x?xf32>
  return %0 : tensor<1x1x1x1x2x?xf32>
}
// CHECK: #[[$ATTR_2:.+]] = affine_map<()[s0] -> (s0 - 5)>
// CHECK-LABEL:   func.func @simple_pad_and_pack_dynamic_tile_not_all_dims_tiled
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1x5x1xf32>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<1x1x1x1x2x?xf32>,
// CHECK-SAME:      %[[VAL_2:.*]]: f32,
// CHECK-SAME:      %[[VAL_3:.*]]: index) -> tensor<1x1x1x1x2x?xf32> {
// CHECK:           %[[VAL_4:.*]] = affine.apply #[[$ATTR_2]](){{\[}}%[[VAL_3]]]
// CHECK:           %[[VAL_5:.*]] = tensor.pad %[[VAL_0]] low[0, 0, 0, 0] high[0, 0, %[[VAL_4]], 1] {
// CHECK:           ^bb0(%[[VAL_6:.*]]: index, %[[VAL_7:.*]]: index, %[[VAL_8:.*]]: index, %[[VAL_9:.*]]: index):
// CHECK:             tensor.yield %[[VAL_2]] : f32
// CHECK:           } : tensor<1x1x5x1xf32> to tensor<1x1x?x2xf32>
// CHECK:           %[[VAL_10:.*]] = tensor.empty(%[[VAL_3]]) : tensor<1x1x2x?xf32>
// CHECK:           %[[VAL_11:.*]] = linalg.transpose
// CHECK-SAME:        ins(%[[VAL_12:.*]] : tensor<1x1x?x2xf32>)
// CHECK-SAME:        outs(%[[VAL_10]] : tensor<1x1x2x?xf32>)
// CHECK-SAME:        permutation = [0, 1, 3, 2]
// CHECK:           %[[VAL_13:.*]] = tensor.insert_slice %[[VAL_11]] into %[[VAL_1]][0, 0, 0, 0, 0, 0] [1, 1, 1, 1, 2, %[[VAL_3]]] [1, 1, 1, 1, 1, 1] : tensor<1x1x2x?xf32> into tensor<1x1x1x1x2x?xf32>
// CHECK:           return %[[VAL_13]] : tensor<1x1x1x1x2x?xf32>

// -----

func.func @simple_NC_to_CNnc(%arg0: tensor<32x8xf32>, %arg1: tensor<1x1x32x8xf32>) -> tensor<1x1x32x8xf32>{
  %0 = linalg.pack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %arg1 : tensor<32x8xf32> -> tensor<1x1x32x8xf32>
  return %0 : tensor<1x1x32x8xf32>
}
// CHECK-LABEL: func.func @simple_NC_to_CNnc
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK-NOT:     linalg.transpose
// CHECK:         %[[INSERT:.+]] = tensor.insert_slice %[[SRC]] into %[[DEST]]
// CHECK-SAME:      [0, 0, 0, 0] [1, 1, 32, 8] [1, 1, 1, 1]
// CHECK:         return %[[INSERT]]

// -----

func.func @simple_CHW_to_CHWhwc(%arg0: tensor<3x5x7xf32>, %arg1: tensor<1x1x1x5x7x3xf32>) -> tensor<1x1x1x5x7x3xf32> {
  %0 = linalg.pack %arg0 inner_dims_pos = [1, 2, 0] inner_tiles = [5, 7, 3] into %arg1 : tensor<3x5x7xf32> -> tensor<1x1x1x5x7x3xf32>
  return %0 : tensor<1x1x1x5x7x3xf32>
}
// CHECK-LABEL: func.func @simple_CHW_to_CHWhwc
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<5x7x3xf32>
// CHECK:         %[[TRANSP:.+]] =  linalg.transpose
// CHECK-SAME:      ins(%[[SRC]] : tensor<3x5x7xf32>)
// CHECK-SAME:      outs(%[[EMPTY]] : tensor<5x7x3xf32>)
// CHECK-SAME:      permutation = [1, 2, 0]
// CHECK:         %[[INSERT:.+]] = tensor.insert_slice %[[TRANSP]] into %[[DEST]]
// CHECK-SAME:      [0, 0, 0, 0, 0, 0] [1, 1, 1, 5, 7, 3] [1, 1, 1, 1, 1, 1]
// CHECK:         return %[[INSERT]]

// -----

func.func @simple_KCRS_to_KRSCsr(%arg0: tensor<1x1x32x8xf32>, %arg1: tensor<1x1x1x1x8x32xf32>) -> tensor<1x1x1x1x8x32xf32> {
  %0 = linalg.pack %arg0 outer_dims_perm = [0, 2, 3, 1] inner_dims_pos = [3, 2] inner_tiles = [8, 32] into %arg1 : tensor<1x1x32x8xf32> -> tensor<1x1x1x1x8x32xf32>
  return %0 : tensor<1x1x1x1x8x32xf32>
}
// CHECK-LABEL: func.func @simple_KCRS_to_KRSCsr
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<1x1x8x32xf32>
// CHECK:         %[[TRANSP:.+]] =  linalg.transpose
// CHECK-SAME:      ins(%[[SRC]] : tensor<1x1x32x8xf32>
// CHECK-SAME:      outs(%[[EMPTY]] : tensor<1x1x8x32xf32>)
// CHECK-SAME:      permutation = [0, 1, 3, 2]
// CHECK:         %[[INSERT:.+]] = tensor.insert_slice %[[TRANSP]] into %[[DEST]]
// CHECK-SAME:      [0, 0, 0, 0, 0, 0] [1, 1, 1, 1, 8, 32] [1, 1, 1, 1, 1, 1]
// CHECK:         return %[[INSERT]]

// -----

// The following example shows a pack operation that is defined with inner
// dimension positions that are not adjacent, i.e. `[2, 0]`. And the outer
// dimensions of the packed tensor are of unit values, i.e. `1x1x1`.
func.func @pack_with_non_adjacent_inner_dims_pos_and_unit_outer(%arg0: tensor<1x1x4xf32>, %arg1: tensor<1x1x1x4x1xf32>) -> tensor<1x1x1x4x1xf32> {
  %pack = linalg.pack %arg0 outer_dims_perm = [1, 2, 0] inner_dims_pos = [2, 0] inner_tiles = [4, 1] into %arg1 : tensor<1x1x4xf32> -> tensor<1x1x1x4x1xf32>
  return %pack : tensor<1x1x1x4x1xf32>
}
// CHECK-LABEL: func.func @pack_with_non_adjacent_inner_dims_pos_and_unit_outer
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<1x4x1xf32>
// CHECK:         %[[TRANSP:.+]] = linalg.transpose
// CHECK-SAME:      ins(%[[SRC]] : tensor<1x1x4xf32>)
// CHECK-SAME:      outs(%[[EMPTY]] : tensor<1x4x1xf32>)
// CHECK-SAME:      permutation = [1, 2, 0]
// CHECK:         %[[INSERT:.+]] = tensor.insert_slice %[[TRANSP]] into %[[DEST]]
// CHECK-SAME:      [0, 0, 0, 0, 0] [1, 1, 1, 4, 1] [1, 1, 1, 1, 1] : tensor<1x4x1xf32> into tensor<1x1x1x4x1xf32>
// CHECK:         return %[[INSERT]]

// -----

// The following example shows a pack operation where the inner dimension
// positions are specified as [2, 1] which are termed adjacent trailing
// dimensions as they contain the last dimension of the source tensor with a
// neighboring dimension. [1, 2] would also be considered trailing adjacent.
// And the outer dimensions of the packed tensor are all set to unit values
// of `1x1x1`.
func.func @pack_with_adjacent_trailing_dimensions_inner_dims_pos_and_unit_outer(%arg0: tensor<1x1x4xf32>, %arg1: tensor<1x1x1x4x1xf32>) -> tensor<1x1x1x4x1xf32> {
  %pack = linalg.pack %arg0 outer_dims_perm = [1, 2, 0] inner_dims_pos = [2, 1] inner_tiles = [4, 1] into %arg1 : tensor<1x1x4xf32> -> tensor<1x1x1x4x1xf32>
  return %pack : tensor<1x1x1x4x1xf32>
}
// CHECK-LABEL: func.func @pack_with_adjacent_trailing_dimensions_inner_dims_pos_and_unit_outer
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<1x4x1xf32>
// CHECK:         %[[TRANSP:.+]] = linalg.transpose
// CHECK-SAME:      ins(%[[SRC]] : tensor<1x1x4xf32>)
// CHECK-SAME:      outs(%[[EMPTY]] : tensor<1x4x1xf32>)
// CHECK-SAME:      permutation = [0, 2, 1]
// CHECK:         %[[INSERT:.+]] = tensor.insert_slice %[[TRANSP]] into %[[DEST]]
// CHECK-SAME:      [0, 0, 0, 0, 0] [1, 1, 1, 4, 1] [1, 1, 1, 1, 1] : tensor<1x4x1xf32> into tensor<1x1x1x4x1xf32>
// CHECK:         return %[[INSERT]]

// -----

// The following example shows a pack operation where the inner dims
// positions are non-adjacent and non-permuted.
func.func @pack_with_non_adjacent_and_non_permuted_inner_dims(%arg0: tensor<8x1x1x1xf32>, %arg1:tensor<1x1x1x1x8x1xf32>) -> tensor<1x1x1x1x8x1xf32> {
  %pack = linalg.pack %arg0 outer_dims_perm = [0, 1, 2, 3] inner_dims_pos = [0, 3] inner_tiles = [8, 1] into %arg1: tensor<8x1x1x1xf32> -> tensor<1x1x1x1x8x1xf32>
  return %pack : tensor<1x1x1x1x8x1xf32>
}

// CHECK-LABEL: func.func @pack_with_non_adjacent_and_non_permuted_inner_dims
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<1x1x8x1xf32>
// CHECK:         %[[TRANSP:.+]] = linalg.transpose
// CHECK-SAME:      ins(%[[SRC]] : tensor<8x1x1x1xf32>)
// CHECK-SAME:      outs(%[[EMPTY]] : tensor<1x1x8x1xf32>)
// CHECK-SAME:      permutation = [1, 2, 0, 3]
// CHECK:         %[[INSERT:.+]] = tensor.insert_slice %[[TRANSP]] into %[[DEST]]
// CHECK-SAME:      [0, 0, 0, 0, 0, 0] [1, 1, 1, 1, 8, 1] [1, 1, 1, 1, 1, 1] : tensor<1x1x8x1xf32> into tensor<1x1x1x1x8x1xf32>
// CHECK:         return %[[INSERT]]
