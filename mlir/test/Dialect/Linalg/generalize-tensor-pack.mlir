// RUN: mlir-opt -split-input-file --test-linalg-transform-patterns="test-generalize-tensor-pack"  %s | FileCheck %s

func.func @simple_KCRS_to_KCRSsr(%arg0: tensor<1x1x32x8xf32>, %arg1: tensor<1x1x1x1x8x32xf32>) -> tensor<1x1x1x1x8x32xf32> {
  %0 = tensor.pack %arg0 inner_dims_pos = [3, 2] inner_tiles = [8, 32] into %arg1 : tensor<1x1x32x8xf32> -> tensor<1x1x1x1x8x32xf32>
  return %0 : tensor<1x1x1x1x8x32xf32>
}
// CHECK-LABEL: func.func @simple_KCRS_to_KCRSsr
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK:         %[[TILE:.+]] = tensor.extract_slice %[[SRC]][0, 0, 0, 0] [1, 1, 32, 8] [1, 1, 1, 1]
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<8x32xf32>
// CHECK:         %[[TRANSP:.+]] =  linalg.transpose
// CHECK-SAME:      ins(%[[TILE]] : tensor<32x8xf32>)
// CHECK-SAME:      outs(%[[EMPTY]] : tensor<8x32xf32>)
// CHECK-SAME:      permutation = [1, 0]
// CHECK:         %[[INSERT:.+]] = tensor.insert_slice %[[TRANSP]] into %[[DEST]]
// CHECK-SAME:      [0, 0, 0, 0, 0, 0] [1, 1, 1, 1, 8, 32] [1, 1, 1, 1, 1, 1]
// CHECK:         return %[[INSERT]]

// -----

func.func @simple_pad_and_pack(%input: tensor<5x1xf32>, %output: tensor<1x1x8x2xf32>, %pad: f32) -> tensor<1x1x8x2xf32> {
  %0 = tensor.pack %input padding_value(%pad : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %output : tensor<5x1xf32> -> tensor<1x1x8x2xf32>
  return %0 : tensor<1x1x8x2xf32>
}
// CHECK: #[[$ATTR_0:.+]] = affine_map<()[s0, s1] -> (s0 - s1)>

// CHECK-LABEL: func.func @simple_pad_and_pack
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[PAD_VAL:[a-zA-Z0-9]+]]
// CHECK:         %[[PAD:.+]] = tensor.pad %[[SRC]] low[0, 0] high[3, 1]
// CHECK:           tensor.yield %[[PAD_VAL]]
// CHECK-NOT:     linalg.transpose
// CHECK:         %[[INSERT:.+]] = tensor.insert_slice %[[PAD]] into %[[DEST]]
// CHECK-SAME:      [0, 0, 0, 0] [1, 1, 8, 2] [1, 1, 1, 1]
// CHECK:         return %[[INSERT]]

/// Same as example above, but with scalable sizes.

/// NOTE: For this example to make sense in practice, the "?" in the output shape
///       should effectively be 8 * vector.vscale (and that's what tensor.dim
///       below should return).

func.func @simple_pad_and_pack_scalable(%input: tensor<5x1xf32>, %output: tensor<1x1x?x2xf32>, %pad: f32) -> tensor<1x1x?x2xf32> {
  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %vscale, %c8 : index
  %0 = tensor.pack %input padding_value(%pad : f32) inner_dims_pos = [0, 1] inner_tiles = [%c8_vscale, 2] into %output : tensor<5x1xf32> -> tensor<1x1x?x2xf32>
  return %0 : tensor<1x1x?x2xf32>
}


// CHECK-LABEL:   func.func @simple_pad_and_pack_scalable(
// CHECK-SAME:      %[[SRC:[a-zA-Z0-9]+]]: tensor<5x1xf32>,
// CHECK-SAME:      %[[DEST:[a-zA-Z0-9]+]]: tensor<1x1x?x2xf32>,
// CHECK-SAME:      %[[PAD_VAL:[a-zA-Z0-9]+]]: f32) -> tensor<1x1x?x2xf32> {
// CHECK:           %[[C2:.+]] = arith.constant 2 : index
// CHECK:           %[[C5:.+]] = arith.constant 5 : index
// CHECK:           %[[C8:.+]] = arith.constant 8 : index
// CHECK:           %[[VS:.+]] = vector.vscale
// CHECK:           %[[C8_VS:.+]] = arith.muli %[[VS]], %[[C8]] : index
// CHECK:           %[[PAD_HIGH:.+]] = affine.apply #[[$ATTR_0]](){{\[}}%[[C8_VS]], %[[C5]]]
// CHECK:           %[[PAD:.+]] = tensor.pad %[[SRC]] low[0, 0] high{{\[}}%[[PAD_HIGH]], 1] {
// CHECK:             tensor.yield %[[PAD_VAL]] : f32
// CHECK-NOT:       linalg.transpose
// CHECK:           %[[SLICE:.+]] = tensor.extract_slice %[[PAD:.+]][0, 0] {{\[}}%[[C8_VS]], 2] [1, 1] : tensor<?x2xf32> to tensor<?x2xf32>
// CHECK:           %[[DIM:.+]] = tensor.dim %[[DEST]], %[[C2]] : tensor<1x1x?x2xf32>
// CHECK:           %[[RES:.+]] = tensor.insert_slice %[[SLICE]] into %[[DEST]][0, 0, 0, 0] [1, 1, %[[DIM]], 2] [1, 1, 1, 1] : tensor<?x2xf32> into tensor<1x1x?x2xf32>
// CHECK:           return %[[RES]] : tensor<1x1x?x2xf32>

// -----

func.func @simple_NC_to_CNnc(%arg0: tensor<32x8xf32>, %arg1: tensor<1x1x32x8xf32>) -> tensor<1x1x32x8xf32>{
  %0 = tensor.pack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %arg1 : tensor<32x8xf32> -> tensor<1x1x32x8xf32>
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
  %0 = tensor.pack %arg0 inner_dims_pos = [1, 2, 0] inner_tiles = [5, 7, 3] into %arg1 : tensor<3x5x7xf32> -> tensor<1x1x1x5x7x3xf32>
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

func.func @simple_KCRS_to_KRSCsr(%arg0: tensor<3x1x32x8xf32>, %arg1: tensor<3x1x1x1x8x32xf32>) -> tensor<3x1x1x1x8x32xf32> {
  %0 = tensor.pack %arg0 outer_dims_perm = [0, 2, 3, 1] inner_dims_pos = [3, 2] inner_tiles = [8, 32] into %arg1 : tensor<3x1x32x8xf32> -> tensor<3x1x1x1x8x32xf32>
  return %0 : tensor<3x1x1x1x8x32xf32>
}
// CHECK-LABEL: func.func @simple_KCRS_to_KRSCsr
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK:         %[[TILE:.+]] = tensor.extract_slice %[[SRC]][0, 0, 0, 0] [3, 1, 32, 8] [1, 1, 1, 1]
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<3x8x32xf32>
// CHECK:         %[[TRANSP:.+]] =  linalg.transpose
// CHECK-SAME:      ins(%[[TILE]] : tensor<3x32x8xf32>)
// CHECK-SAME:      outs(%[[EMPTY]] : tensor<3x8x32xf32>)
// CHECK-SAME:      permutation = [0, 2, 1]
// CHECK:         %[[INSERT:.+]] = tensor.insert_slice %[[TRANSP]] into %[[DEST]]
// CHECK-SAME:      [0, 0, 0, 0, 0, 0] [3, 1, 1, 1, 8, 32] [1, 1, 1, 1, 1, 1]
// CHECK:         return %[[INSERT]]
