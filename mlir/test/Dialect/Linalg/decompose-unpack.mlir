// RUN: mlir-opt -split-input-file \
// RUN: -transform-preload-library='transform-library-paths=%p/td/decompose-unpack.mlir' \
// RUN: -transform-interpreter=entry-point=decompose_unpack %s | FileCheck %s

func.func @simple_KCRSsr_to_KCRS(%arg0: tensor<1x1x1x1x8x32xf32>, %arg1: tensor<1x1x32x8xf32>) -> tensor<1x1x32x8xf32> {
  %0 = linalg.unpack %arg0 inner_dims_pos = [3, 2] inner_tiles = [8, 32] into %arg1 : tensor<1x1x1x1x8x32xf32> -> tensor<1x1x32x8xf32>
  return %0 : tensor<1x1x32x8xf32>
}
// CHECK-LABEL: func.func @simple_KCRSsr_to_KCRS
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK:         %[[TILE:.+]] = tensor.extract_slice %[[SRC]][0, 0, 0, 0, 0, 0] [1, 1, 1, 1, 8, 32] [1, 1, 1, 1, 1, 1]
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<32x8xf32>
// CHECK:         %[[TRANSP:.+]] =  linalg.transpose
// CHECK-SAME:      ins(%[[TILE]] : tensor<8x32xf32>)
// CHECK-SAME:      outs(%[[EMPTY]] : tensor<32x8xf32>)
// CHECK-SAME:      permutation = [1, 0]
// CHECK:         %[[INSERT:.+]] = tensor.insert_slice %[[TRANSP]] into %[[DEST]]
// CHECK-SAME:      [0, 0, 0, 0] [1, 1, 32, 8] [1, 1, 1, 1]
// CHECK:         return %[[INSERT]]

// -----

func.func @simple_unpack_static_tiles(%input: tensor<1x1x8x2xf32>, %output: tensor<5x1xf32>) -> tensor<5x1xf32> {
  %0 = linalg.unpack %input inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %output : tensor<1x1x8x2xf32> -> tensor<5x1xf32>
  return %0 : tensor<5x1xf32>
}
// CHECK-LABEL: func.func @simple_unpack_static_tiles
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK:         %[[TILE:.+]] = tensor.extract_slice %[[SRC]][0, 0, 0, 0] [1, 1, 8, 2] [1, 1, 1, 1]
// CHECK-NOT:     linalg.transpose
//                They have the same type, so the insert_slice op is folded
//                away.
// CHECK:         %[[SLICE:.+]] = tensor.extract_slice %[[TILE]][0, 0] [5, 1] [1, 1]
// CHECK:         return %[[SLICE]]

/// Same as example above, but with 1 dynamic tile size.

func.func @simple_unpack_dynamic_tile(%input: tensor<1x1x?x2xf32>, %output: tensor<5x1xf32>, %tile_dim: index) -> tensor<5x1xf32> {
  %0 = linalg.unpack %input inner_dims_pos = [0, 1] inner_tiles = [%tile_dim, 2] into %output : tensor<1x1x?x2xf32> -> tensor<5x1xf32>
  return %0 : tensor<5x1xf32>
}
// CHECK-LABEL: func.func @simple_unpack_dynamic_tile
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[TILE_DIM:[a-zA-Z0-9]+]]
// CHECK:         %[[TILE:.+]] = tensor.extract_slice %[[SRC]][0, 0, 0, 0] [1, 1, %[[TILE_DIM]], 2] [1, 1, 1, 1]
// CHECK-NOT:     linalg.transpose
//                They have the same type, so the insert_slice op is folded
//                away.
// CHECK:         %[[SLICE:.+]] = tensor.extract_slice %[[TILE]][0, 0] [5, 1] [1, 1]
// CHECK:         return %[[SLICE]]

/// Same as example above, but with 1 dynamic tile size and a trasnpose

func.func @simple_unpack_dynamic_tile_transpose(%src: tensor<1x1x2x?xf32>, %dest: tensor<5x1xf32>, %tile_dim: index) -> tensor<5x1xf32> {
  %0 = linalg.unpack %src inner_dims_pos = [1, 0] inner_tiles = [2, %tile_dim] into %dest : tensor<1x1x2x?xf32> -> tensor<5x1xf32>
  return %0 : tensor<5x1xf32>
}
// CHECK-LABEL:   func.func @simple_unpack_dynamic_tile_transpose
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[TILE_DIM:[a-zA-Z0-9]+]]
// CHECK:           %[[TILE:.*]] = tensor.extract_slice %[[SRC]][0, 0, 0, 0] [1, 1, 2, %[[TILE_DIM]]] [1, 1, 1, 1] : tensor<1x1x2x?xf32> to tensor<2x?xf32>
// CHECK:           %[[EMPTY:.*]] = tensor.empty(%[[TILE_DIM]]) : tensor<?x2xf32>
// CHECK:           %[[TRANSP:.*]] = linalg.transpose
// CHECK-SAME:        ins(%[[TILE]] : tensor<2x?xf32>)
// CHECK-SAME:        outs(%[[EMPTY]] : tensor<?x2xf32>)
// CHECK-SAME:        permutation = [1, 0]
// CHECK:           %[[SLICE:.*]] = tensor.extract_slice %[[TRANSP]][0, 0] [5, 1] [1, 1] : tensor<?x2xf32> to tensor<5x1xf32>
// CHECK:           return %[[SLICE]] : tensor<5x1xf32>


/// Same as example above, but with 1 scalable tile size.

func.func @simple_unpack_scalable_tile(%input: tensor<1x1x?x2xf32>, %output: tensor<5x1xf32>) -> tensor<5x1xf32> {
  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  %c8_vscale = arith.muli %vscale, %c8 : index
  %0 = linalg.unpack %input inner_dims_pos = [0, 1] inner_tiles = [%c8_vscale, 2] into %output : tensor<1x1x?x2xf32> -> tensor<5x1xf32>
  return %0 : tensor<5x1xf32>
}
// CHECK-LABEL: func.func @simple_unpack_scalable_tile
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG:     %[[VS:.+]] = vector.vscale
// CHECK:         %[[C8_VS:.+]] = arith.muli %[[VS]], %[[C8]] : index
// CHECK:         %[[TILE:.+]] = tensor.extract_slice %[[SRC]][0, 0, 0, 0] [1, 1, %[[C8_VS]], 2] [1, 1, 1, 1]
// CHECK-NOT:     linalg.transpose
//                They have the same type, so the insert_slice op is folded
//                away.
// CHECK:         %[[SLICE:.+]] = tensor.extract_slice %[[TILE]][0, 0] [5, 1] [1, 1]
// CHECK:         return %[[SLICE]]

// -----

func.func @simple_CNnc_to_NC(%arg0: tensor<1x1x32x8xf32>, %arg1: tensor<32x8xf32>) -> tensor<32x8xf32>{
  %0 = linalg.unpack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %arg1 : tensor<1x1x32x8xf32> -> tensor<32x8xf32>
  return %0 : tensor<32x8xf32>
}
// CHECK-LABEL: func.func @simple_CNnc_to_NC
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK:         %[[TILE:.+]] = tensor.extract_slice %[[SRC]][0, 0, 0, 0] [1, 1, 32, 8] [1, 1, 1, 1]
// CHECK-NOT:     linalg.transpose
//                They have the same type, so the insert_slice op is folded
//                away.
// CHECK:         return %[[TILE]]

// -----

func.func @simple_NCHWc_to_NCHW(%arg0: tensor<2x1x16x8x32xf32>, %arg1: tensor<2x32x16x8xf32>) -> tensor<2x32x16x8xf32> {
  %0 = linalg.unpack %arg0 inner_dims_pos = [1] inner_tiles = [32] into %arg1 : tensor<2x1x16x8x32xf32> -> tensor<2x32x16x8xf32>
  return %0 : tensor<2x32x16x8xf32>
}
// CHECK-LABEL: func.func @simple_NCHWc_to_NCHW
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK:         %[[TILE:.+]] = tensor.extract_slice %[[SRC]][0, 0, 0, 0, 0] [2, 1, 16, 8, 32] [1, 1, 1, 1, 1]
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<2x32x16x8xf32>
// CHECK:         %[[TRANSP:.+]] =  linalg.transpose
// CHECK-SAME:      ins(%[[TILE]] : tensor<2x16x8x32xf32>)
// CHECK-SAME:      outs(%[[EMPTY]] : tensor<2x32x16x8xf32>)
// CHECK-SAME:      permutation = [0, 3, 1, 2]
//                They have the same type, so the insert_slice op is folded
//                away.
// CHECK:         return %[[TRANSP]]

// -----

func.func @simple_NHWC_to_NCHW(%arg0: tensor<1x16x8x32xf32>, %arg1: tensor<1x32x16x8xf32>) -> tensor<1x32x16x8xf32> {
  %0 = linalg.unpack %arg0 outer_dims_perm = [0, 2, 3, 1] inner_dims_pos = [] inner_tiles = [] into %arg1 : tensor<1x16x8x32xf32> -> tensor<1x32x16x8xf32>
  return %0 : tensor<1x32x16x8xf32>
}
// CHECK-LABEL: func.func @simple_NHWC_to_NCHW
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK:         %[[TILE:.+]] = tensor.extract_slice %[[SRC]][0, 0, 0, 0] [1, 16, 8, 32] [1, 1, 1, 1]
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<32x16x8xf32>
// CHECK:         %[[TRANSP:.+]] =  linalg.transpose
// CHECK-SAME:      ins(%[[TILE]] : tensor<16x8x32xf32>)
// CHECK-SAME:      outs(%[[EMPTY]] : tensor<32x16x8xf32>)
// CHECK-SAME:      permutation = [2, 0, 1]
// CHECK:         %[[INSERT:.+]] = tensor.insert_slice %[[TRANSP]] into %[[DEST]]
// CHECK-SAME:      [0, 0, 0, 0] [1, 32, 16, 8] [1, 1, 1, 1]
// CHECK:         return %[[INSERT]]

// -----

func.func @unpack_with_dynamic_dims(%arg0: tensor<?x1x1x1x8x32xf32>, %arg1: tensor<?x1x32x8xf32>) -> tensor<?x1x32x8xf32> {
  %0 = linalg.unpack %arg0 inner_dims_pos = [3, 2] inner_tiles = [8, 32] into %arg1 : tensor<?x1x1x1x8x32xf32> -> tensor<?x1x32x8xf32>
  return %0 : tensor<?x1x32x8xf32>
}
// CHECK-LABEL: func.func @unpack_with_dynamic_dims
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK:         %[[C0:.+]] = arith.constant 0 : index
// CHECK:         %[[DIM0_SRC:.+]] = tensor.dim %[[SRC]], %[[C0]] : tensor<?x1x1x1x8x32xf32>
// CHECK:         %[[TILE:.+]] = tensor.extract_slice %[[SRC]][0, 0, 0, 0, 0, 0] [%[[DIM0_SRC]], 1, 1, 1, 8, 32] [1, 1, 1, 1, 1, 1]
// CHECK:         %[[EMPTY:.+]] = tensor.empty(%[[DIM0_SRC]]) : tensor<?x32x8xf32>
// CHECK:         %[[TRANSP:.+]] =  linalg.transpose
// CHECK-SAME:      ins(%[[TILE]] : tensor<?x8x32xf32>)
// CHECK-SAME:      outs(%[[EMPTY]] : tensor<?x32x8xf32>)
// CHECK-SAME:      permutation = [0, 2, 1]
// CHECK:         %[[DIM0_DEST:.+]] = tensor.dim %[[DEST]], %[[C0]] : tensor<?x1x32x8xf32>
// CHECK:         %[[EXTRACT_SLICE:.+]] = tensor.extract_slice %[[TRANSP]][0, 0, 0] [%[[DIM0_DEST]], 32, 8] [1, 1, 1] : tensor<?x32x8xf32> to tensor<?x32x8xf32>
// CHECK:         %[[INSERT:.+]] = tensor.insert_slice %[[EXTRACT_SLICE]] into %[[DEST]]
// CHECK-SAME:      [0, 0, 0, 0] [%[[DIM0_DEST]], 1, 32, 8] [1, 1, 1, 1]
// CHECK:         return %[[INSERT]]
