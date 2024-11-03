// RUN: mlir-opt -split-input-file --test-linalg-transform-patterns="test-generalize-tensor-unpack"  %s | FileCheck %s

func.func @simple_KCRSsr_to_KCRS(%arg0: tensor<1x1x1x1x8x32xf32>, %arg1: tensor<1x1x32x8xf32>) -> tensor<1x1x32x8xf32> {
  %0 = tensor.unpack %arg0 inner_dims_pos = [3, 2] inner_tiles = [8, 32] into %arg1 : tensor<1x1x1x1x8x32xf32> -> tensor<1x1x32x8xf32>
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

func.func @simple_unpack_and_extract_slice(%input: tensor<1x1x8x2xf32>, %output: tensor<5x1xf32>) -> tensor<5x1xf32> {
  %0 = tensor.unpack %input inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %output : tensor<1x1x8x2xf32> -> tensor<5x1xf32>
  return %0 : tensor<5x1xf32>
}
// CHECK-LABEL: func.func @simple_unpack_and_extract_slice
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK:         %[[TILE:.+]] = tensor.extract_slice %[[SRC]][0, 0, 0, 0] [1, 1, 8, 2] [1, 1, 1, 1]
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<8x2xf32>
// CHECK:         %[[TRANSP:.+]] =  linalg.transpose
// CHECK-SAME:      ins(%[[TILE]] : tensor<8x2xf32>)
// CHECK-SAME:      outs(%[[EMPTY]] : tensor<8x2xf32>)
// CHECK-SAME:      permutation = [0, 1]
//                They have the same type, so the insert_slice op is folded
//                away.
// CHECK:         %[[SLICE:.+]] = tensor.extract_slice %[[TRANSP]][0, 0] [5, 1] [1, 1]
// CHECK:         return %[[SLICE]]

// -----

func.func @simple_CNnc_to_NC(%arg0: tensor<1x1x32x8xf32>, %arg1: tensor<32x8xf32>) -> tensor<32x8xf32>{
  %0 = tensor.unpack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %arg1 : tensor<1x1x32x8xf32> -> tensor<32x8xf32>
  return %0 : tensor<32x8xf32>
}
// CHECK-LABEL: func.func @simple_CNnc_to_NC
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK:         %[[TILE:.+]] = tensor.extract_slice %[[SRC]][0, 0, 0, 0] [1, 1, 32, 8] [1, 1, 1, 1]
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<32x8xf32>
// CHECK:         %[[TRANSP:.+]] =  linalg.transpose
// CHECK-SAME:      ins(%[[TILE]] : tensor<32x8xf32>)
// CHECK-SAME:      outs(%[[EMPTY]] : tensor<32x8xf32>)
// CHECK-SAME:      permutation = [0, 1]
//                They have the same type, so the insert_slice op is folded
//                away.
// CHECK:         return %[[TRANSP]]
