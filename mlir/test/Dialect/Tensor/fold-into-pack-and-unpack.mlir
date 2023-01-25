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
