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

func.func @foo(%arg0: tensor<56x57x1x64xf32>) -> tensor<1x2x56x57x32xf32> {
  %0 = tensor.empty() : tensor<1x56x57x64xf32>
  %transposed = linalg.transpose
    ins(%arg0 : tensor<56x57x1x64xf32>)
    outs(%0 : tensor<1x56x57x64xf32>)
    permutation = [2, 0, 1, 3]
  %1 = tensor.empty() : tensor<1x2x56x57x32xf32>

  // [2, 3, 0, 1]

  %pack = tensor.pack %transposed
    outer_dims_perm = [0, 3, 1, 2]
    inner_dims_pos = [3]
    inner_tiles = [32]
    into %1 : tensor<1x56x57x64xf32> -> tensor<1x2x56x57x32xf32>
  return %pack : tensor<1x2x56x57x32xf32>
}


func.func @foo1(%arg0: tensor<56x57x1x64xf32>) -> tensor<1x2x56x57x32xf32> {
  %0 = tensor.empty() : tensor<56x57x1x2x32xf32>
  %pack = tensor.pack %arg0
    outer_dims_perm = [0, 1, 2, 3]
    inner_dims_pos = [3]
    inner_tiles = [32]
    into %0 : tensor<56x57x1x64xf32> -> tensor<56x57x1x2x32xf32>

    // [2, 3, 0, 1]

    %1 = tensor.empty() : tensor<1x2x56x57x32xf32>
    %transposed = linalg.transpose
    ins(%pack : tensor<56x57x1x2x32xf32>)
    outs(%1 : tensor<1x2x56x57x32xf32>)
    permutation = [2, 3, 0, 1, 4]
  return %transposed : tensor<1x2x56x57x32xf32>
}
