// RUN: mlir-opt -split-input-file -test-tensor-transform-patterns=test-decompose-concat -cse  %s | FileCheck %s

func.func @decompose_dynamic_concat(%arg0 : tensor<8x4xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tensor.concat dim(1) %arg0, %arg1 : (tensor<8x4xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: func @decompose_dynamic_concat(
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<8x4xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>

//   CHECK-DAG:     %[[C8:.+]] = arith.constant 8 : index
//   CHECK-DAG:     %[[C4:.+]] = arith.constant 4 : index
//   CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
//       CHECK:     %[[DIM:.+]] = tensor.dim %[[ARG1]], %[[C1]] : tensor<?x?xf32>
//       CHECK:     %[[CONCAT_SIZE:.+]] = arith.addi %[[DIM]], %[[C4]] : index
//       CHECK:     %[[EMPTY:.+]] = tensor.empty(%[[C8]], %[[CONCAT_SIZE]]) : tensor<?x?xf32>
//       CHECK:     %[[SLICE0:.+]] = tensor.insert_slice %[[ARG0]] into %[[EMPTY]][0, 0] [8, 4] [1, 1] : tensor<8x4xf32> into tensor<?x?xf32>
//       CHECK:     %[[OFFSET:.+]] = tensor.dim %[[ARG1]], %[[C0]] : tensor<?x?xf32>
//       CHECK:     %[[CONCAT:.+]] = tensor.insert_slice %[[ARG1]] into %[[SLICE0]][0, %[[DIM]]] [%[[OFFSET]], %[[DIM]]] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
//       CHECK:     return %[[CONCAT]] : tensor<?x?xf32>

// -----

func.func @decompose_1d_concat(%arg0 : tensor<1xf32>,
                            %arg1 : tensor<2xf32>,
                            %arg2 : tensor<3xf32>,
                            %arg3: tensor<4xf32>) -> tensor<10xf32> {
  %0 = tensor.concat dim(0) %arg0, %arg1, %arg2, %arg3
             : (tensor<1xf32>, tensor<2xf32>, tensor<3xf32>, tensor<4xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}
// CHECK-LABEL: func @decompose_1d_concat
//       CHECK:    tensor.empty() : tensor<10xf32>
//       CHECK:    tensor.insert_slice %{{.*}}[0] [1] [1] : tensor<1xf32> into tensor<10xf32>
//       CHECK:    tensor.insert_slice %{{.*}}[2] [2] [1] : tensor<2xf32> into tensor<10xf32>
//       CHECK:    tensor.insert_slice %{{.*}}[5] [3] [1] : tensor<3xf32> into tensor<10xf32>
//       CHECK:    %[[CONCAT:.+]] = tensor.insert_slice %{{.*}}[9] [4] [1] : tensor<4xf32> into tensor<10xf32>
//       CHECK:    return %[[CONCAT]] : tensor<10xf32>
