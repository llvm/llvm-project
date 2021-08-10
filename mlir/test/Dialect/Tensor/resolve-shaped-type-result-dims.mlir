// RUN: mlir-opt -resolve-shaped-type-result-dims -split-input-file %s | FileCheck %s

func @insert_slice(
    %arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?x?x?xf32>,
    %arg2 : index, %arg3 : index, %arg4 : index) -> (index, index, index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %d2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %0 = tensor.insert_slice %arg0 into %arg1[%arg2, %arg3, %arg4] [%d0, %d1, %d2] [1, 1, 1] : tensor<?x?x?xf32> into tensor<?x?x?xf32>
  %1 = tensor.dim %0, %c0 : tensor<?x?x?xf32>
  %2 = tensor.dim %0, %c1 : tensor<?x?x?xf32>
  %3 = tensor.dim %0, %c2 : tensor<?x?x?xf32>
  return %1, %2, %3 : index, index, index
}
// CHECK-LABEL: func @insert_slice(
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = constant 2 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//   CHECK-DAG:   %[[D2:.+]] = tensor.dim %[[ARG1]], %[[C2]]
//       CHECK:   return %[[D0]], %[[D1]], %[[D2]]
