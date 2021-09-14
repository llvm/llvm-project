// RUN: mlir-opt %s -test-linalg-transform-patterns="test-tile-scalarize-dynamic-dims" -for-loop-canonicalization -canonicalize -split-input-file | \
// RUN:     FileCheck %s

// CHECK-LABEL: func @matmul_partly_dynamic_tensor(
//  CHECK-SAME:     %[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x2000xf32>
//   CHECK-DAG:   %[[C0:.*]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = constant 1 : index
//       CHECK:   tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?xf32>
//       CHECK:   %[[UB1:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?xf32>
//       CHECK:   %[[UB2:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?xf32>
//       CHECK:   scf.for %[[IV0:.*]] = %[[C0]] to %[[UB1]] step %[[C1]]
//       CHECK:     scf.for %[[IV1:.*]] = %[[C0]] to %[[UB2]] step %[[C1]]
//       CHECK:       %[[S1:.*]] = tensor.extract_slice %[[ARG0]][%[[IV0]], %[[IV1]]] [1, 1] [1, 1] : tensor<?x?xf32> to tensor<1x1xf32>
//       CHECK:       %[[S2:.*]] = tensor.extract_slice %[[ARG1]][%[[IV1]], 0] [1, 2000] [1, 1] : tensor<?x2000xf32> to tensor<1x2000xf32>
//       CHECK:       %[[S3:.*]] = tensor.extract_slice %{{.*}}[%[[IV0]], 0] [1, 2000] [1, 1] : tensor<?x2000xf32> to tensor<1x2000xf32>
//       CHECK:       linalg.matmul ins(%[[S1]], %[[S2]] : tensor<1x1xf32>, tensor<1x2000xf32>) outs(%[[S3]] : tensor<1x2000xf32>) -> tensor<1x2000xf32>
func @matmul_partly_dynamic_tensor(%arg0: tensor<?x?xf32>, %arg1: tensor<?x2000xf32>)
    -> tensor<?x2000xf32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %out = linalg.init_tensor [%d0, 2000] : tensor<?x2000xf32>
  %r = linalg.matmul {__internal_linalg_transform__ = "tile"}
      ins(%arg0, %arg1: tensor<?x?xf32>, tensor<?x2000xf32>)
      outs(%out: tensor<?x2000xf32>) -> tensor<?x2000xf32>
  return %r : tensor<?x2000xf32>
}
