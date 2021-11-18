// RUN: mlir-opt -test-linalg-transform-patterns=test-tile-pattern %s | FileCheck %s

func @matmul_zero_tile(
  %arg0: tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul {__internal_linalg_transform__ = "tile"}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: matmul_zero_tile
//       CHECK:   linalg.matmul
//   CHECK-NOT:   __internal_linalg_transform__
