// The following test examples of linalg named ops lowered to linalg.generic
// and then lifted back up to named op.

// RUN: mlir-opt %s -split-input-file -linalg-morph-ops=named-to-generic \
// RUN: | mlir-opt -split-input-file -linalg-morph-ops=generic-to-named \
// RUN: | FileCheck %s

func.func @matmul(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>,
    %Out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul
    ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%Out : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: @matmul
// CHECK-SAME: %[[A:.+]]: tensor<?x?xf32>, %[[B:.+]]: tensor<?x?xf32>,
// CHECK-SAME: %[[OUT:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.matmul
// CHECK-SAME: ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME: outs(%[[OUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>

// -----

func.func @matmul_bool(%A: tensor<?x?xi1>, %B: tensor<?x?xi1>,
    %Out: tensor<?x?xi1>) -> tensor<?x?xi1> {
  %0 = linalg.matmul
    ins(%A, %B : tensor<?x?xi1>, tensor<?x?xi1>)
    outs(%Out : tensor<?x?xi1>) -> tensor<?x?xi1>
  return %0 : tensor<?x?xi1>
}

// CHECK-LABEL: @matmul_bool
// CHECK-SAME: %[[A:.+]]: tensor<?x?xi1>, %[[B:.+]]: tensor<?x?xi1>,
// CHECK-SAME: %[[OUT:.+]]: tensor<?x?xi1>) -> tensor<?x?xi1>
// CHECK-NOT: linalg.generic
// CHECK: linalg.matmul
// CHECK-SAME: ins(%[[A]], %[[B]] : tensor<?x?xi1>, tensor<?x?xi1>)
// CHECK-SAME: outs(%[[OUT]] : tensor<?x?xi1>) -> tensor<?x?xi1>

// -----

// Check matmul with unsigned cast is correctly raised back to named op.
func.func @matmul_unsigned_cast(%A: tensor<16x8xi16>, %B: tensor<8x32xi64>,
    %Out: tensor<16x32xi32>) -> tensor<16x32xi32> {
  %0 = linalg.matmul {cast = #linalg.type_fn<cast_unsigned>}
    ins(%A, %B : tensor<16x8xi16>, tensor<8x32xi64>)
    outs(%Out : tensor<16x32xi32>) -> tensor<16x32xi32>
  return %0 : tensor<16x32xi32>
}

// CHECK-LABEL: @matmul_unsigned_cast
// CHECK-NOT: linalg.generic
// CHECK: linalg.matmul
// CHECK-SAME: {cast = #linalg.type_fn<cast_unsigned>}

