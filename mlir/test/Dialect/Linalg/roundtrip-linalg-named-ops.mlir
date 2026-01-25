// The following test examples of linalg named ops lowered to linalg.generic and then
// lifted back up to named op.
// RUN: mlir-opt %s -linalg-generalize-named-ops | mlir-opt --linalg-specialize-generic-ops | FileCheck %s

func.func @unary_exp(%A: memref<7x14x21xf32>, %Out: memref<7x14x21xf32>) {
  linalg.exp ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  return
}

// CHECK-LABEL: unary_exp
// CHECK-SAME: %[[A:.+]]: memref<7x14x21xf32>, %[[Out:.+]]: memref<7x14x21xf32>)
// CHECK-NOT: linalg.generic
// CHECK: linalg.exp ins(%[[A]] : memref<7x14x21xf32>) outs(%[[Out]] : memref<7x14x21xf32>)

// -----

func.func @binary_add(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %Out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.add ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>) outs(%Out : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: binary_add
// CHECK-SAME: %[[A:.+]]: tensor<?x?xf32>, %[[B:.+]]: tensor<?x?xf32>,  %[[Out:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.add ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[Out]] : tensor<?x?xf32>) -> tensor<?x?xf32>

// -----


///----------------------------------------------------------------------------------------
/// Tests for linalg.matmul
///----------------------------------------------------------------------------------------

func.func @matmul(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %Out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>) outs(%Out : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: @matmul
// CHECK-SAME: %[[A:.+]]: tensor<?x?xf32>, %[[B:.+]]: tensor<?x?xf32>, %[[Out:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.matmul ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[Out]] : tensor<?x?xf32>) -> tensor<?x?xf32>

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
// CHECK: linalg.matmul {cast = #linalg.type_fn<cast_unsigned>}

// -----

func.func @mixed_named_ops(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>,
                                   %C: tensor<?x?xf32>, %Out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %AB = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>) outs(%Out : tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.add ins(%AB, %C : tensor<?x?xf32>, tensor<?x?xf32>) outs(%Out : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// CHECK-LABEL: @mixed_named_ops
// CHECK-SAME: %[[A:.+]]: tensor<?x?xf32>, %[[B:.+]]: tensor<?x?xf32>, %[[C:.+]]: tensor<?x?xf32>, %[[Out:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: %[[AB:.+]] = linalg.matmul ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[Out]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK: linalg.add ins(%[[AB]], %[[C]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[Out]] : tensor<?x?xf32>) -> tensor<?x?xf32>
