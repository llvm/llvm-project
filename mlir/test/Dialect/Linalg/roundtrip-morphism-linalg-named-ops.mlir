// The following test examples of linalg named ops lowered to linalg.generic
// and then lifted back up to named op.

// RUN: mlir-opt %s -split-input-file -linalg-morph-ops=named-to-generic \
// RUN: | mlir-opt -split-input-file -linalg-morph-ops=generic-to-named \
// RUN: | FileCheck %s

func.func @unary_ops(%A: memref<7x14x21xf32>, %Out: memref<7x14x21xf32>) {
  linalg.exp ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.log ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.abs ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.ceil ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.floor ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.negf ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.reciprocal ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.round ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.sqrt ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.rsqrt ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.square ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.tanh ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  linalg.erf ins(%A : memref<7x14x21xf32>) outs(%Out : memref<7x14x21xf32>)
  return
}

// CHECK-LABEL: unary_ops
// CHECK-SAME: %[[A:.+]]: memref<7x14x21xf32>, %[[OUT:.+]]: memref<7x14x21xf32>)
// CHECK-NOT: linalg.generic
// CHECK: linalg.exp
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.log
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.abs
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.ceil
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.floor
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.negf
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.reciprocal
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.round
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.sqrt
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.rsqrt
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.square
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.tanh
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)
// CHECK: linalg.erf
// CHECK-SAME: ins(%[[A]] : memref<7x14x21xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<7x14x21xf32>)

// -----

func.func @binary_add(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>,
    %Out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.add
    ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%Out : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: binary_add
// CHECK-SAME: %[[A:.+]]: tensor<?x?xf32>, %[[B:.+]]: tensor<?x?xf32>,
// CHECK-SAME: %[[OUT:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.add
// CHECK-SAME: ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME: outs(%[[OUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>

// -----


///----------------------------------------------------------------------------------------
/// Tests for linalg.matmul
///----------------------------------------------------------------------------------------

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

// -----

func.func @mixed_named_ops(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>,
    %C: tensor<?x?xf32>, %Out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %AB = linalg.matmul
    ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%Out : tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.add
    ins(%AB, %C : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%Out : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// CHECK-LABEL: @mixed_named_ops
// CHECK-SAME: %[[A:.+]]: tensor<?x?xf32>, %[[B:.+]]: tensor<?x?xf32>, %[[C:.+]]: tensor<?x?xf32>,
// CHECK-SAME: %[[OUT:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: %[[AB:.+]] = linalg.matmul
// CHECK-SAME: ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME: outs(%[[OUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK: linalg.add
// CHECK-SAME: ins(%[[AB]], %[[C]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME: outs(%[[OUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>
