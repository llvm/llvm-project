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
              
func.func @binary_ops_int(%A: memref<10xi32>, %B: memref<10xi32>,
                          %Out: memref<10xi32>) {
  linalg.add ins(%A, %B : memref<10xi32>, memref<10xi32>) outs(%Out : memref<10xi32>)
  linalg.sub ins(%A, %B : memref<10xi32>, memref<10xi32>) outs(%Out : memref<10xi32>)
  linalg.mul ins(%A, %B : memref<10xi32>, memref<10xi32>) outs(%Out : memref<10xi32>)
  linalg.div ins(%A, %B : memref<10xi32>, memref<10xi32>) outs(%Out : memref<10xi32>)
  linalg.div_unsigned ins(%A, %B : memref<10xi32>, memref<10xi32>) outs(%Out : memref<10xi32>)
  linalg.max ins(%A, %B : memref<10xi32>, memref<10xi32>) outs(%Out : memref<10xi32>)
  linalg.min ins(%A, %B : memref<10xi32>, memref<10xi32>) outs(%Out : memref<10xi32>)
  return
}

// CHECK-LABEL: binary_ops_int
// CHECK-SAME: %[[A:.+]]: memref<10xi32>, %[[B:.+]]: memref<10xi32>,
// CHECK-SAME: %[[OUT:.+]]: memref<10xi32>)
// CHECK-NOT: linalg.generic
// CHECK: linalg.add
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xi32>, memref<10xi32>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xi32>)
// CHECK: linalg.sub
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xi32>, memref<10xi32>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xi32>)
// CHECK: linalg.mul
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xi32>, memref<10xi32>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xi32>)
// CHECK: linalg.div
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xi32>, memref<10xi32>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xi32>)
// CHECK: linalg.div_unsigned
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xi32>, memref<10xi32>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xi32>)
// CHECK: linalg.max
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xi32>, memref<10xi32>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xi32>)
// CHECK: linalg.min
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xi32>, memref<10xi32>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xi32>)

// -----
              
func.func @binary_ops_float(%A: memref<10xf32>, %B: memref<10xf32>,
                            %Out: memref<10xf32>) {
  linalg.add ins(%A, %B : memref<10xf32>, memref<10xf32>) outs(%Out : memref<10xf32>)
  linalg.sub ins(%A, %B : memref<10xf32>, memref<10xf32>) outs(%Out : memref<10xf32>)
  linalg.mul ins(%A, %B : memref<10xf32>, memref<10xf32>) outs(%Out : memref<10xf32>)
  linalg.div ins(%A, %B : memref<10xf32>, memref<10xf32>) outs(%Out : memref<10xf32>)
  linalg.max ins(%A, %B : memref<10xf32>, memref<10xf32>) outs(%Out : memref<10xf32>)
  linalg.min ins(%A, %B : memref<10xf32>, memref<10xf32>) outs(%Out : memref<10xf32>)
  linalg.powf ins(%A, %B : memref<10xf32>, memref<10xf32>) outs(%Out : memref<10xf32>)
  return
}

// CHECK-LABEL: binary_ops_float
// CHECK-SAME: %[[A:.+]]: memref<10xf32>, %[[B:.+]]: memref<10xf32>,
// CHECK-SAME: %[[OUT:.+]]: memref<10xf32>)
// CHECK-NOT: linalg.generic
// CHECK: linalg.add
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xf32>, memref<10xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xf32>)
// CHECK: linalg.sub
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xf32>, memref<10xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xf32>)
// CHECK: linalg.mul
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xf32>, memref<10xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xf32>)
// CHECK: linalg.div
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xf32>, memref<10xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xf32>)
// CHECK: linalg.max
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xf32>, memref<10xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xf32>)
// CHECK: linalg.min
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xf32>, memref<10xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xf32>)
// CHECK: linalg.powf
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xf32>, memref<10xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xf32>)

// -----
              
func.func @binary_ops_complex(%A: memref<10xcomplex<f32>>, %B: memref<10xcomplex<f32>>,
                              %Out: memref<10xcomplex<f32>>) {
  linalg.add ins(%A, %B : memref<10xcomplex<f32>>, memref<10xcomplex<f32>>)
             outs(%Out : memref<10xcomplex<f32>>)
  linalg.sub ins(%A, %B : memref<10xcomplex<f32>>, memref<10xcomplex<f32>>)
             outs(%Out : memref<10xcomplex<f32>>)
  linalg.mul ins(%A, %B : memref<10xcomplex<f32>>, memref<10xcomplex<f32>>)
             outs(%Out : memref<10xcomplex<f32>>)
  linalg.div ins(%A, %B : memref<10xcomplex<f32>>, memref<10xcomplex<f32>>)
             outs(%Out : memref<10xcomplex<f32>>)
  return
}

// CHECK-LABEL: binary_ops_complex
// CHECK-SAME: %[[A:.+]]: memref<10xcomplex<f32>>, %[[B:.+]]: memref<10xcomplex<f32>>,
// CHECK-SAME: %[[OUT:.+]]: memref<10xcomplex<f32>>)
// CHECK-NOT: linalg.generic
// CHECK: linalg.add
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xcomplex<f32>>, memref<10xcomplex<f32>>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xcomplex<f32>>)
// CHECK: linalg.sub
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xcomplex<f32>>, memref<10xcomplex<f32>>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xcomplex<f32>>)
// CHECK: linalg.mul
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xcomplex<f32>>, memref<10xcomplex<f32>>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xcomplex<f32>>)
// CHECK: linalg.div
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xcomplex<f32>>, memref<10xcomplex<f32>>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xcomplex<f32>>)

// -----
              
func.func @binary_ops_bool(%A: memref<10xi1>, %B: memref<10xi1>,
                           %Out: memref<10xi1>) {
  linalg.add ins(%A, %B : memref<10xi1>, memref<10xi1>) outs(%Out : memref<10xi1>)
  linalg.mul ins(%A, %B : memref<10xi1>, memref<10xi1>) outs(%Out : memref<10xi1>)
  return
}

// CHECK-LABEL: binary_ops_bool
// CHECK-SAME: %[[A:.+]]: memref<10xi1>, %[[B:.+]]: memref<10xi1>,
// CHECK-SAME: %[[OUT:.+]]: memref<10xi1>)
// CHECK-NOT: linalg.generic
// CHECK: linalg.add
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xi1>, memref<10xi1>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xi1>)
// CHECK: linalg.mul
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<10xi1>, memref<10xi1>)
// CHECK-SAME: outs(%[[OUT]] : memref<10xi1>)

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
