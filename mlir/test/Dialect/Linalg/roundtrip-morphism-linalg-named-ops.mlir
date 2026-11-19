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

// -----

func.func @reduce_1d_sum(%A: tensor<?xf32>, %Out: tensor<f32>) -> tensor<f32> {
  %0 = linalg.reduce { arith.addf }
    ins(%A : tensor<?xf32>)
    outs(%Out : tensor<f32>)
    dimensions = [0]
  return %0 : tensor<f32>
}

// CHECK-LABEL: @reduce_1d_sum
// CHECK-SAME: %[[A:.+]]: tensor<?xf32>, %[[OUT:.+]]: tensor<f32>) -> tensor<f32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.reduce { arith.addf }
// CHECK-SAME: ins(%[[A]] : tensor<?xf32>)
// CHECK-SAME: outs(%[[OUT]] : tensor<f32>)
// CHECK-SAME: dimensions = [0]

// -----

func.func @reduce_2d_inner_mul(%A: tensor<?x?xf32>, %Out: tensor<?xf32>) -> tensor<?xf32> {
  %0 = linalg.reduce { arith.mulf }
    ins(%A : tensor<?x?xf32>)
    outs(%Out : tensor<?xf32>)
    dimensions = [1]
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: @reduce_2d_inner_mul
// CHECK-SAME: %[[A:.+]]: tensor<?x?xf32>, %[[OUT:.+]]: tensor<?xf32>) -> tensor<?xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.reduce { arith.mulf }
// CHECK-SAME: ins(%[[A]] : tensor<?x?xf32>)
// CHECK-SAME: outs(%[[OUT]] : tensor<?xf32>)
// CHECK-SAME: dimensions = [1]

// -----

func.func @reduce_3d_multi_dims_max(%A: tensor<?x?x?xi32>, %Out: tensor<?xi32>) -> tensor<?xi32> {
  %0 = linalg.reduce { arith.maxsi }
    ins(%A : tensor<?x?x?xi32>)
    outs(%Out : tensor<?xi32>)
    dimensions = [0, 2]
  return %0 : tensor<?xi32>
}

// CHECK-LABEL: @reduce_3d_multi_dims_max
// CHECK-SAME: %[[A:.+]]: tensor<?x?x?xi32>, %[[OUT:.+]]: tensor<?xi32>) -> tensor<?xi32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.reduce { arith.maxsi }
// CHECK-SAME: ins(%[[A]] : tensor<?x?x?xi32>)
// CHECK-SAME: outs(%[[OUT]] : tensor<?xi32>)
// CHECK-SAME: dimensions = [0, 2]

// -----

// Long-form combiner body with a cast is preserved through the roundtrip.
func.func @reduce_with_cast_body(%A: tensor<?x?xf16>, %Out: tensor<?xf32>) -> tensor<?xf32> {
  %0 = linalg.reduce
    ins(%A : tensor<?x?xf16>)
    outs(%Out : tensor<?xf32>)
    dimensions = [1]
    (%in: f16, %acc: f32) {
      %1 = arith.extf %in : f16 to f32
      %2 = arith.addf %acc, %1 : f32
      linalg.yield %2 : f32
    }
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: @reduce_with_cast_body
// CHECK-SAME: %[[A:.+]]: tensor<?x?xf16>, %[[OUT:.+]]: tensor<?xf32>) -> tensor<?xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.reduce ins(%[[A]] : tensor<?x?xf16>)
// CHECK-SAME: outs(%[[OUT]] : tensor<?xf32>)
// CHECK-SAME: dimensions = [1]
// CHECK-NEXT: (%[[IN:.+]]: f16, %[[ACC:.+]]: f32) {
// CHECK-NEXT: %[[EXT:.+]] = arith.extf %[[IN]] : f16 to f32
// CHECK-NEXT: %[[ADD:.+]] = arith.addf %[[ACC]], %[[EXT]] : f32
// CHECK-NEXT: linalg.yield %[[ADD]] : f32

// -----

func.func @reduce_memref(%A: memref<?x?xf32>, %Out: memref<?xf32>) {
  linalg.reduce { arith.addf }
    ins(%A : memref<?x?xf32>)
    outs(%Out : memref<?xf32>)
    dimensions = [0]
  return
}

// CHECK-LABEL: @reduce_memref
// CHECK-SAME: %[[A:.+]]: memref<?x?xf32>, %[[OUT:.+]]: memref<?xf32>)
// CHECK-NOT: linalg.generic
// CHECK: linalg.reduce { arith.addf }
// CHECK-SAME: ins(%[[A]] : memref<?x?xf32>)
// CHECK-SAME: outs(%[[OUT]] : memref<?xf32>)
// CHECK-SAME: dimensions = [0]

// -----

// Multiple inputs/inits (variadic reduce) are preserved through the roundtrip.
func.func @reduce_multi_input(%A: tensor<?x?xf32>, %B: tensor<?x?xi32>,
    %OutA: tensor<?xf32>, %OutB: tensor<?xi32>) -> (tensor<?xf32>, tensor<?xi32>) {
  %0:2 = linalg.reduce
    ins(%A, %B : tensor<?x?xf32>, tensor<?x?xi32>)
    outs(%OutA, %OutB : tensor<?xf32>, tensor<?xi32>)
    dimensions = [1]
    (%a: f32, %b: i32, %accA: f32, %accB: i32) {
      %1 = arith.addf %accA, %a : f32
      %2 = arith.addi %accB, %b : i32
      linalg.yield %1, %2 : f32, i32
    }
  return %0#0, %0#1 : tensor<?xf32>, tensor<?xi32>
}

// CHECK-LABEL: @reduce_multi_input
// CHECK-SAME: %[[A:.+]]: tensor<?x?xf32>, %[[B:.+]]: tensor<?x?xi32>,
// CHECK-SAME: %[[OUTA:.+]]: tensor<?xf32>, %[[OUTB:.+]]: tensor<?xi32>)
// CHECK-NOT: linalg.generic
// CHECK: linalg.reduce ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xi32>)
// CHECK-SAME: outs(%[[OUTA]], %[[OUTB]] : tensor<?xf32>, tensor<?xi32>)
// CHECK-SAME: dimensions = [1]
// CHECK-NEXT: (%[[IA:.+]]: f32, %[[IB:.+]]: i32, %[[ACCA:.+]]: f32, %[[ACCB:.+]]: i32) {
// CHECK-NEXT: %[[ADDF:.+]] = arith.addf %[[ACCA]], %[[IA]] : f32
// CHECK-NEXT: %[[ADDI:.+]] = arith.addi %[[ACCB]], %[[IB]] : i32
// CHECK-NEXT: linalg.yield %[[ADDF]], %[[ADDI]] : f32, i32

