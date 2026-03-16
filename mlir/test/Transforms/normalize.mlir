// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(normalize))" -split-input-file | FileCheck %s

// CHECK-LABEL: func @multiple_memref_store
//  CHECK-SAME:   %[[ARG0:.*]]: index,
//  CHECK-SAME:   %[[ARG1:.*]]: memref<?xf32>
func.func @multiple_memref_store(%arg0: index, %arg1 : memref<?xf32>) {
  %f0 = arith.constant 0.0 : f32
  %f1 = arith.constant 1.0 : f32
  %add = arith.addi %arg0, %arg0 : index
  %sub = arith.subi %arg0, %arg0 : index
  memref.store %f0, %arg1[%add] : memref<?xf32>
  memref.store %f1, %arg1[%sub] : memref<?xf32>
  return
}

// CHECK-NEXT: %[[C0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT: %[[ADD:.*]] = arith.addi %[[ARG0]], %[[ARG0]] : index
// CHECK-NEXT: memref.store %[[C0]], %[[ARG1]]{{\[}}%[[ADD]]] : memref<?xf32>
// CHECK-NEXT: %[[C1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-NEXT: %[[SUB:.*]] = arith.subi %[[ARG0]], %[[ARG0]] : index
// CHECK-NEXT: memref.store %[[C1]], %[[ARG1]]{{\[}}%[[SUB]]] : memref<?xf32>

// -----

// CHECK-LABEL: func @return_multiple_operands
//  CHECK-SAME:   %[[ARG0:.*]]: index
func.func @return_multiple_operands (%arg0: index) -> (index, index) {
  %0 = arith.addi %arg0, %arg0 : index
  %1 = arith.subi %arg0, %arg0 : index
  return %1, %0 : index, index
}

// CHECK-NEXT: %[[SUB:.*]] = arith.subi %[[ARG0]], %[[ARG0]] : index
// CHECK-NEXT: %[[ADD:.*]] = arith.addi %[[ARG0]], %[[ARG0]] : index
// CHECK-NEXT: return %[[SUB]], %[[ADD]] : index, index

// -----

// CHECK-LABEL: func @cross_region
//  CHECK-SAME:   %[[ARG0:.*]]: f32,
//  CHECK-SAME:   %[[ARG1:.*]]: memref<10xf32>
func.func @cross_region(%arg0: f32, %arg1 : memref<10xf32>) {
  %add = arith.addf %arg0, %arg0 : f32
  affine.for %i = 0 to 5 {
    memref.store %add, %arg1[%i] : memref<10xf32>
  }
  %exp = math.log2 %add : f32
  affine.for %i = 6 to 10 {
    memref.store %exp, %arg1[%i] : memref<10xf32>
  } 
  return
}

//      CHECK: affine.for %[[IV:.*]] = 6 to 10 {
// CHECK-NEXT:   %[[LOG:.*]] = math.log2
// CHECK-NEXT:   memref.store %[[LOG]], %[[ARG1]]{{\[}}%[[IV]]] : memref<10xf32>
// CHECK-NEXT: }

// -----

// This test verifies the reordering of scf.for ops.
// The memref.store within the scf.for causes the loop to have side effects.
// The lower bound of the scf.for remains in its original position
// because the upper bound depends on it, but the step has been reordered.

// CHECK-LABEL: func @side_effect_loop_op
//  CHECK-SAME:   %[[ARG0:.*]]: memref<?xf32>
func.func @side_effect_loop_op(%arg1 : memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %upper = memref.dim %arg1, %c0 : memref<?xf32>
  %f1 = arith.constant 1.0 : f32
  scf.for %i = %c0 to %upper step %c1 {
    memref.store %f1, %arg1[%i] : memref<?xf32>
  }
  return
}

// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[DIM:.*]] = memref.dim %[[ARG0]], %[[C0]] : memref<?xf32>
// CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT: scf.for %[[IV:.*]] = %[[C0]] to %[[DIM]] step %[[C1]] {
// CHECK-NEXT:   %[[F1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:   memref.store %[[F1]], %[[ARG0]]{{\[}}%[[IV]]] : memref<?xf32>
// CHECK-NEXT: }
