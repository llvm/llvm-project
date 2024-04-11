// RUN: mlir-opt --resolve-ranked-shaped-type-result-dims --split-input-file %s | FileCheck %s

// CHECK-LABEL: func @dim_out_of_bounds(
//  CHECK-NEXT:   arith.constant
//  CHECK-NEXT:   memref.dim
//  CHECK-NEXT:   return
func.func @dim_out_of_bounds(%m : memref<7x8xf32>) -> index {
  %idx = arith.constant 7 : index
  %0 = memref.dim %m, %idx : memref<7x8xf32>
  return %0 : index
}

// -----

// CHECK-LABEL: func @dim_out_of_bounds_2(
//  CHECK-NEXT:   arith.constant
//  CHECK-NEXT:   arith.constant
//  CHECK-NEXT:   bufferization.alloc_tensor
//  CHECK-NEXT:   tensor.dim
//  CHECK-NEXT:   return
func.func @dim_out_of_bounds_2(%idx1 : index, %idx2 : index) -> index {
  %idx = arith.constant 7 : index
  %sz = arith.constant 5 : index
  %alloc = bufferization.alloc_tensor(%sz, %sz) : tensor<?x?xf32>
  %0 = tensor.dim %alloc, %idx : tensor<?x?xf32>
  return %0 : index
}

// -----

// Test case: Folding of memref.dim(memref.expand_shape)
// CHECK-LABEL: func @dim_of_memref_expand_shape(
//  CHECK-SAME:     %[[MEM:[0-9a-z]+]]: memref<?x8xi32>
//  CHECK-NEXT:   %[[IDX:.*]] = arith.constant 0
//  CHECK-NEXT:   %[[DIM:.*]] = memref.dim %[[MEM]], %[[IDX]] : memref<?x8xi32>
//       CHECK:   return %[[DIM]] : index
func.func @dim_of_memref_expand_shape(%arg0: memref<?x8xi32>)
    -> index {
  %c1 = arith.constant 1 : index
  %0 = memref.expand_shape %arg0 [[0, 1], [2, 3]]: memref<?x8xi32> into memref<1x?x2x4xi32>
  %1 = memref.dim %0, %c1 : memref<1x?x2x4xi32>
  return %1 : index
}

// -----

// Test case: Folding of memref.dim(memref.collapse_shape)
// CHECK-LABEL: func @dim_of_memref_collapse_shape(
//  CHECK-SAME:     %[[MEM:[0-9a-z]+]]: memref<1x?x2x4xi32>
//  CHECK-NEXT:   %[[IDX:.*]] = arith.constant 1
//  CHECK-NEXT:   %[[DIM:.*]] = memref.dim %[[MEM]], %[[IDX]] : memref<1x?x2x4xi32>
//       CHECK:   return %[[DIM]] : index
func.func @dim_of_memref_collapse_shape(%arg0: memref<1x?x2x4xi32>)
    -> index {
  %c0 = arith.constant 0 : index
  %0 = memref.collapse_shape %arg0 [[0, 1], [2, 3]]: memref<1x?x2x4xi32> into memref<?x8xi32>
  %1 = memref.dim %0, %c0 : memref<?x8xi32>
  return %1 : index
}
