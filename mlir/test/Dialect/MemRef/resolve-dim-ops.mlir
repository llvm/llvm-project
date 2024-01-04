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
