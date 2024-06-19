// RUN: mlir-opt %s -convert-vector-to-arm-sme -split-input-file -allow-unregistered-dialect -verify-diagnostics | FileCheck %s

//===----------------------------------------------------------------------===//
// vector.transfer_read
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @transfer_read_2d__bad_type
// CHECK-NOT: arm_sme.tile_load
// CHECK: vector.transfer_read
func.func @transfer_read_2d__bad_type(%src : memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f64
  %0 = vector.transfer_read %src[%c0, %c0], %pad {permutation_map = affine_map<(d0, d1) -> (d1, d0)>, in_bounds = [false, false]} : memref<?x?xf64>, vector<[4]x[4]xf64>
  "prevent.dce"(%0) : (vector<[4]x[4]xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: @transfer_read_2d__non_memref_type
// CHECK-NOT: arm_sme.tile_load
// CHECK: vector.transfer_read
func.func @transfer_read_2d__non_memref_type(%src : tensor<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f64
  %0 = vector.transfer_read %src[%c0, %c0], %pad {permutation_map = affine_map<(d0, d1) -> (d1, d0)>, in_bounds = [true, true]} : tensor<?x?xf64>, vector<[2]x[2]xf64>
  "prevent.dce"(%0) : (vector<[2]x[2]xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: @transfer_read_2d__bad_transfer_rank
// CHECK-NOT: arm_sme.tile_load
// CHECK: vector.transfer_read
func.func @transfer_read_2d__bad_transfer_rank(%src : memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f64
  %0 = vector.transfer_read %src[%c0, %c0], %pad {permutation_map = affine_map<(d0, d1) -> (d0)>, in_bounds = [true]} : memref<?x?xf64>, vector<[2]xf64>
  "prevent.dce"(%0) : (vector<[2]xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: @transfer_read_2d__non_transpose
// CHECK-NOT: arm_sme.tile_load
// CHECK: vector.transfer_read
func.func @transfer_read_2d__non_transpose(%src : memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f64
  %0 = vector.transfer_read %src[%c0, %c0], %pad {permutation_map = affine_map<(d0, d1) -> (d0, 0)>, in_bounds = [true, true]} : memref<?x?xf64>, vector<[2]x[2]xf64>
  "prevent.dce"(%0) : (vector<[2]x[2]xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: @transfer_read_2d__out_of_bounds
// CHECK-NOT: arm_sme.tile_load
// CHECK: vector.transfer_read
func.func @transfer_read_2d__out_of_bounds(%src : memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f64
  %0 = vector.transfer_read %src[%c0, %c0], %pad {permutation_map = affine_map<(d0, d1) -> (d1, d0)>, in_bounds = [false, false]} : memref<?x?xf64>, vector<[2]x[2]xf64>
  "prevent.dce"(%0) : (vector<[2]x[2]xf64>) -> ()
  return
}

//===----------------------------------------------------------------------===//
// vector.transfer_write
//===----------------------------------------------------------------------===//

// -----

// The following tests check the 'vector.transfer_write' -> 'arm_sme.intr.zero'
// lowering only occurs for vector types of correct rank, shape, element size
// and number of scalable dims.

// CHECK-LABEL: @transfer_write_2d_zero__bad_type
// CHECK: vector.transfer_write
// CHECK-NOT: arm_sme.intr.zero
func.func @transfer_write_2d_zero__bad_type(%arg0 : memref<?x?xi4>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0> : vector<[16]x[16]xi4>
  vector.transfer_write %cst, %arg0[%c0, %c0] {in_bounds = [true, true]} : vector<[16]x[16]xi4>, memref<?x?xi4>
  return
}

// -----

// CHECK-LABEL: @transfer_write_2d_zero__bad_shape
// CHECK: vector.transfer_write
// CHECK-NOT: arm_sme.tile_store
func.func @transfer_write_2d_zero__bad_shape(%arg0 : memref<?x?xi8>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0> : vector<[8]x[8]xi8>
  vector.transfer_write %cst, %arg0[%c0, %c0] {in_bounds = [true, true]} : vector<[8]x[8]xi8>, memref<?x?xi8>
  return
}

// -----

// CHECK-LABEL: @transfer_write_2d_zero__bad_rank
// CHECK: vector.transfer_write
// CHECK-NOT: arm_sme.tile_store
func.func @transfer_write_2d_zero__bad_rank(%arg0 : memref<?x?x?xi8>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0> : vector<[16]x[16]x[16]xi8>
  vector.transfer_write %cst, %arg0[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<[16]x[16]x[16]xi8>, memref<?x?x?xi8>
  return
}

// -----

// CHECK-LABEL: @transfer_write_2d_zero__non_memref_type
// CHECK: vector.transfer_write
// CHECK-NOT: arm_sme.tile_store
func.func @transfer_write_2d_zero__non_memref_type(%arg0 : tensor<?x?xi8>) -> tensor<?x?xi8> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0> : vector<[16]x[16]xi8>
  %0 = vector.transfer_write %cst, %arg0[%c0, %c0] {in_bounds = [true, true]} : vector<[16]x[16]xi8>, tensor<?x?xi8>
  return %0 : tensor<?x?xi8>
}

// -----

// CHECK-LABEL: @transfer_write_2d__fixed
// CHECK: vector.transfer_write
// CHECK-NOT: arm_sme.tile_store
func.func @transfer_write_2d__fixed(%vector : vector<16x16xi8>, %dest : memref<?x?xi8>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %vector, %dest[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xi8>, memref<?x?xi8>
  return
}

// -----

// CHECK-LABEL: @transfer_write_2d__out_of_bounds
// CHECK: vector.transfer_write
// CHECK-NOT: arm_sme.tile_store
func.func @transfer_write_2d__out_of_bounds(%vector : vector<[4]x[4]xf32>, %dest : memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %vector, %dest[%c0, %c0] {in_bounds = [false, false]} : vector<[4]x[4]xf32>, memref<?x?xf32>
  return
}

// -----

// CHECK-LABEL: func.func @transfer_write_slice_unsupported_permutation
// CHECK-NOT: arm_sme.store_tile_slice
func.func @transfer_write_slice_unsupported_permutation(%vector: vector<[4]x[4]xf32>, %dest : memref<?x?xf32>, %slice_index: index) {
  %c0 = arith.constant 0 : index
  %slice = vector.extract %vector[%slice_index] : vector<[4]xf32> from vector<[4]x[4]xf32>
  vector.transfer_write %slice, %dest[%slice_index, %c0] { permutation_map = affine_map<(d0, d1) -> (d0)>, in_bounds = [true] }: vector<[4]xf32>, memref<?x?xf32>
  return
}


//===----------------------------------------------------------------------===//
// vector.outerproduct
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @vector_outerproduct_unsupported_axpy
// CHECK-NOT: arm_sme.outerproduct
// CHECK:     vector.outerproduct
func.func @vector_outerproduct_unsupported_axpy(%lhs : vector<[2]xf64>, %rhs : f64, %acc : vector<[2]xf64>) -> vector<[2]xf64> {
  %0 = vector.outerproduct %lhs, %rhs, %acc {kind = #vector.kind<mul>} : vector<[2]xf64>, f64
  return %0 : vector<[2]xf64>
}

// -----

// CHECK-LABEL: @vector_outerproduct_unsupported_kind
// CHECK-NOT: arm_sme.outerproduct
// CHECK:     vector.outerproduct
func.func @vector_outerproduct_unsupported_kind(%lhs : vector<[2]xf64>, %rhs : vector<[2]xf64>) {
  %acc = arm_sme.get_tile : vector<[2]x[2]xf64>
  %0 = vector.outerproduct %lhs, %rhs, %acc {kind = #vector.kind<mul>} : vector<[2]xf64>, vector<[2]xf64>
  "prevent.dce"(%0) : (vector<[2]x[2]xf64>) -> ()
}

// -----

// CHECK-LABEL: @vector_outerproduct_unknown_mask
// CHECK-NOT: arm_sme.outerproduct
// CHECK:     vector.outerproduct
func.func @vector_outerproduct_unknown_mask(%lhs : vector<[4]xf32>, %rhs : vector<[4]xf32>, %mask : vector<[4]x[4]xi1>) {
  %acc = arm_sme.get_tile : vector<[4]x[4]xf32>
  %0 = vector.mask %mask { vector.outerproduct %lhs, %rhs, %acc {kind = #vector.kind<add>} : vector<[4]xf32>, vector<[4]xf32> } : vector<[4]x[4]xi1> -> vector<[4]x[4]xf32>
  "prevent.dce"(%0) : (vector<[4]x[4]xf32>) -> ()
}
