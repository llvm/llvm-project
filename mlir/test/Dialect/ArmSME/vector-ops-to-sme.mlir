// RUN: mlir-opt %s -convert-vector-to-arm-sme -split-input-file | mlir-opt | FileCheck %s


// CHECK-LABEL:   func.func @transfer_write_2d_zero(
// CHECK-SAME:      %[[ARG_0:.*]]: memref<?x?xi8>) {
func.func @transfer_write_2d_zero(%arg0 : memref<?x?xi8>) {
// CHECK:           %[[C_0:.*]] = arith.constant 0 : index
// CHECK:           %[[ZERO:.*]] = arm_sme.zero : vector<[16]x[16]xi8>
// CHECK:           arm_sme.tile_store %[[ZERO]], %[[ARG_0]][%[[C_0]], %[[C_0]]] : memref<?x?xi8>, vector<[16]x[16]xi8>
// CHECK:           return
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0> : vector<[16]x[16]xi8>
  vector.transfer_write %cst, %arg0[%c0, %c0] {in_bounds = [true, true]} : vector<[16]x[16]xi8>, memref<?x?xi8>
  return
}

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

// CHECK-LABEL: @transfer_write_2d_zero__non_zero_value
// CHECK: vector.transfer_write
// CHECK-NOT: arm_sme.tile_store
func.func @transfer_write_2d_zero__non_zero_value(%arg0 : memref<?x?xi8>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<1> : vector<[16]x[16]xi8>
  vector.transfer_write %cst, %arg0[%c0, %c0] {in_bounds = [true, true]} : vector<[16]x[16]xi8>, memref<?x?xi8>
  return
}

// -----

// CHECK-LABEL: @transfer_write_2d_zero__vec_unknown_defining_op
// CHECK: vector.transfer_write
// CHECK-NOT: arm_sme.tile_store
func.func @transfer_write_2d_zero__vec_unknown_defining_op(%arg0 : memref<?x?xi8>, %arg1 : vector<[16]x[16]xi8>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %arg1, %arg0[%c0, %c0] {in_bounds = [true, true]} : vector<[16]x[16]xi8>, memref<?x?xi8>
  return
}
