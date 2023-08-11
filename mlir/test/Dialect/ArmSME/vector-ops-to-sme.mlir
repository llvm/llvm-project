// RUN: mlir-opt %s -convert-vector-to-arm-sme -split-input-file -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func.func @transfer_write_2d_i8(
// CHECK-SAME:                                   %[[VECTOR:.*]]: vector<[16]x[16]xi8>,
// CHECK-SAME:                                   %[[DEST:.*]]: memref<?x?xi8>) {
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         arm_sme.tile_store %[[VECTOR]], %[[DEST]]{{\[}}%[[C0]], %[[C0]]] : memref<?x?xi8>, vector<[16]x[16]xi8>
func.func @transfer_write_2d_i8(%vector : vector<[16]x[16]xi8>, %dest : memref<?x?xi8>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %vector, %dest[%c0, %c0] {in_bounds = [true, true]} : vector<[16]x[16]xi8>, memref<?x?xi8>
  return
}

// -----

// CHECK-LABEL: func.func @transfer_write_2d_i16(
// CHECK-SAME:                                   %[[VECTOR:.*]]: vector<[8]x[8]xi16>,
// CHECK-SAME:                                   %[[DEST:.*]]: memref<?x?xi16>) {
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         arm_sme.tile_store %[[VECTOR]], %[[DEST]]{{\[}}%[[C0]], %[[C0]]] : memref<?x?xi16>, vector<[8]x[8]xi16>
func.func @transfer_write_2d_i16(%vector : vector<[8]x[8]xi16>, %dest : memref<?x?xi16>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %vector, %dest[%c0, %c0] {in_bounds = [true, true]} : vector<[8]x[8]xi16>, memref<?x?xi16>
  return
}

// -----

// CHECK-LABEL: func.func @transfer_write_2d_i32(
// CHECK-SAME:                                   %[[VECTOR:.*]]: vector<[4]x[4]xi32>,
// CHECK-SAME:                                   %[[DEST:.*]]: memref<?x?xi32>) {
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         arm_sme.tile_store %[[VECTOR]], %[[DEST]]{{\[}}%[[C0]], %[[C0]]] : memref<?x?xi32>, vector<[4]x[4]xi32>
func.func @transfer_write_2d_i32(%vector : vector<[4]x[4]xi32>, %dest : memref<?x?xi32>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %vector, %dest[%c0, %c0] {in_bounds = [true, true]} : vector<[4]x[4]xi32>, memref<?x?xi32>
  return
}

// -----

// CHECK-LABEL: func.func @transfer_write_2d_i64(
// CHECK-SAME:                                   %[[VECTOR:.*]]: vector<[2]x[2]xi64>,
// CHECK-SAME:                                   %[[DEST:.*]]: memref<?x?xi64>) {
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         arm_sme.tile_store %[[VECTOR]], %[[DEST]]{{\[}}%[[C0]], %[[C0]]] : memref<?x?xi64>, vector<[2]x[2]xi64>
func.func @transfer_write_2d_i64(%vector : vector<[2]x[2]xi64>, %dest : memref<?x?xi64>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %vector, %dest[%c0, %c0] {in_bounds = [true, true]} : vector<[2]x[2]xi64>, memref<?x?xi64>
  return
}

// -----

// CHECK-LABEL: func.func @transfer_write_2d_f16(
// CHECK-SAME:                                   %[[VECTOR:.*]]: vector<[8]x[8]xf16>,
// CHECK-SAME:                                   %[[DEST:.*]]: memref<?x?xf16>) {
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         arm_sme.tile_store %[[VECTOR]], %[[DEST]]{{\[}}%[[C0]], %[[C0]]] : memref<?x?xf16>, vector<[8]x[8]xf16>
func.func @transfer_write_2d_f16(%vector : vector<[8]x[8]xf16>, %dest : memref<?x?xf16>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %vector, %dest[%c0, %c0] {in_bounds = [true, true]} : vector<[8]x[8]xf16>, memref<?x?xf16>
  return
}

// -----

// CHECK-LABEL: func.func @transfer_write_2d_bf16(
// CHECK-SAME:                                   %[[VECTOR:.*]]: vector<[8]x[8]xbf16>,
// CHECK-SAME:                                   %[[DEST:.*]]: memref<?x?xbf16>) {
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         arm_sme.tile_store %[[VECTOR]], %[[DEST]]{{\[}}%[[C0]], %[[C0]]] : memref<?x?xbf16>, vector<[8]x[8]xbf16>
func.func @transfer_write_2d_bf16(%vector : vector<[8]x[8]xbf16>, %dest : memref<?x?xbf16>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %vector, %dest[%c0, %c0] {in_bounds = [true, true]} : vector<[8]x[8]xbf16>, memref<?x?xbf16>
  return
}

// -----

// CHECK-LABEL: func.func @transfer_write_2d_f32(
// CHECK-SAME:                                   %[[VECTOR:.*]]: vector<[4]x[4]xf32>,
// CHECK-SAME:                                   %[[DEST:.*]]: memref<?x?xf32>) {
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         arm_sme.tile_store %[[VECTOR]], %[[DEST]]{{\[}}%[[C0]], %[[C0]]] : memref<?x?xf32>, vector<[4]x[4]xf32>
func.func @transfer_write_2d_f32(%vector : vector<[4]x[4]xf32>, %dest : memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %vector, %dest[%c0, %c0] {in_bounds = [true, true]} : vector<[4]x[4]xf32>, memref<?x?xf32>
  return
}

// -----

// CHECK-LABEL: func.func @transfer_write_2d_f64(
// CHECK-SAME:                                   %[[VECTOR:.*]]: vector<[2]x[2]xf64>,
// CHECK-SAME:                                   %[[DEST:.*]]: memref<?x?xf64>) {
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         arm_sme.tile_store %[[VECTOR]], %[[DEST]]{{\[}}%[[C0]], %[[C0]]] : memref<?x?xf64>, vector<[2]x[2]xf64>
func.func @transfer_write_2d_f64(%vector : vector<[2]x[2]xf64>, %dest : memref<?x?xf64>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %vector, %dest[%c0, %c0] {in_bounds = [true, true]} : vector<[2]x[2]xf64>, memref<?x?xf64>
  return
}

// -----

// CHECK-LABEL: @arith_constant_dense_2d_zero_i8
// CHECK: %[[ZERO:.*]] = arm_sme.zero : vector<[16]x[16]xi8>
func.func @arith_constant_dense_2d_zero_i8() {
  %zero = arith.constant dense<0> : vector<[16]x[16]xi8>
  "prevent.dce"(%zero) : (vector<[16]x[16]xi8>) -> ()
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
