// RUN: mlir-opt %s -convert-vector-to-llvm="enable-arm-sme" -split-input-file | mlir-opt | FileCheck %s

// CHECK-LABEL: @transfer_write_2d_zero_i8
// CHECK: %[[C255:.*]] = arith.constant 255 : i32
// CHECK: "arm_sme.intr.zero"(%[[C255]]) : (i32) -> ()
func.func @transfer_write_2d_zero_i8() {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %vscale = vector.vscale
  %dim = arith.muli %c16, %vscale : index
  %0 = memref.alloc(%dim, %dim) : memref<?x?xi8>
  %cst = arith.constant dense<0> : vector<[16x16]xi8>
  vector.transfer_write %cst, %0[%c0, %c0] {in_bounds = [true, true]} : vector<[16x16]xi8>, memref<?x?xi8>
  memref.dealloc %0 : memref<?x?xi8>
  return
}

// -----

// The following tests check the 'vector.transfer_write' -> 'arm_sme.intr.zero'
// lowering only occurs for vector types of correct rank, shape, element size
// and number of scalable dims.

// CHECK-LABEL: @transfer_write_2d_zero__bad_type
// CHECK: vector.transfer_write
// CHECK-NOT: arm_sme.intr.zero
func.func @transfer_write_2d_zero__bad_type() {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %vscale = vector.vscale
  %dim = arith.muli %c16, %vscale : index
  %0 = memref.alloc(%dim, %dim) : memref<?x?xi4>
  %cst = arith.constant dense<0> : vector<[16x16]xi4>
  vector.transfer_write %cst, %0[%c0, %c0] {in_bounds = [true, true]} : vector<[16x16]xi4>, memref<?x?xi4>
  memref.dealloc %0 : memref<?x?xi4>
  return
}

// -----

// CHECK-LABEL: @transfer_write_2d_zero__bad_shape
// CHECK: vector.transfer_write
// CHECK-NOT: arm_sme.intr.zero
func.func @transfer_write_2d_zero__bad_shape() {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %vscale = vector.vscale
  %dim = arith.muli %c8, %vscale : index
  %0 = memref.alloc(%dim, %dim) : memref<?x?xi8>
  %cst = arith.constant dense<0> : vector<[8x8]xi8>
  vector.transfer_write %cst, %0[%c0, %c0] {in_bounds = [true, true]} : vector<[8x8]xi8>, memref<?x?xi8>
  memref.dealloc %0 : memref<?x?xi8>
  return
}

// -----

// CHECK-LABEL: @transfer_write_2d_zero__bad_rank
// CHECK: vector.transfer_write
// CHECK-NOT: arm_sme.intr.zero
func.func @transfer_write_2d_zero__bad_rank() {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %vscale = vector.vscale
  %dim = arith.muli %c16, %vscale : index
  %0 = memref.alloc(%dim, %dim, %dim) : memref<?x?x?xi8>
  %cst = arith.constant dense<0> : vector<[16x16x16]xi8>
  vector.transfer_write %cst, %0[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<[16x16x16]xi8>, memref<?x?x?xi8>
  memref.dealloc %0 : memref<?x?x?xi8>
  return
}

// -----

// CHECK-LABEL: @transfer_write_2d_zero__bad_num_scalable_dims
// CHECK: vector.transfer_write
// CHECK-NOT: arm_sme.intr.zero
func.func @transfer_write_2d_zero__bad_num_scalable_dims() {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %vscale = vector.vscale
  %dim = arith.muli %c16, %vscale : index
  %0 = memref.alloc(%dim) : memref<16x?xi8>
  %cst = arith.constant dense<0> : vector<16x[16]xi8>
  vector.transfer_write %cst, %0[%c0, %c0] {in_bounds = [true, true]} : vector<16x[16]xi8>, memref<16x?xi8>
  memref.dealloc %0 : memref<16x?xi8>
  return
}
