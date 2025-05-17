// RUN: mlir-opt %s -test-slp-vectorization | FileCheck %s

// CHECK-LABEL: func @basic_slp
func.func @basic_slp(%arg0: memref<8xi32>, %arg1: memref<8xi32>) {
  // CHECK: vector.transfer_read
  // CHECK: arith.addi
  // CHECK: vector.transfer_write
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index

  %0 = memref.load %arg0[%c0] : memref<8xi32>
  %1 = memref.load %arg0[%c1] : memref<8xi32>
  %2 = memref.load %arg0[%c2] : memref<8xi32>
  %3 = memref.load %arg0[%c3] : memref<8xi32>

  %4 = memref.load %arg1[%c0] : memref<8xi32>
  %5 = memref.load %arg1[%c1] : memref<8xi32>
  %6 = memref.load %arg1[%c2] : memref<8xi32>
  %7 = memref.load %arg1[%c3] : memref<8xi32>

  %8 = arith.addi %0, %4 : i32
  %9 = arith.addi %1, %5 : i32
  %10 = arith.addi %2, %6 : i32
  %11 = arith.addi %3, %7 : i32

  memref.store %8, %arg0[%c0] : memref<8xi32>
  memref.store %9, %arg0[%c1] : memref<8xi32>
  memref.store %10, %arg0[%c2] : memref<8xi32>
  memref.store %11, %arg0[%c3] : memref<8xi32>

  return
}
