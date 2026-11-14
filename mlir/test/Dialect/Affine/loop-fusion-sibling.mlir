// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(affine-loop-fusion{maximal mode=sibling}))' | FileCheck %s

// Test cases specifically for sibling fusion. Note that sibling fusion test
// cases also exist in loop-fusion*.mlir.

// CHECK-LABEL: func @disjoint_stores
func.func @disjoint_stores(%0: memref<8xf32>) {
  %alloc_1 = memref.alloc() : memref<16xf32>
  // The affine stores below are to different parts of the memrefs. Sibling
  // fusion helps improve reuse and is valid.
  affine.for %arg2 = 0 to 8 {
    %2 = affine.load %0[%arg2] : memref<8xf32>
    affine.store %2, %alloc_1[%arg2] : memref<16xf32>
  }
  affine.for %arg2 = 0 to 8 {
    %2 = affine.load %0[%arg2] : memref<8xf32>
    %3 = arith.negf %2 : f32
    affine.store %3, %alloc_1[%arg2 + 8] : memref<16xf32>
  }
  // CHECK: affine.for
  // CHECK-NOT: affine.for
  return
}

// CHECK-LABEL: func.func @sibling_reduction_result
// CHECK: %[[FUSED:.*]] = affine.for {{.*}} iter_args
// CHECK:   affine.store
// CHECK:   affine.yield
// CHECK: %[[SECOND_REDUCTION:.*]] = affine.for {{.*}} iter_args
// CHECK: %[[SUM:.*]] = arith.addi %[[FUSED]], %[[SECOND_REDUCTION]] : i64
// CHECK: arith.trunci %[[SUM]] : i64 to i32
func.func @sibling_reduction_result() -> i32 {
  %c7_i64 = arith.constant 7 : i64
  %c3_i64 = arith.constant 3 : i64
  %c0_i64 = arith.constant 0 : i64
  %c97_i64 = arith.constant 97 : i64
  %c1_i64 = arith.constant 1 : i64

  %alloc = memref.alloc() : memref<8xi64>
  %alloc_0 = memref.alloc() : memref<8xi64>
  %alloc_1 = memref.alloc() : memref<8xi64>

  affine.for %arg0 = 0 to 8 {
    %4 = arith.index_cast %arg0 : index to i64
    %5 = arith.addi %4, %c1_i64 : i64
    %6 = arith.remsi %5, %c97_i64 : i64
    affine.store %6, %alloc[%arg0] : memref<8xi64>
  }

  affine.for %arg0 = 0 to 8 {
    affine.store %c0_i64, %alloc_0[%arg0] : memref<8xi64>
  }

  affine.for %arg0 = 0 to 8 {
    affine.store %c0_i64, %alloc_1[%arg0] : memref<8xi64>
  }

  affine.for %arg0 = 0 to 8 {
    %4 = affine.load %alloc[%arg0] : memref<8xi64>
    %5 = arith.muli %4, %c3_i64 : i64
    affine.store %5, %alloc_0[%arg0] : memref<8xi64>
  }

  affine.for %arg0 = 0 to 8 {
    %4 = affine.load %alloc_0[%arg0] : memref<8xi64>
    %5 = arith.addi %4, %c7_i64 : i64
    affine.store %5, %alloc_1[%arg0] : memref<8xi64>
  }

  %0 = affine.for %arg0 = 0 to 8
      iter_args(%arg1 = %c0_i64) -> (i64) {
    %4 = affine.load %alloc_0[%arg0] : memref<8xi64>
    %5 = arith.addi %arg1, %4 : i64
    affine.yield %5 : i64
  }

  %1 = affine.for %arg0 = 0 to 8
      iter_args(%arg1 = %c0_i64) -> (i64) {
    %4 = affine.load %alloc_1[%arg0] : memref<8xi64>
    %5 = arith.addi %arg1, %4 : i64
    affine.yield %5 : i64
  }

  %2 = arith.addi %0, %1 : i64
  %3 = arith.trunci %2 : i64 to i32

  memref.dealloc %alloc : memref<8xi64>
  memref.dealloc %alloc_0 : memref<8xi64>
  memref.dealloc %alloc_1 : memref<8xi64>

  return %3 : i32
}
