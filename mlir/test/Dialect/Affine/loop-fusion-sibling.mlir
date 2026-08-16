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

// CHECK-LABEL: func @sibling_with_used_loop_result
func.func @sibling_with_used_loop_result(%m: memref<4xi32>, %n: memref<4xi32>,
                                         %init: i32) -> i32 {
  // CHECK: %[[RESULT:.*]] = affine.for {{.*}} iter_args
  %a = affine.for %i = 0 to 4 iter_args(%x = %init) -> (i32) {
    %v = affine.load %m[%i] : memref<4xi32>
    %t = arith.addi %x, %v : i32
    affine.yield %t : i32
  }
  affine.for %i = 0 to 4 {
    %v = affine.load %m[%i] : memref<4xi32>
    affine.store %v, %n[%i] : memref<4xi32>
  }
  // CHECK: return %[[RESULT]]
  return %a : i32
}
