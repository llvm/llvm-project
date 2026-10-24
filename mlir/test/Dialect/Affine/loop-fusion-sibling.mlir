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
  // CHECK: %[[RESULT:.*]] = affine.for %[[I:.*]] = 0 to 4 iter_args(%[[CARRIED:.*]] = %{{.*}}) -> (i32) {
  // CHECK:   %[[LOAD:.*]] = affine.load %{{.*}}[%[[I]]] : memref<4xi32>
  // CHECK:   %[[ADD:.*]] = arith.addi %[[CARRIED]], %[[LOAD]] : i32
  // CHECK:   affine.store
  // CHECK:   affine.yield %[[ADD]] : i32
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

// CHECK-LABEL: func @sibling_with_multiple_used_loop_results
func.func @sibling_with_multiple_used_loop_results(%m: memref<4xi32>,
                                                   %n: memref<4xi32>,
                                                   %init0: i32,
                                                   %init1: i32) -> (i32, i32) {
  // CHECK: %[[RESULT:.*]]:2 = affine.for %[[I:.*]] = 0 to 4 iter_args(%[[CARRIED0:.*]] = %{{.*}}, %[[CARRIED1:.*]] = %{{.*}}) -> (i32, i32) {
  // CHECK:   %[[LOAD:.*]] = affine.load %{{.*}}[%[[I]]] : memref<4xi32>
  // CHECK:   %[[ADD0:.*]] = arith.addi %[[CARRIED0]], %[[LOAD]] : i32
  // CHECK:   %[[ADD1:.*]] = arith.addi %[[CARRIED1]], %[[LOAD]] : i32
  // CHECK:   affine.store
  // CHECK:   affine.yield %[[ADD0]], %[[ADD1]] : i32, i32
  %a:2 = affine.for %i = 0 to 4 iter_args(%x = %init0, %y = %init1) -> (i32, i32) {
    %v = affine.load %m[%i] : memref<4xi32>
    %t0 = arith.addi %x, %v : i32
    %t1 = arith.addi %y, %v : i32
    affine.yield %t0, %t1 : i32, i32
  }
  affine.for %i = 0 to 4 {
    %v = affine.load %m[%i] : memref<4xi32>
    affine.store %v, %n[%i] : memref<4xi32>
  }
  // CHECK: return %[[RESULT]]#0, %[[RESULT]]#1
  return %a#0, %a#1 : i32, i32
}

// CHECK-LABEL: func @sibling_with_second_of_multiple_loop_results_used
func.func @sibling_with_second_of_multiple_loop_results_used(%m: memref<4xi32>,
                                                             %n: memref<4xi32>,
                                                             %init0: i32,
                                                             %init1: i32) -> i32 {
  // CHECK: %[[RESULT:.*]] = affine.for %[[I:.*]] = 0 to 4 iter_args(%[[CARRIED:.*]] = %{{.*}}) -> (i32) {
  // CHECK:   %[[LOAD:.*]] = affine.load %{{.*}}[%[[I]]] : memref<4xi32>
  // CHECK:   arith.addi %{{.*}}, %[[LOAD]] : i32
  // CHECK:   %[[ADD:.*]] = arith.addi %[[CARRIED]], %[[LOAD]] : i32
  // CHECK:   affine.store
  // CHECK:   affine.yield %[[ADD]] : i32
  %a:2 = affine.for %i = 0 to 4 iter_args(%x = %init0, %y = %init1) -> (i32, i32) {
    %v = affine.load %m[%i] : memref<4xi32>
    %t0 = arith.addi %x, %v : i32
    %t1 = arith.addi %y, %v : i32
    affine.yield %t0, %t1 : i32, i32
  }
  affine.for %i = 0 to 4 {
    %v = affine.load %m[%i] : memref<4xi32>
    affine.store %v, %n[%i] : memref<4xi32>
  }
  // CHECK: return %[[RESULT]]
  return %a#1 : i32
}

// CHECK-LABEL: func @sibling_result_used_before_destination
func.func @sibling_result_used_before_destination(%m: memref<4xi32>,
                                                  %n: memref<4xi32>,
                                                  %init: i32) -> i32 {
  // CHECK: %[[A:.*]] = affine.for
  %a = affine.for %i = 0 to 4 iter_args(%x = %init) -> (i32) {
    %v = affine.load %m[%i] : memref<4xi32>
    %t = arith.addi %x, %v : i32
    affine.yield %t : i32
  }
  // CHECK: %[[B:.*]] = arith.addi %[[A]],
  %b = arith.addi %a, %init : i32
  // CHECK: affine.for
  affine.for %i = 0 to 4 {
    %v = affine.load %m[%i] : memref<4xi32>
    affine.store %v, %n[%i] : memref<4xi32>
  }
  // CHECK: return %[[B]]
  return %b : i32
}

// CHECK-LABEL: func @destination_result_used_before_sibling
func.func @destination_result_used_before_sibling(%m: memref<4xi32>,
                                                  %n: memref<4xi32>,
                                                  %init: i32) -> i32 {
  // CHECK: %[[A:.*]] = affine.for
  %a = affine.for %i = 0 to 4 iter_args(%x = %init) -> (i32) {
    %v = affine.load %m[%i] : memref<4xi32>
    affine.store %v, %n[%i] : memref<4xi32>
    affine.yield %v : i32
  }
  // CHECK: %[[B:.*]] = arith.addi %[[A]],
  %b = arith.addi %a, %init : i32
  // CHECK: affine.for
  affine.for %i = 0 to 4 {
    %v = affine.load %m[%i] : memref<4xi32>
    affine.store %v, %n[%i] : memref<4xi32>
  }
  // CHECK: return %[[B]]
  return %b : i32
}
