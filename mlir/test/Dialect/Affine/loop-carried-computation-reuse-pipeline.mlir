// RUN: mlir-opt %s --pass-pipeline='builtin.module(func.func(affine-loop-fusion{mode=producer maximal},affine-loop-unroll{unroll-factor=-1 unroll-full-threshold=2},affine-scalrep,affine-loop-carried-computation-reuse),canonicalize,cse)' | FileCheck %s

// Existing fusion, short-loop unrolling, and scalar replacement expose the
// translated producer pair. Computation reuse is responsible only for the
// final loop-carried SSA value.

// CHECK-LABEL: func.func @fusible
// CHECK-NOT: memref.alloc
// CHECK: %[[A:.*]] = affine.load %[[SRC:.*]][0]
// CHECK: %[[B:.*]] = affine.load %[[SRC]][1]
// CHECK: %[[INIT:.*]] = arith.muli %[[A]], %[[B]]
// CHECK: affine.for %[[I:.*]] = 0 to 16 iter_args(%[[PREV:.*]] = %[[INIT]])
// CHECK: %[[B2:.*]] = affine.load %[[SRC]][%[[I]] + 1]
// CHECK: %[[C:.*]] = affine.load %[[SRC]][%[[I]] + 2]
// CHECK: %[[CURRENT:.*]] = arith.muli %[[B2]], %[[C]]
// CHECK: arith.subi %[[CURRENT]], %[[PREV]]
// CHECK: affine.yield %[[CURRENT]]
// CHECK: return
func.func @fusible(%src0: memref<18xi32>, %out0: memref<16xi32>) {
  %src, %out = memref.distinct_objects %src0, %out0
      : memref<18xi32>, memref<16xi32>
  %temporary = memref.alloc() : memref<17xi32>
  affine.for %f = 0 to 17 {
    %a = affine.load %src[%f] : memref<18xi32>
    %b = affine.load %src[%f + 1] : memref<18xi32>
    %product = arith.muli %a, %b : i32
    affine.store %product, %temporary[%f] : memref<17xi32>
  }
  affine.for %i = 0 to 16 {
    %left = affine.load %temporary[%i] : memref<17xi32>
    %right = affine.load %temporary[%i + 1] : memref<17xi32>
    %difference = arith.subi %right, %left : i32
    affine.store %difference, %out[%i] : memref<16xi32>
  }
  memref.dealloc %temporary : memref<17xi32>
  return
}

// Current affine-loop-fusion intentionally skips a destination loop with
// results. Keep this boundary visible instead of silently growing the custom
// reuse pass into a fusion implementation.

// CHECK-LABEL: func.func @result_bearing_consumer
// CHECK: %[[TEMP:.*]] = memref.alloc
// CHECK: affine.for %{{.*}} = 0 to 17
// CHECK: affine.store %{{.*}}, %[[TEMP]]
// CHECK: %[[SUM:.*]] = affine.for %{{.*}} = 0 to 16 iter_args
// CHECK: affine.load %[[TEMP]]
// CHECK: affine.load %[[TEMP]]
// CHECK: return %[[SUM]]
func.func @result_bearing_consumer(%src0: memref<18xi32>,
                                   %out0: memref<16xi32>) -> i32 {
  %src, %out = memref.distinct_objects %src0, %out0
      : memref<18xi32>, memref<16xi32>
  %temporary = memref.alloc() : memref<17xi32>
  %zero = arith.constant 0 : i32
  affine.for %f = 0 to 17 {
    %a = affine.load %src[%f] : memref<18xi32>
    %b = affine.load %src[%f + 1] : memref<18xi32>
    %product = arith.muli %a, %b : i32
    affine.store %product, %temporary[%f] : memref<17xi32>
  }
  %sum = affine.for %i = 0 to 16 iter_args(%acc = %zero) -> i32 {
    %left = affine.load %temporary[%i] : memref<17xi32>
    %right = affine.load %temporary[%i + 1] : memref<17xi32>
    %difference = arith.subi %right, %left : i32
    affine.store %difference, %out[%i] : memref<16xi32>
    %next = arith.addi %acc, %difference : i32
    affine.yield %next : i32
  }
  memref.dealloc %temporary : memref<17xi32>
  return %sum : i32
}
