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

// Test fusion of two independent reductions with live SSA results.
// Both original loop results must be preserved after fusion.

// CHECK-LABEL: func.func @independent_reductions
// CHECK: %[[FUSED:[0-9]+]]:2 = affine.for
// CHECK: affine.yield {{.*}} : f32, f32
// CHECK: arith.subf %[[FUSED]]#1, %[[FUSED]]#0
func.func @independent_reductions(%a: memref<32xf32>) -> f32 {
  %zero = arith.constant 0.0 : f32
  %two = arith.constant 2.0 : f32

  %first = affine.for %i = 0 to 32
      iter_args(%acc = %zero) -> f32 {
    %x = affine.load %a[%i] : memref<32xf32>
    %sum = arith.addf %acc, %x : f32
    affine.yield %sum : f32
  }

  %second = affine.for %i = 0 to 32
      iter_args(%acc = %zero) -> f32 {
    %x = affine.load %a[%i] : memref<32xf32>
    %scaled = arith.mulf %x, %two : f32
    %sum = arith.addf %acc, %scaled : f32
    affine.yield %sum : f32
  }

  %result = arith.subf %first, %second : f32
  return %result : f32
}

// Test that sibling fusion is skipped when the second loop directly
// depends on the first loop's SSA result.

// CHECK-LABEL: func.func @direct_ssa_dependence
// CHECK: %[[FIRST:[0-9]+]] = affine.for
// CHECK: %[[SECOND:[0-9]+]] = affine.for {{.*}}iter_args({{.*}} = %[[FIRST]])
func.func @direct_ssa_dependence(%a: memref<32xf32>) -> f32 {
  %zero = arith.constant 0.0 : f32

  %first = affine.for %i = 0 to 32
      iter_args(%acc = %zero) -> f32 {
    %x = affine.load %a[%i] : memref<32xf32>
    %sum = arith.addf %acc, %x : f32
    affine.yield %sum : f32
  }

  %second = affine.for %i = 0 to 32
      iter_args(%acc = %first) -> f32 {
    %x = affine.load %a[%i] : memref<32xf32>
    %sum = arith.addf %acc, %x : f32
    affine.yield %sum : f32
  }

  return %second : f32
}

// Test that transitive SSA dependencies also prevent sibling fusion.

// CHECK-LABEL: func.func @transitive_ssa_dependence
// CHECK: %[[FIRST:[0-9]+]] = affine.for
// CHECK: %[[SEED:[0-9]+]] = arith.addf %[[FIRST]],
// CHECK: %[[SECOND:[0-9]+]] = affine.for {{.*}}iter_args({{.*}} = %[[SEED]])
func.func @transitive_ssa_dependence(%a: memref<32xf32>) -> f32 {
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32

  %first = affine.for %i = 0 to 32
      iter_args(%acc = %zero) -> f32 {
    %x = affine.load %a[%i] : memref<32xf32>
    %sum = arith.addf %acc, %x : f32
    affine.yield %sum : f32
  }

  %seed = arith.addf %first, %one : f32

  %second = affine.for %i = 0 to 32
      iter_args(%acc = %seed) -> f32 {
    %x = affine.load %a[%i] : memref<32xf32>
    %sum = arith.addf %acc, %x : f32
    affine.yield %sum : f32
  }

  return %second : f32
}

// Test that placing an SSA usage in a nested region prevents the fusion

// CHECK-LABEL: func.func @intervening_nested_result_user
// CHECK: %[[FIRST:[0-9]+]] = affine.for
// CHECK: affine.for
// CHECK: affine.store %[[FIRST]],
// CHECK: %[[SECOND:[0-9]+]] = affine.for

func.func @intervening_nested_result_user(
    %a: memref<8xf32>,
    %out: memref<4xf32>) -> f32 {
  %zero = arith.constant 0.0 : f32

  %first = affine.for %i = 0 to 8
      iter_args(%acc = %zero) -> f32 {
    %x = affine.load %a[%i] : memref<8xf32>
    %next = arith.addf %acc, %x : f32
    affine.yield %next : f32
  }

  affine.for %j = 0 to 4 {
    affine.store %first, %out[%j] : memref<4xf32>
  }

  %second = affine.for %i = 0 to 8
      iter_args(%acc = %zero) -> f32 {
    %x = affine.load %a[%i] : memref<8xf32>
    %next = arith.addf %acc, %x : f32
    affine.yield %next : f32
  }

  return %second : f32
}

// Test sibling fusion of three independent reductions that share
// different input memrefs. All three SSA results must be preserved.

// CHECK-LABEL: func.func @alternating_dot
// CHECK: %[[FUSED:[0-9]+]]:3 = affine.for
// CHECK: affine.yield {{.*}} : f32, f32, f32
// CHECK: arith.addf
// CHECK: arith.addf
// CHECK: return

func.func @alternating_dot(
    %a: memref<32xf32>,
    %b: memref<32xf32>,
    %c: memref<32xf32>) -> f32 {
  %zero = arith.constant 0.0 : f32

  %tmp1 = affine.for %i = 0 to 32
      iter_args(%acc = %zero) -> f32 {
    %x = affine.load %a[%i] : memref<32xf32>
    %y = affine.load %b[%i] : memref<32xf32>
    %product = arith.mulf %x, %y : f32
    %sum = arith.addf %acc, %product : f32
    affine.yield %sum : f32
  }

  %tmp2 = affine.for %i = 0 to 32
      iter_args(%acc = %zero) -> f32 {
    %x = affine.load %b[%i] : memref<32xf32>
    %y = affine.load %c[%i] : memref<32xf32>
    %product = arith.mulf %x, %y : f32
    %sum = arith.addf %acc, %product : f32
    affine.yield %sum : f32
  }

  %tmp3 = affine.for %i = 0 to 32
      iter_args(%acc = %zero) -> f32 {
    %x = affine.load %a[%i] : memref<32xf32>
    %y = affine.load %c[%i] : memref<32xf32>
    %product = arith.mulf %x, %y : f32
    %sum = arith.addf %acc, %product : f32
    affine.yield %sum : f32
  }

  %sum12 = arith.addf %tmp1, %tmp2 : f32
  %result = arith.addf %sum12, %tmp3 : f32
  return %result : f32
}
