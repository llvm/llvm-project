// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(affine-loop-unroll-jam{unroll-jam-factor=2}))" | FileCheck %s --check-prefixes=CHECK,FACTOR2
// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(affine-loop-unroll-jam{unroll-jam-factor=4}))" | FileCheck %s --check-prefixes=CHECK,FACTOR4

// FACTOR2-LABEL: func.func @addi_reduction
// FACTOR2-SAME: %[[INIT:.*]]: i32
// FACTOR2: %[[ZERO:.*]] = arith.constant 0 : i32
// FACTOR2: %[[RESULT:.*]]:2 = affine.for {{.*}} step 2 iter_args(%[[ACC0:.*]] = %[[INIT]], %[[ACC1:.*]] = %[[ZERO]]) -> (i32, i32) {
// FACTOR2:   %[[ADD0:.*]] = arith.addi %[[ACC0]],
// FACTOR2:   %[[ADD1:.*]] = arith.addi %[[ACC1]],
// FACTOR2:   affine.yield %[[ADD0]], %[[ADD1]]
// FACTOR2: }
// FACTOR2: %[[COMBINED:.*]] = arith.addi %[[RESULT]]#0, %[[RESULT]]#1
// FACTOR2: return %[[COMBINED]]

// FACTOR4-LABEL: func.func @addi_reduction
// FACTOR4-SAME: %[[INIT:.*]]: i32
// FACTOR4: %[[ZERO:.*]] = arith.constant 0 : i32
// FACTOR4: %[[RESULT:.*]]:4 = affine.for {{.*}} step 4 iter_args(%{{.*}} = %[[INIT]], %{{.*}} = %[[ZERO]], %{{.*}} = %[[ZERO]], %{{.*}} = %[[ZERO]]) -> (i32, i32, i32, i32) {
// FACTOR4: }
// FACTOR4: %[[COMBINED0:.*]] = arith.addi %[[RESULT]]#0, %[[RESULT]]#3
// FACTOR4: %[[COMBINED1:.*]] = arith.addi %[[COMBINED0]], %[[RESULT]]#2
// FACTOR4: %[[COMBINED2:.*]] = arith.addi %[[COMBINED1]], %[[RESULT]]#1
// FACTOR4: return %[[COMBINED2]]

func.func @addi_reduction(%init: i32) -> i32 {
  %result = affine.for %i = 0 to 8 iter_args(%acc = %init) -> (i32) {
    %one = arith.constant 1 : i32
    %next = arith.addi %acc, %one : i32
    affine.yield %next : i32
  }
  return %result : i32
}

// FACTOR2-LABEL: func.func @muli_reduction
// FACTOR2-SAME: %[[INIT:.*]]: i32
// FACTOR2: %[[ONE:.*]] = arith.constant 1 : i32
// FACTOR2: affine.for {{.*}} step 2 iter_args(%{{.*}} = %[[INIT]], %{{.*}} = %[[ONE]])

func.func @muli_reduction(%init: i32) -> i32 {
  %result = affine.for %i = 0 to 8 iter_args(%acc = %init) -> (i32) {
    %two = arith.constant 2 : i32
    %next = arith.muli %acc, %two : i32
    affine.yield %next : i32
  }
  return %result : i32
}

// FACTOR2-LABEL: func.func @multiple_reductions
// FACTOR2-SAME: %[[ADD_INIT:.*]]: i32, %[[MUL_INIT:.*]]: i32
// FACTOR2: %[[ZERO:.*]] = arith.constant 0 : i32
// FACTOR2: %[[ONE:.*]] = arith.constant 1 : i32
// FACTOR2: affine.for {{.*}} step 2 iter_args(%{{.*}} = %[[ADD_INIT]], %{{.*}} = %[[MUL_INIT]], %{{.*}} = %[[ZERO]], %{{.*}} = %[[ONE]])

func.func @multiple_reductions(%add_init: i32, %mul_init: i32) -> (i32, i32) {
  %result:2 = affine.for %i = 0 to 8
      iter_args(%add_acc = %add_init, %mul_acc = %mul_init) -> (i32, i32) {
    %one = arith.constant 1 : i32
    %two = arith.constant 2 : i32
    %next_add = arith.addi %add_acc, %one : i32
    %next_mul = arith.muli %mul_acc, %two : i32
    affine.yield %next_add, %next_mul : i32, i32
  }
  return %result#0, %result#1 : i32, i32
}

// The target loop uses an identity for its extra reduction lane. The nested
// loop still uses the mapped initializer for each independent outer lane.
// FACTOR2-LABEL: func.func @nested_iter_args
// FACTOR2-SAME: %[[INIT:.*]]: i32, %[[INNER_INIT:.*]]: i32
// FACTOR2: %[[ZERO:.*]] = arith.constant 0 : i32
// FACTOR2: affine.for {{.*}} step 2 iter_args(%[[OUTER0:.*]] = %[[INIT]], %[[OUTER1:.*]] = %[[ZERO]]) -> (i32, i32) {
// FACTOR2:   affine.for {{.*}} iter_args(%{{.*}} = %[[INNER_INIT]], %{{.*}} = %[[INNER_INIT]])

func.func @nested_iter_args(%init: i32, %inner_init: i32) -> i32 {
  %result = affine.for %i = 0 to 8 iter_args(%acc = %init) -> (i32) {
    %inner = affine.for %j = 0 to 4
        iter_args(%inner_acc = %inner_init) -> (i32) {
      %one = arith.constant 1 : i32
      %next_inner = arith.addi %inner_acc, %one : i32
      affine.yield %next_inner : i32
    }
    %next = arith.addi %acc, %inner : i32
    affine.yield %next : i32
  }
  return %result : i32
}

// CHECK-LABEL: func.func @unsupported_subi
// CHECK: affine.for {{.*}} = 0 to 8 iter_args({{.*}}) -> (i32) {
// CHECK-NOT: step 2
// CHECK-NOT: step 4
// CHECK:   arith.subi
// CHECK: }

func.func @unsupported_subi(%init: i32) -> i32 {
  %result = affine.for %i = 0 to 8 iter_args(%acc = %init) -> (i32) {
    %one = arith.constant 1 : i32
    %next = arith.subi %acc, %one : i32
    affine.yield %next : i32
  }
  return %result : i32
}

// CHECK-LABEL: func.func @mixed_reductions
// CHECK: affine.for {{.*}} = 0 to 8 iter_args({{.*}}) -> (i32, i32) {
// CHECK-NOT: step 2
// CHECK-NOT: step 4
// CHECK:   arith.addi
// CHECK:   arith.subi
// CHECK: }

func.func @mixed_reductions(%add_init: i32, %sub_init: i32) -> (i32, i32) {
  %result:2 = affine.for %i = 0 to 8
      iter_args(%add_acc = %add_init, %sub_acc = %sub_init) -> (i32, i32) {
    %one = arith.constant 1 : i32
    %next_add = arith.addi %add_acc, %one : i32
    %next_sub = arith.subi %sub_acc, %one : i32
    affine.yield %next_add, %next_sub : i32, i32
  }
  return %result#0, %result#1 : i32, i32
}

// CHECK-LABEL: func.func @float_reduction
// CHECK: affine.for {{.*}} = 0 to 8 iter_args({{.*}}) -> (f32) {
// CHECK-NOT: step 2
// CHECK-NOT: step 4
// CHECK:   arith.addf
// CHECK: }

func.func @float_reduction(%init: f32) -> f32 {
  %result = affine.for %i = 0 to 8 iter_args(%acc = %init) -> (f32) {
    %one = arith.constant 1.0 : f32
    %next = arith.addf %acc, %one : f32
    affine.yield %next : f32
  }
  return %result : f32
}

// CHECK-LABEL: func.func @index_reduction
// CHECK: affine.for {{.*}} = 0 to 8 iter_args({{.*}}) -> (index) {
// CHECK-NOT: step 2
// CHECK-NOT: step 4
// CHECK:   arith.addi
// CHECK: }

func.func @index_reduction(%init: index) -> index {
  %result = affine.for %i = 0 to 8 iter_args(%acc = %init) -> (index) {
    %one = arith.constant 1 : index
    %next = arith.addi %acc, %one : index
    affine.yield %next : index
  }
  return %result : index
}

// CHECK-LABEL: func.func @vector_reduction
// CHECK: affine.for {{.*}} = 0 to 8 iter_args({{.*}}) -> (vector<2xi32>) {
// CHECK-NOT: step 2
// CHECK-NOT: step 4
// CHECK:   arith.addi
// CHECK: }

func.func @vector_reduction(%init: vector<2xi32>, %value: vector<2xi32>)
    -> vector<2xi32> {
  %result = affine.for %i = 0 to 8
      iter_args(%acc = %init) -> (vector<2xi32>) {
    %next = arith.addi %acc, %value : vector<2xi32>
    affine.yield %next : vector<2xi32>
  }
  return %result : vector<2xi32>
}

// CHECK-LABEL: func.func @flagged_reduction
// CHECK: affine.for {{.*}} = 0 to 8 iter_args({{.*}}) -> (i32) {
// CHECK-NOT: step 2
// CHECK-NOT: step 4
// CHECK:   arith.addi {{.*}} overflow<nsw>
// CHECK: }

func.func @flagged_reduction(%init: i32) -> i32 {
  %result = affine.for %i = 0 to 8 iter_args(%acc = %init) -> (i32) {
    %one = arith.constant 1 : i32
    %next = arith.addi %acc, %one overflow<nsw> : i32
    affine.yield %next : i32
  }
  return %result : i32
}

// CHECK-LABEL: func.func @inexpressible_cleanup
// CHECK-NOT: arith.constant 0 : i32
// CHECK: affine.for {{.*}} = max
// CHECK-NOT: step 2
// CHECK-NOT: step 4
// CHECK: return

func.func @inexpressible_cleanup(%init: i32, %lb0: index, %lb1: index,
                                 %ub: index) -> i32 {
  %result = affine.for %i = max affine_map<()[s0, s1] -> (s0, s1)>()
      [%lb0, %lb1] to %ub iter_args(%acc = %init) -> (i32) {
    %one = arith.constant 1 : i32
    %next = arith.addi %acc, %one : i32
    affine.yield %next : i32
  }
  return %result : i32
}
