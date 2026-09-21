// RUN: mlir-opt %s --mem2reg --split-input-file | FileCheck %s \
// RUN:   -implicit-check-not "memref.alloca" \
// RUN:   -implicit-check-not "memref.load" \
// RUN:   -implicit-check-not "memref.store"

/// Check promotion through a for loop with a load and store in the body.

// CHECK-LABEL: func.func @for_load_and_store
// CHECK-SAME: (%[[LB:.*]]: index, %[[UB:.*]]: index)
// CHECK-DAG: %[[C5:.*]] = arith.constant 5 : i32
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: %[[RES:.*]] = affine.for %{{.*}} = %[[LB]] to %[[UB]] iter_args(%[[ARG:.*]] = %[[C5]]) -> (i32) {
// CHECK:   %[[NEW:.*]] = arith.addi %[[ARG]], %[[C1]] : i32
// CHECK:   affine.yield %[[NEW]] : i32
// CHECK: }
// CHECK: return %[[RES]] : i32
func.func @for_load_and_store(%lb: index, %ub: index) -> i32 {
  %c5 = arith.constant 5 : i32
  %c1 = arith.constant 1 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c5, %alloca[] : memref<i32>
  affine.for %i = %lb to %ub {
    %load = memref.load %alloca[] : memref<i32>
    %new = arith.addi %load, %c1 : i32
    memref.store %new, %alloca[] : memref<i32>
  }
  %load2 = memref.load %alloca[] : memref<i32>
  return %load2 : i32
}

// -----

/// Check promotion adds a second iter_arg when one already exists.

// CHECK-LABEL: func.func @for_existing_iter_arg
// CHECK-SAME: (%[[LB:.*]]: index, %[[UB:.*]]: index, %[[INIT:.*]]: i32)
// CHECK-DAG: %[[C5:.*]] = arith.constant 5 : i32
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: %[[RES:.*]]:2 = affine.for %{{.*}} = %[[LB]] to %[[UB]] iter_args(%[[MUL_ARG:.*]] = %[[INIT]], %[[SLOT_ARG:.*]] = %[[C5]]) -> (i32, i32) {
// CHECK:   %[[MUL:.*]] = arith.muli %[[MUL_ARG]], %[[MUL_ARG]] : i32
// CHECK:   %[[NEW:.*]] = arith.addi %[[SLOT_ARG]], %[[C1]] : i32
// CHECK:   affine.yield %[[MUL]], %[[NEW]] : i32, i32
// CHECK: }
// CHECK: return %[[RES]]#1 : i32
func.func @for_existing_iter_arg(%lb: index, %ub: index, %init: i32) -> i32 {
  %c5 = arith.constant 5 : i32
  %c1 = arith.constant 1 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c5, %alloca[] : memref<i32>
  %mul_res = affine.for %i = %lb to %ub iter_args(%mul_arg = %init) -> i32 {
    %mul = arith.muli %mul_arg, %mul_arg : i32
    %load = memref.load %alloca[] : memref<i32>
    %new = arith.addi %load, %c1 : i32
    memref.store %new, %alloca[] : memref<i32>
    affine.yield %mul : i32
  }
  %load2 = memref.load %alloca[] : memref<i32>
  return %load2 : i32
}

// -----

/// Check load-only promotion through a for loop generates no iter_arg.

func.func private @use(i32)

// CHECK-LABEL: func.func @for_load_only
// CHECK-SAME: (%[[LB:.*]]: index, %[[UB:.*]]: index)
// CHECK: %[[C5:.*]] = arith.constant 5 : i32
// CHECK: affine.for %{{.*}} = %[[LB]] to %[[UB]] {
// CHECK:   call @use(%[[C5]])
// CHECK: }
// CHECK: return %[[C5]] : i32
func.func @for_load_only(%lb: index, %ub: index) -> i32 {
  %c5 = arith.constant 5 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c5, %alloca[] : memref<i32>
  affine.for %i = %lb to %ub {
    %load = memref.load %alloca[] : memref<i32>
    func.call @use(%load) : (i32) -> ()
  }
  %load2 = memref.load %alloca[] : memref<i32>
  return %load2 : i32
}

// -----

/// Check promotion through nested for loops with a load and store in the inner loop.

// CHECK-LABEL: func.func @for_nested_load_and_store
// CHECK-SAME: (%[[LB:.*]]: index, %[[UB:.*]]: index)
// CHECK-DAG: %[[C5:.*]] = arith.constant 5 : i32
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: %[[OUTER:.*]] = affine.for %{{.*}} = %[[LB]] to %[[UB]] iter_args(%[[OUTER_ARG:.*]] = %[[C5]]) -> (i32) {
// CHECK:   %[[INNER:.*]] = affine.for %{{.*}} = 0 to 4 iter_args(%[[INNER_ARG:.*]] = %[[OUTER_ARG]]) -> (i32) {
// CHECK:     %[[NEW:.*]] = arith.addi %[[INNER_ARG]], %[[C1]] : i32
// CHECK:     affine.yield %[[NEW]] : i32
// CHECK:   }
// CHECK:   affine.yield %[[INNER]] : i32
// CHECK: }
// CHECK: return %[[OUTER]] : i32
func.func @for_nested_load_and_store(%lb: index, %ub: index) -> i32 {
  %c5 = arith.constant 5 : i32
  %c1 = arith.constant 1 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c5, %alloca[] : memref<i32>
  affine.for %i = %lb to %ub {
    affine.for %j = 0 to 4 {
      %load = memref.load %alloca[] : memref<i32>
      %new = arith.addi %load, %c1 : i32
      memref.store %new, %alloca[] : memref<i32>
    }
  }
  %load2 = memref.load %alloca[] : memref<i32>
  return %load2 : i32
}
