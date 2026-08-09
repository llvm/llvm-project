// RUN: mlir-opt %s -test-split-for-op-at-point -split-input-file -verify-diagnostics | FileCheck %s

// Split [0, 10) at 9 into [0, 9) and [9, 10).
func.func @basic_split(%mem: memref<?xf32>) {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c10 step %c1 {
    memref.store %cst, %mem[%i] : memref<?xf32>
  } {test.split_at = 9 : index}
  return
}
// CHECK-LABEL: func @basic_split
//       CHECK: scf.for %{{.*}} = %c0 to %c9 step %c1
//  CHECK-NEXT: memref.store
//       CHECK: scf.for %{{.*}} = %c9 to %c10 step %c1
//  CHECK-NEXT: memref.store

// -----

// Chain loop-carried values across the split.
func.func @split_with_iter_args() -> i32 {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %r = scf.for %i = %c0 to %c10 step %c1 iter_args(%arg = %c0_i32) -> i32 {
    %one = arith.constant 1 : i32
    %add = arith.addi %arg, %one : i32
    scf.yield %add : i32
  } {test.split_at = 9 : index}
  return %r : i32
}
// CHECK-LABEL: func @split_with_iter_args
//       CHECK: %[[FIRST:.*]] = scf.for %{{.*}} = %c0 to %c9 step %c1 iter_args(%{{.*}} = %c0_i32) -> (i32)
//       CHECK: %[[RESULT:.*]] = scf.for %{{.*}} = %c9 to %c10 step %c1 iter_args(%{{.*}} = %[[FIRST]]) -> (i32)
//       CHECK: return %[[RESULT]] : i32

// -----

// Invalid split point is rejected when bounds are constant.
func.func @invalid_split(%mem: memref<?xf32>) {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  // expected-error @+1 {{failed to split scf.for}}
  scf.for %i = %c0 to %c10 step %c1 {
    memref.store %cst, %mem[%i] : memref<?xf32>
  } {test.split_at = 10 : index}
  return
}

// -----

// Split point must be lowerBound + k * step.
func.func @invalid_split_not_multiple(%mem: memref<?xf32>) {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c3 = arith.constant 3 : index
  // expected-error @+1 {{failed to split scf.for}}
  scf.for %i = %c0 to %c10 step %c3 {
    memref.store %cst, %mem[%i] : memref<?xf32>
  } {test.split_at = 8 : index}
  return
}

// -----

// Dynamic upper bound: split happens and `split < ub` is checked at runtime.
func.func @dynamic_ub_split(%mem: memref<?xf32>, %ub: index) {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %ub step %c1 {
    memref.store %cst, %mem[%i] : memref<?xf32>
  } {test.split_at = 9 : index}
  return
}
// CHECK-LABEL: func @dynamic_ub_split
//  CHECK-SAME: %[[MEM:.*]]: memref<?xf32>, %[[UB:.*]]: index
//       CHECK: %[[C9:.*]] = arith.constant 9 : index
//       CHECK: %[[CMP:.*]] = arith.cmpi slt, %[[C9]], %[[UB]]
//       CHECK: cf.assert %[[CMP]]
//       CHECK: scf.for %{{.*}} = %c0 to %[[C9]] step %c1
//  CHECK-NEXT: memref.store
//       CHECK: scf.for %{{.*}} = %[[C9]] to %[[UB]] step %c1
//  CHECK-NEXT: memref.store

// -----

// Dynamic step: split happens and lattice alignment is checked at runtime.
func.func @dynamic_step_split(%mem: memref<?xf32>, %step: index) {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c12 = arith.constant 12 : index
  scf.for %i = %c0 to %c12 step %step {
    memref.store %cst, %mem[%i] : memref<?xf32>
  } {test.split_at = 6 : index}
  return
}
// CHECK-LABEL: func @dynamic_step_split
//  CHECK-SAME: %[[MEM:.*]]: memref<?xf32>, %[[STEP:.*]]: index
//       CHECK: %[[C6:.*]] = arith.constant 6 : index
//       CHECK: arith.cmpi sgt, %[[STEP]]
//       CHECK: cf.assert
//       CHECK: %[[DIFF:.*]] = arith.subi %[[C6]], %c0
//       CHECK: %[[REM:.*]] = arith.remsi %[[DIFF]], %[[STEP]]
//       CHECK: %[[ALIGNED:.*]] = arith.cmpi eq, %[[REM]]
//       CHECK: cf.assert %[[ALIGNED]]
//       CHECK: scf.for %{{.*}} = %c0 to %[[C6]] step %[[STEP]]
//       CHECK: scf.for %{{.*}} = %[[C6]] to %c12 step %[[STEP]]

// -----

// Split at the lower bound (k = 0): first loop is empty, second keeps the range.
func.func @split_at_lower_bound(%mem: memref<?xf32>) {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c10 step %c1 {
    memref.store %cst, %mem[%i] : memref<?xf32>
  } {test.split_at = 0 : index}
  return
}
// CHECK-LABEL: func @split_at_lower_bound
//       CHECK: %[[LB:.*]] = arith.constant 0 : index
//       CHECK: %[[UB:.*]] = arith.constant 10 : index
//       CHECK: %[[STEP:.*]] = arith.constant 1 : index
//       CHECK: %[[SPLIT:.*]] = arith.constant 0 : index
//       CHECK: scf.for %{{.*}} = %[[LB]] to %[[SPLIT]] step %[[STEP]]
//       CHECK: scf.for %{{.*}} = %[[SPLIT]] to %[[UB]] step %[[STEP]]

// -----

// Constant split point below the lower bound is rejected.
func.func @invalid_split_below_lb(%mem: memref<?xf32>) {
  %cst = arith.constant 0.0 : f32
  %c5 = arith.constant 5 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  // expected-error @+1 {{failed to split scf.for}}
  scf.for %i = %c5 to %c10 step %c1 {
    memref.store %cst, %mem[%i] : memref<?xf32>
  } {test.split_at = 3 : index}
  return
}

// -----

// Constant non-positive step is rejected.
func.func @invalid_zero_step(%mem: memref<?xf32>) {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  // expected-error @+1 {{failed to split scf.for}}
  scf.for %i = %c0 to %c10 step %c0 {
    memref.store %cst, %mem[%i] : memref<?xf32>
  } {test.split_at = 0 : index}
  return
}

// -----

func.func @invalid_negative_step(%mem: memref<?xf32>) {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %cm1 = arith.constant -1 : index
  // expected-error @+1 {{failed to split scf.for}}
  scf.for %i = %c0 to %c10 step %cm1 {
    memref.store %cst, %mem[%i] : memref<?xf32>
  } {test.split_at = 0 : index}
  return
}

// -----

// Integer (non-index) induction variable.
func.func @split_integer_iv() -> i32 {
  %c0 = arith.constant 0 : i32
  %c10 = arith.constant 10 : i32
  %c1 = arith.constant 1 : i32
  %init = arith.constant 0 : i32
  %r = scf.for %i = %c0 to %c10 step %c1 iter_args(%acc = %init) -> i32 : i32 {
    %one = arith.constant 1 : i32
    %add = arith.addi %acc, %one : i32
    scf.yield %add : i32
  } {test.split_at = 9 : i32}
  return %r : i32
}
// CHECK-LABEL: func @split_integer_iv
//       CHECK: %[[FIRST:.*]] = scf.for %{{.*}} = %{{.*}} to %c9_i32 step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (i32)
//       CHECK: %[[RESULT:.*]] = scf.for %{{.*}} = %c9_i32 to %c10_i32 step %{{.*}} iter_args(%{{.*}} = %[[FIRST]]) -> (i32)
//       CHECK: return %[[RESULT]] : i32

// -----

// Unsigned i32 split uses unsigned compares when a bound is dynamic.
func.func @split_unsigned_i32() -> i32 {
  %c0 = arith.constant 0 : i32
  %c10 = arith.constant 10 : i32
  %c1 = arith.constant 1 : i32
  %init = arith.constant 0 : i32
  %r = scf.for unsigned %i = %c0 to %c10 step %c1
      iter_args(%acc = %init) -> i32 : i32 {
    %one = arith.constant 1 : i32
    %add = arith.addi %acc, %one : i32
    scf.yield %add : i32
  } {test.split_at = 9 : i32}
  return %r : i32
}
// CHECK-LABEL: func @split_unsigned_i32
//       CHECK: %[[FIRST:.*]] = scf.for unsigned %{{.*}} = %{{.*}} to %c9_i32 step %{{.*}}
//       CHECK: %[[RESULT:.*]] = scf.for unsigned %{{.*}} = %c9_i32 to %c10_i32 step %{{.*}}
//       CHECK: return %[[RESULT]] : i32

// -----

// Narrow unsigned IV: split at 4 in [0, 5) as i3. Sign-extending 4:i3 is -4
// and would reject this split; zero-extension keeps it in range.
func.func @split_unsigned_i3() -> i32 {
  %c0 = arith.constant 0 : i3
  %c5 = arith.constant 5 : i3
  %c1 = arith.constant 1 : i3
  %init = arith.constant 0 : i32
  %r = scf.for unsigned %i = %c0 to %c5 step %c1
      iter_args(%acc = %init) -> i32 : i3 {
    %one = arith.constant 1 : i32
    %add = arith.addi %acc, %one : i32
    scf.yield %add : i32
  } {test.split_at = 4 : i3}
  return %r : i32
}
// CHECK-LABEL: func @split_unsigned_i3
//       CHECK: %[[FIRST:.*]] = scf.for unsigned %{{.*}} = %{{.*}} to %c-4_i3 step %{{.*}}
//       CHECK: %[[RESULT:.*]] = scf.for unsigned %{{.*}} = %c-4_i3 to %{{.*}} step %{{.*}}
//       CHECK: return %[[RESULT]] : i32

// -----

// Dynamic unsigned upper bound: `split < ub` is an unsigned compare.
func.func @unsigned_dynamic_ub(%ub: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %init = arith.constant 0 : i32
  %r = scf.for unsigned %i = %c0 to %ub step %c1
      iter_args(%acc = %init) -> i32 : i32 {
    %one = arith.constant 1 : i32
    %add = arith.addi %acc, %one : i32
    scf.yield %add : i32
  } {test.split_at = 9 : i32}
  return %r : i32
}
// CHECK-LABEL: func @unsigned_dynamic_ub
//  CHECK-SAME: %[[UB:.*]]: i32
//       CHECK: %[[C9:.*]] = arith.constant 9 : i32
//       CHECK: %[[CMP:.*]] = arith.cmpi ult, %[[C9]], %[[UB]]
//       CHECK: cf.assert %[[CMP]]
//       CHECK: scf.for unsigned %{{.*}} = %{{.*}} to %[[C9]]
//       CHECK: scf.for unsigned %{{.*}} = %[[C9]] to %[[UB]]

// -----

// Dynamic unsigned step: `step > 0` and lattice use unsigned ops.
func.func @unsigned_dynamic_step(%step: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c12 = arith.constant 12 : i32
  %init = arith.constant 0 : i32
  %r = scf.for unsigned %i = %c0 to %c12 step %step
      iter_args(%acc = %init) -> i32 : i32 {
    %one = arith.constant 1 : i32
    %add = arith.addi %acc, %one : i32
    scf.yield %add : i32
  } {test.split_at = 6 : i32}
  return %r : i32
}
// CHECK-LABEL: func @unsigned_dynamic_step
//  CHECK-SAME: %[[STEP:.*]]: i32
//       CHECK: %[[C6:.*]] = arith.constant 6 : i32
//       CHECK: arith.cmpi ugt, %[[STEP]]
//       CHECK: cf.assert
//       CHECK: %[[DIFF:.*]] = arith.subi %[[C6]], %c0_i32
//       CHECK: %[[REM:.*]] = arith.remui %[[DIFF]], %[[STEP]]
//       CHECK: %[[ALIGNED:.*]] = arith.cmpi eq, %[[REM]]
//       CHECK: cf.assert %[[ALIGNED]]
//       CHECK: scf.for unsigned %{{.*}} = %{{.*}} to %[[C6]] step %[[STEP]]
//       CHECK: scf.for unsigned %{{.*}} = %[[C6]] to %{{.*}} step %[[STEP]]

// -----

// Dynamic lower bound: `lb <= split` and lattice alignment are runtime checks.
func.func @dynamic_lb_split(%mem: memref<?xf32>, %lb: index) {
  %cst = arith.constant 0.0 : f32
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %lb to %c10 step %c1 {
    memref.store %cst, %mem[%i] : memref<?xf32>
  } {test.split_at = 9 : index}
  return
}
// CHECK-LABEL: func @dynamic_lb_split
//  CHECK-SAME: %[[MEM:.*]]: memref<?xf32>, %[[LB:.*]]: index
//       CHECK: %[[C9:.*]] = arith.constant 9 : index
//       CHECK: %[[CMP:.*]] = arith.cmpi sle, %[[LB]], %[[C9]]
//       CHECK: cf.assert %[[CMP]]
//       CHECK: %[[DIFF:.*]] = arith.subi %[[C9]], %[[LB]]
//       CHECK: %[[REM:.*]] = arith.remsi %[[DIFF]], %c1
//       CHECK: %[[ALIGNED:.*]] = arith.cmpi eq, %[[REM]]
//       CHECK: cf.assert %[[ALIGNED]]
//       CHECK: scf.for %{{.*}} = %[[LB]] to %[[C9]] step %c1
//       CHECK: scf.for %{{.*}} = %[[C9]] to %c10 step %c1

// -----

// Fully dynamic bounds: every check is a runtime assert.
func.func @dynamic_all_split(%mem: memref<?xf32>, %lb: index, %ub: index,
                             %step: index) {
  %cst = arith.constant 0.0 : f32
  scf.for %i = %lb to %ub step %step {
    memref.store %cst, %mem[%i] : memref<?xf32>
  } {test.split_at = 9 : index}
  return
}
// CHECK-LABEL: func @dynamic_all_split
//  CHECK-SAME: %[[MEM:.*]]: memref<?xf32>, %[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index
//       CHECK: %[[C9:.*]] = arith.constant 9 : index
//       CHECK: arith.cmpi sle, %[[LB]], %[[C9]]
//       CHECK: cf.assert
//       CHECK: arith.cmpi slt, %[[C9]], %[[UB]]
//       CHECK: cf.assert
//       CHECK: arith.cmpi sgt, %[[STEP]]
//       CHECK: cf.assert
//       CHECK: %[[DIFF:.*]] = arith.subi %[[C9]], %[[LB]]
//       CHECK: %[[REM:.*]] = arith.remsi %[[DIFF]], %[[STEP]]
//       CHECK: %[[ALIGNED:.*]] = arith.cmpi eq, %[[REM]]
//       CHECK: cf.assert %[[ALIGNED]]
//       CHECK: scf.for %{{.*}} = %[[LB]] to %[[C9]] step %[[STEP]]
//       CHECK: scf.for %{{.*}} = %[[C9]] to %[[UB]] step %[[STEP]]

// -----

// Dynamic split point (function argument): range and lattice are runtime.
func.func @dynamic_split_point(%mem: memref<?xf32>, %split: index) {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c10 step %c1 {
    memref.store %cst, %mem[%i] : memref<?xf32>
  } {test.split_arg = 1 : i64}
  return
}
// CHECK-LABEL: func @dynamic_split_point
//  CHECK-SAME: %[[MEM:.*]]: memref<?xf32>, %[[SPLIT:.*]]: index
//       CHECK: arith.cmpi sle, %c0, %[[SPLIT]]
//       CHECK: cf.assert
//       CHECK: arith.cmpi slt, %[[SPLIT]], %c10
//       CHECK: cf.assert
//       CHECK: %[[DIFF:.*]] = arith.subi %[[SPLIT]], %c0
//       CHECK: %[[REM:.*]] = arith.remsi %[[DIFF]], %c1
//       CHECK: %[[ALIGNED:.*]] = arith.cmpi eq, %[[REM]]
//       CHECK: cf.assert %[[ALIGNED]]
//       CHECK: scf.for %{{.*}} = %c0 to %[[SPLIT]] step %c1
//       CHECK: scf.for %{{.*}} = %[[SPLIT]] to %c10 step %c1

// -----

// Dynamic ub with a constant zero step: the static step check fails.
func.func @dynamic_ub_zero_step(%mem: memref<?xf32>, %ub: index) {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{failed to split scf.for}}
  scf.for %i = %c0 to %ub step %c0 {
    memref.store %cst, %mem[%i] : memref<?xf32>
  } {test.split_at = 0 : index}
  return
}
