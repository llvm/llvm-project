// RUN: mlir-opt %s -allow-unregistered-dialect -scf-for-loop-peeling=peel-back=true -split-input-file | FileCheck %s

// Remainder last peel: [0, 17) step 4 -> main [0, 16) and last [16, 17).
// affine.min is not simplified (unlike peelForLoopAndSimplifyBounds).
#map = affine_map<(d0, d1)[s0] -> (s0, d0 - d1)>
func.func @peel_last_remainder() -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %lb = arith.constant 0 : index
  %step = arith.constant 4 : index
  %ub = arith.constant 17 : index
  %r = scf.for %iv = %lb to %ub step %step iter_args(%arg = %c0_i32) -> i32 {
    %s = affine.min #map(%ub, %iv)[%step]
    %casted = arith.index_cast %s : index to i32
    %0 = arith.addi %arg, %casted : i32
    scf.yield %0 : i32
  }
  return %r : i32
}
// CHECK-LABEL: func @peel_last_remainder
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C4:.*]] = arith.constant 4 : index
// CHECK: %[[C17:.*]] = arith.constant 17 : index
// CHECK: %[[MAIN:.*]] = scf.for %{{.*}} = %[[C0]] to %[[C16:.*]] step %[[C4]]
// CHECK:   affine.min
// CHECK: %[[RESULT:.*]] = scf.for %{{.*}} = %[[C16]] to %[[C17]] step %[[C4]] iter_args(%{{.*}} = %[[MAIN]])
// CHECK:   affine.min
// CHECK: return %[[RESULT]]

// -----

// Evenly divisible range: last peel peels a full last iteration.
func.func @peel_last_evenly_divisible() {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c4 = arith.constant 4 : index
  scf.for %iv = %c0 to %c16 step %c4 {
    "test.use"(%iv) : (index) -> ()
  }
  return
}
// CHECK-LABEL: func @peel_last_evenly_divisible
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C16:.*]] = arith.constant 16 : index
// CHECK: %[[C4:.*]] = arith.constant 4 : index
// CHECK: %[[C12:.*]] = arith.constant 12 : index
// CHECK: scf.for %{{.*}} = %[[C0]] to %[[C12]] step %[[C4]]
// CHECK: scf.for %{{.*}} = %[[C12]] to %[[C16]] step %[[C4]]

// -----

// Single iteration: trip count is 1, so the loop is left unchanged.
func.func @no_peel_one_iteration() {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  scf.for %iv = %c0 to %c4 step %c4 {
    "test.use"(%iv) : (index) -> ()
  }
  return
}
// CHECK-LABEL: func @no_peel_one_iteration
// CHECK: scf.for %{{.*}} = %c0 to %c4 step %c4
// CHECK-NOT: scf.for

// -----

// Step size of 1 with more than 1 iteration: peels the last iteration.
func.func @peel_last_step_one() {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c10 step %c1 {
    "test.use"(%iv) : (index) -> ()
  }
  return
}
// CHECK-LABEL: func @peel_last_step_one
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C10:.*]] = arith.constant 10 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[C9:.*]] = arith.constant 9 : index
// CHECK: scf.for %{{.*}} = %[[C0]] to %[[C9]] step %[[C1]]
// CHECK: scf.for %{{.*}} = %[[C9]] to %[[C10]] step %[[C1]]

// -----

// Dynamic bounds: split bound is computed at runtime via min/sub/rem/select.
func.func @peel_last_dynamic(%lb: index, %ub: index) {
  %step = arith.constant 4 : index
  scf.for %iv = %lb to %ub step %step {
    "test.use"(%iv) : (index) -> ()
  }
  return
}
// CHECK-LABEL: func @peel_last_dynamic
// CHECK-SAME:  %[[LB:.*]]: index, %[[UB:.*]]: index
// CHECK-DAG:   %[[STEP:.*]] = arith.constant 4 : index
// CHECK-DAG:   %[[ZERO:.*]] = arith.constant 0 : index
// CHECK:       %[[MIN:.*]] = arith.minsi %[[LB]], %[[UB]]
// CHECK:       %[[DIFF:.*]] = arith.subi %[[UB]], %[[MIN]]
// CHECK:       %[[REM:.*]] = arith.remui %[[DIFF]], %[[STEP]]
// CHECK:       %[[IS_ZERO:.*]] = arith.cmpi eq, %[[REM]], %[[ZERO]]
// CHECK:       %[[NOT_EMPTY:.*]] = arith.cmpi ne, %[[DIFF]], %[[ZERO]]
// CHECK:       %[[PEEL_FULL:.*]] = arith.andi %[[IS_ZERO]], %[[NOT_EMPTY]]
// CHECK:       %[[SUB:.*]] = arith.select %[[PEEL_FULL]], %[[STEP]], %[[REM]]
// CHECK:       %[[SPLIT:.*]] = arith.subi %[[UB]], %[[SUB]]
// CHECK:       scf.for %{{.*}} = %[[LB]] to %[[SPLIT]] step %[[STEP]]
// CHECK:       scf.for %{{.*}} = %[[SPLIT]] to %[[UB]] step %[[STEP]]

// -----

// Non-zero lower bound: [2, 17) step 4 -> main [2, 14) and last [14, 17).
func.func @peel_last_nonzero_lb() {
  %c2 = arith.constant 2 : index
  %c17 = arith.constant 17 : index
  %c4 = arith.constant 4 : index
  scf.for %iv = %c2 to %c17 step %c4 {
    "test.use"(%iv) : (index) -> ()
  }
  return
}
// CHECK-LABEL: func @peel_last_nonzero_lb
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[C17:.*]] = arith.constant 17 : index
// CHECK: %[[C4:.*]] = arith.constant 4 : index
// CHECK: scf.for %{{.*}} = %[[C2]] to %[[C14:.*]] step %[[C4]]
// CHECK: scf.for %{{.*}} = %[[C14]] to %[[C17]] step %[[C4]]

// -----

// Multiple iter_args chained across peeled loops.
func.func @peel_last_multiple_iter_args(%lb: index, %ub: index) -> (i32, f32) {
  %c0_i32 = arith.constant 0 : i32
  %c0_f32 = arith.constant 0.0 : f32
  %step = arith.constant 4 : index
  %r:2 = scf.for %iv = %lb to %ub step %step
               iter_args(%a1 = %c0_i32, %a2 = %c0_f32) -> (i32, f32) {
    %one_i32 = arith.constant 1 : i32
    %one_f32 = arith.constant 1.0 : f32
    %n1 = arith.addi %a1, %one_i32 : i32
    %n2 = arith.addf %a2, %one_f32 : f32
    scf.yield %n1, %n2 : i32, f32
  }
  return %r#0, %r#1 : i32, f32
}
// CHECK-LABEL: func @peel_last_multiple_iter_args
// CHECK: %[[MAIN:.*]]:2 = scf.for %{{.*}} = %arg0 to %{{.*}} step %c4 iter_args(%{{.*}} = %c0_i32, %{{.*}} = %{{.*}}) -> (i32, f32)
// CHECK: %[[PEELED:.*]]:2 = scf.for %{{.*}} = %{{.*}} to %arg1 step %c4 iter_args(%{{.*}} = %[[MAIN]]#0, %{{.*}} = %[[MAIN]]#1) -> (i32, f32)
// CHECK: return %[[PEELED]]#0, %[[PEELED]]#1

// -----

// Unsigned loop peeling.
func.func @peel_last_unsigned(%lb: i32, %ub: i32) {
  %step = arith.constant 4 : i32
  scf.for unsigned %iv = %lb to %ub step %step : i32 {
    "test.use"(%iv) : (i32) -> ()
  }
  return
}
// CHECK-LABEL: func @peel_last_unsigned
// CHECK: arith.minui
// CHECK: arith.subi
// CHECK: arith.remui
// CHECK: arith.cmpi eq
// CHECK: arith.select
// CHECK: arith.subi
// CHECK: scf.for unsigned %{{.*}} = %arg0 to %{{.*}} step %c4_i32 : i32
// CHECK: scf.for unsigned %{{.*}} = %{{.*}} to %arg1 step %c4_i32 : i32

// -----

// Constant empty range: no peel, even when the step is dynamic.
func.func @no_peel_const_empty_dynamic_step(%step: index) {
  %lb = arith.constant 4 : index
  %ub = arith.constant 0 : index
  scf.for %iv = %lb to %ub step %step {
    "test.use"(%iv) : (index) -> ()
  }
  return
}
// CHECK-LABEL: func @no_peel_const_empty_dynamic_step
// CHECK-SAME:  %[[STEP:.*]]: index
// CHECK: %[[C4:.*]] = arith.constant 4 : index
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: scf.for %{{.*}} = %[[C4]] to %[[C0]] step %[[STEP]]
// CHECK-NOT: scf.for
