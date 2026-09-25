// RUN: mlir-opt --int-range-optimizations %s | FileCheck %s

// Verify that `IntegerRangeAnalysis` infers tight bounds for loop-carried
// values that are structurally bounded inside the loop body (via
// `arith.minsi`, `arith.andi`, etc.). Convergence is guaranteed by the
// per-state widening budget on `IntegerValueRangeLattice`; the budget is
// large enough that these naturally bounded ratchets reach a fixpoint
// without being widened to `[INT_MIN, INT_MAX]`.

// CHECK-LABEL: func @bounded_acc_for
// CHECK: test.reflect_bounds {smax = 10 : si32, smin = 0 : si32, umax = 10 : ui32, umin = 0 : ui32}
func.func @bounded_acc_for(%n: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c10 = arith.constant 10 : i32
  %res = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %c0) -> i32  : i32 {
    %incr = arith.addi %acc, %c1 : i32
    %clamped = arith.minsi %incr, %c10 : i32
    scf.yield %clamped : i32
  }
  %r = test.reflect_bounds %res : i32
  return %r : i32
}

// The `arith.cmpi slt, %acc, 100` should fold to `true` once the analysis
// proves the iter arg stays in `[0, 10]`, exposing a downstream
// optimization that the previous yield-based widening masked.
// CHECK-LABEL: func @bounded_acc_while
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: scf.condition(%[[TRUE]])
// CHECK: test.reflect_bounds {smax = 10 : si32, smin = 0 : si32, umax = 10 : ui32, umin = 0 : ui32}
func.func @bounded_acc_while() -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c10 = arith.constant 10 : i32
  %c100 = arith.constant 100 : i32
  %res = scf.while (%acc = %c0) : (i32) -> i32 {
    %cond = arith.cmpi slt, %acc, %c100 : i32
    scf.condition(%cond) %acc : i32
  } do {
  ^bb0(%a: i32):
    %incr = arith.addi %a, %c1 : i32
    %clamped = arith.minsi %incr, %c10 : i32
    scf.yield %clamped : i32
  }
  %r = test.reflect_bounds %res : i32
  return %r : i32
}

// CHECK-LABEL: func @bounded_mask_for
// CHECK: test.reflect_bounds {smax = 15 : si32, smin = 0 : si32, umax = 15 : ui32, umin = 0 : ui32}
func.func @bounded_mask_for(%n: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c15 = arith.constant 15 : i32
  %res = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %c0) -> i32  : i32 {
    %incr = arith.addi %acc, %c1 : i32
    %masked = arith.andi %incr, %c15 : i32
    scf.yield %masked : i32
  }
  %r = test.reflect_bounds %res : i32
  return %r : i32
}
