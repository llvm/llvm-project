// RUN: mlir-opt --int-range-optimizations %s | FileCheck %s

// Verify that `IntegerRangeAnalysis` threads integer bounds across early-exit
// edges (`scf.break`) modeled by the `RegionBranchOpInterface`. The result of
// the `scf.execute_region` is the union of the bounds along the early-exit
// (`scf.break`) path and the fall-through (`scf.yield`) path.

// The result is either 0 (break) or 10 (yield), i.e. in [0, 10]. If the
// early-exit edge were not modeled, only the yield (10) would be seen.

// CHECK-LABEL: func @early_exit_range
// CHECK: test.reflect_bounds {smax = 10 : si32, smin = 0 : si32, umax = 10 : ui32, umin = 0 : ui32}
func.func @early_exit_range(%cond: i1) -> i32 {
  %0 = scf.execute_region -> i32 {
  ^bb0(%tok: token):
    %c0 = arith.constant 0 : i32
    %c10 = arith.constant 10 : i32
    scf.if %cond {
      scf.break %tok, %c0 : i32
    }
    scf.yield %c10 : i32
  }
  %r = test.reflect_bounds %0 : i32
  return %r : i32
}

// -----

// Because the result is provably in [0, 5], the comparison `%0 < 100` folds to
// `true`. This is the dead-code-elimination scenario enabled by early exit:
// without the early-exit edge the analysis would still see only the yield, but
// the point here is that the break value participates in the bound.

// CHECK-LABEL: func @early_exit_dead_code
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: return %[[TRUE]] : i1
func.func @early_exit_dead_code(%cond: i1) -> i1 {
  %0 = scf.execute_region -> i32 {
  ^bb0(%tok: token):
    %c0 = arith.constant 0 : i32
    %c5 = arith.constant 5 : i32
    scf.if %cond {
      scf.break %tok, %c0 : i32
    }
    scf.yield %c5 : i32
  }
  %c100 = arith.constant 100 : i32
  %cmp = arith.cmpi slt, %0, %c100 : i32
  return %cmp : i1
}

// -----

// A break that targets the *outer* `scf.execute_region` skips the inner one.
// The outer result is either 1 (break) or 7 (inner yield forwarded by the
// outer yield), i.e. in [1, 7].

// CHECK-LABEL: func @outer_early_exit_range
// CHECK: test.reflect_bounds {smax = 7 : si32, smin = 1 : si32, umax = 7 : ui32, umin = 1 : ui32}
func.func @outer_early_exit_range(%cond: i1) -> i32 {
  %0 = scf.execute_region -> i32 {
  ^bb0(%tok_outer: token):
    %c1 = arith.constant 1 : i32
    %1 = scf.execute_region -> i32 {
    ^bb1(%tok_inner: token):
      %c7 = arith.constant 7 : i32
      scf.if %cond {
        scf.break %tok_outer, %c1 : i32
      }
      scf.yield %c7 : i32
    }
    scf.yield %1 : i32
  }
  %r = test.reflect_bounds %0 : i32
  return %r : i32
}

// -----

// The immediate (and only) terminator of the `scf.execute_region` is an
// `scf.break`. The result is exactly the broken-out value, so it is in [3, 3].

// CHECK-LABEL: func @immediate_break
// CHECK: test.reflect_bounds {smax = 3 : si32, smin = 3 : si32, umax = 3 : ui32, umin = 3 : ui32}
func.func @immediate_break(%cond: i1) -> i32 {
  %0 = scf.execute_region -> i32 {
  ^bb0(%tok: token):
    %c3 = arith.constant 3 : i32
    scf.break %tok, %c3 : i32
  }
  %r = test.reflect_bounds %0 : i32
  return %r : i32
}

// -----

// All terminators of the inner `scf.execute_region` break to the outer one, so
// the inner result is never produced and the analysis infers no bounds for it
// (the `test.reflect_bounds` on it stays unannotated). The outer result is
// exactly the break value 4.

// CHECK-LABEL: func @range_all_break_outer
// The inner (never-produced) result has no inferred bounds:
// CHECK: test.reflect_bounds %{{[a-z0-9_]+}} : i32
// The outer result is the break value, in [4, 4]:
// CHECK: test.reflect_bounds {smax = 4 : si32, smin = 4 : si32, umax = 4 : ui32, umin = 4 : ui32}
func.func @range_all_break_outer(%cond: i1) -> i32 {
  %0 = scf.execute_region -> i32 {
  ^bb0(%tok_outer: token):
    %c4 = arith.constant 4 : i32
    %1 = scf.execute_region -> i32 {
    ^bb1(%tok_inner: token):
      scf.break %tok_outer, %c4 : i32
    }
    %r1 = test.reflect_bounds %1 : i32
    scf.yield %r1 : i32
  }
  %r = test.reflect_bounds %0 : i32
  return %r : i32
}

// -----

// Both exits are breaks to the same `scf.execute_region`: a nested one (early
// exit through the `scf.if`) and the block's immediate terminator. The result
// is either 2 or 8, i.e. in [2, 8]. This covers both the "early exit through a
// transparent op" and the "immediate-terminator break" edges for one op.

// CHECK-LABEL: func @nested_and_immediate_break
// CHECK: test.reflect_bounds {smax = 8 : si32, smin = 2 : si32, umax = 8 : ui32, umin = 2 : ui32}
func.func @nested_and_immediate_break(%cond: i1) -> i32 {
  %0 = scf.execute_region -> i32 {
  ^bb0(%tok: token):
    %c2 = arith.constant 2 : i32
    %c8 = arith.constant 8 : i32
    scf.if %cond {
      scf.break %tok, %c2 : i32
    }
    scf.break %tok, %c8 : i32
  }
  %r = test.reflect_bounds %0 : i32
  return %r : i32
}
