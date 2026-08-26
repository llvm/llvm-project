// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline="builtin.module(func.func(sccp))" -split-input-file | FileCheck %s

/// The inner loop's only normal exit (break 1) carries the constant 5.
/// The early-exit path (break 3) carries -5 but bypasses the code after the
/// inner loop entirely — it exits both loops at once.
///
/// Without early-exit support the analysis would conservatively join {5, -5}
/// and mark %inner as overdefined, keeping the dead branch alive.
/// With early-exit support SCCP sees that %inner is always 5, folds the
/// comparison, and removes the dead scf.if.
///
/// Pseudocode:
///   loop {
///     a = loop {
///       if (cond) { break_all -5; }  // early exit from both loops
///       break 5;                      // normal exit from inner loop
///     };
///     // here a == 5 always
///     if (a < 0) "dead"();            // dead code
///     break a;
///   }

// CHECK-LABEL: func @early_exit_dead_code(
// CHECK-DAG:     %[[FALSE:.*]] = arith.constant false
// CHECK-DAG:     %[[C5:.*]] = arith.constant 5 : i32
// CHECK:         scf.loop
// CHECK:           scf.loop
// CHECK:           scf.if %[[FALSE]]
// CHECK:             "test.dead_op"
// CHECK:           scf.break {{.*}} %[[C5]] : i32
func.func @early_exit_dead_code(%cond: i1) -> i32 {
  %c0 = arith.constant 0 : i32
  %c5 = arith.constant 5 : i32
  %cm5 = arith.constant -5 : i32
  %outer = scf.loop token(%outer_token) -> i32 {
    %inner = scf.loop token(%inner_token) -> i32 {
      scf.if %cond {
        scf.break [%outer_token] %cm5 : i32
      }
      scf.break [%inner_token] %c5 : i32
    }
    %is_neg = arith.cmpi slt, %inner, %c0 : i32
    scf.if %is_neg {
      "test.dead_op"() : () -> ()
    }
    scf.break [%outer_token] %inner : i32
  }
  return %outer : i32
}

// -----

/// For contrast: the same logic emulated with scf.while and a boolean flag
/// instead of early exit.  Without break 3 to bypass the code after the inner
/// loop, both values (-5 and 5) flow through %inner and SCCP sees it as
/// overdefined — the comparison and dead branch survive.
///
/// Pseudocode:
///   done = false
///   while (!done) {
///     a, should_break = while (!done) {
///       if (cond) yield -5, true   // want to break all, but can't
///       else      yield  5, false
///     };
///     if (!should_break) {
///       // a ∈ {5, -5} — flag can't help the analysis narrow it
///       if (a < 0) "not dead"();
///     }
///     done = true
///   }

// CHECK-LABEL: func @no_early_exit_not_foldable(
// CHECK:         scf.while
// CHECK:           scf.while
// CHECK:           scf.if
// CHECK:             arith.cmpi slt,
// CHECK:             scf.if
// CHECK:               "test.not_dead_op"
func.func @no_early_exit_not_foldable(%cond: i1) -> i32 {
  %c0 = arith.constant 0 : i32
  %c5 = arith.constant 5 : i32
  %cm5 = arith.constant -5 : i32
  %false = arith.constant false
  %true = arith.constant true

  // Outer while: (result, done)
  %outer, %_ = scf.while(%o_res = %c0, %o_done = %false) : (i32, i1) -> (i32, i1) {
    %o_go = arith.xori %o_done, %true : i1
    scf.condition(%o_go) %o_res, %o_done : i32, i1
  } do {
  ^bb0(%o_res: i32, %o_done: i1):
    // Inner while: (result, done, should_break_outer)
    %inner, %_2, %break_flag = scf.while(%i_res = %c0, %i_done = %false, %i_brk = %false)
        : (i32, i1, i1) -> (i32, i1, i1) {
      %i_go = arith.xori %i_done, %true : i1
      scf.condition(%i_go) %i_res, %i_done, %i_brk : i32, i1, i1
    } do {
    ^bb0(%i_res: i32, %i_done: i1, %i_brk: i1):
      // if (cond) a = -5, flag = true; else a = 5, flag = false
      %a = arith.select %cond, %cm5, %c5 : i32
      %brk = arith.select %cond, %true, %false : i1
      scf.yield %a, %true, %brk : i32, i1, i1
    }
    // %inner ∈ {5, -5}: both values reach here (no early exit to bypass)
    // %break_flag correlates with %inner but SCCP can't exploit that
    %not_break = arith.xori %break_flag, %true : i1
    scf.if %not_break {
      // %inner ∈ {5} in theory: but the analysis can't recover this here.
      %is_neg = arith.cmpi slt, %inner, %c0 : i32
      scf.if %is_neg {
        "test.not_dead_op"() : () -> ()
      }
    }
    scf.yield %inner, %true : i32, i1
  }
  return %outer : i32
}
