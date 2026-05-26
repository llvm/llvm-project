// RUN: mlir-opt -convert-scf-to-cf -split-input-file -verify-diagnostics %s

// Lowering of `scf.if` whose then/else region is terminated by an
// `scf.break` / `scf.continue` (region-breaking terminator) is not yet
// implemented. The pattern reports a match failure and the partial
// conversion driver leaves the op unconverted, which causes the pass to
// fail. `scf.loop` is not in the conversion target, so it stays legal.

func.func @if_break_unsupported(%cond: i1, %v: i32) -> i32 {
  %r = scf.loop %t -> i32 {
    // expected-error@+1 {{failed to legalize operation 'scf.if' that was explicitly marked illegal}}
    scf.if %cond {
      scf.break %t, %v : token, i32
    }
    scf.continue %t : token
  }
  return %r : i32
}

// -----

func.func @if_continue_unsupported(%cond: i1, %init: i32) {
  scf.loop %t iter_args(%i = %init) : i32 {
    // expected-error@+1 {{failed to legalize operation 'scf.if' that was explicitly marked illegal}}
    scf.if %cond {
      scf.continue %t, %i : token, i32
    }
    scf.continue %t, %i : token, i32
  }
  return
}
