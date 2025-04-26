
// RUN: mlir-opt %s --split-input-file --verify-diagnostics

func.func @loop_result_mismatch(%value : f32) {
 // expected-error @+1 {{'scf.loop' op along control flow edge from Operation scf.break to Operation scf.loop: successor operand type #0 'f32' should match successor input type #0 'i32'}}
 %result = scf.loop token(%loop) -> i32 {
   scf.break [%loop] %value : f32 // expected-note {{region branch point}}
 }
 return
}

// -----

// A PropagateControlFlowBreak op that is immediately nested in a
// RegionBranchOpInterface can contain a region terminator that targets an
// ancestor of that RegionBranchOpInterface. The intermediate RegionBranchOp
// must expose this as a possible propagated edge so the addressed receiver can
// verify the break operands against its results.
func.func @outer_loop_result_mismatch_through_if_in_inner_loop(%cond : i1,
                                                              %value : f32) {
 // expected-error @+1 {{'scf.loop' op along control flow edge from Operation scf.break to Operation scf.loop: successor operand type #0 'f32' should match successor input type #0 'i32'}}
 %result = scf.loop token(%outer) -> i32 {
   scf.loop token(%inner) {
     scf.if %cond {
       scf.break [%outer] %value : f32 // expected-note {{region branch point}}
     }
     scf.continue [%inner]
   }
   scf.continue [%outer]
 }
 return
}

// -----

func.func @loop_result_number_mismatch(%value : f32) {
 // expected-error @+1 {{'scf.loop' op along control flow edge from Operation scf.break to Operation scf.loop: region branch point has 1 operands, but region successor needs 2 inputs}}
 %result:2 = scf.loop token(%loop) -> f32, f32 {
   scf.break [%loop] %value : f32 // expected-note {{region branch point}}
 }
 return
}

// -----

func.func @loop_continue_mismatch(%init : i32, %value : f32) {
 // expected-error @+1 {{'scf.loop' op along control flow edge from Operation scf.continue to Region #0: successor operand type #0 'f32' should match successor input type #0 'i32'}}
 scf.loop token(%loop) iter_args(%next = %init) : i32 {
   scf.continue [%loop] %value : f32 // expected-note {{region branch point}}
 }
 return
}


// -----

func.func @loop_iterargs_mismatch(%init : i32, %value : f32) {
 // expected-error @+2 {{'scf.loop' op along control flow edge from parent to Region #0: successor operand type #0 'i32' should match successor input type #0 'f32'}}
 // expected-note @+1 {{region branch point}}
 "scf.loop"(%init) ({
    ^body(%token : token, %next : f32):
   scf.continue [%token] %init : i32
 })  : (i32) -> ()
 return
}

// -----

func.func @loop_iterargs_mismatch(%init : i32, %value : f32) {
 // expected-error @+2 {{'scf.loop' op along control flow edge from parent to Region #0: region branch point has 1 operands, but region successor needs 2 inputs}}
 // expected-note @+1 {{region branch point}}
 "scf.loop"(%init) ({
    ^body(%token : token, %next : i32, %next2 : f32):
   scf.continue [%token] %init : i32
 })  : (i32) -> ()
 return
}

// -----

// scf.for lacks PropagateControlFlowBreak, so it cannot be an intermediate
// parent for a break targeting an enclosing loop token.
func.func @break_through_for_missing_trait(%lb: index, %ub: index, %step: index, %cond: i1) {
  scf.loop token(%loop) {
    scf.for %i = %lb to %ub step %step {
      scf.if %cond {
        // expected-error @+1 {{target token crosses an op that does not have the PropagateControlFlowBreak trait}}
        scf.break [%loop]
      }
    }
  }
  return
}

// -----

// scf.while lacks PropagateControlFlowBreak, so break through it is rejected.
// The scf.while verifier catches this first as the after region must terminate
// with scf.yield.
func.func @break_through_while(%cond: i1) {
  %init = arith.constant true
  scf.loop token(%loop) {
    // expected-error @+1 {{'scf.while' op expects the 'after' region to terminate with 'scf.yield'}}
    scf.while(%arg = %init) : (i1) -> i1 {
      scf.condition(%arg) %arg : i1
    } do {
    ^bb0(%arg2: i1):
      // expected-note @+1 {{terminator here}}
      scf.break [%loop]
    }
  }
  return
}

// -----

// Verify that a break without a token target is rejected.
func.func @break_without_token() {
  scf.loop token(%loop) {
    // expected-error @+1 {{'scf.break' op expected 1 or more operands}}
    "scf.break"() : () -> ()
  }
  return
}

// -----

// Verify that a loop whose first region argument is not a control token is
// rejected. The break targets the outer loop so the inner loop does not need a
// self-targeting terminator, isolating the control-token check.
func.func @loop_non_token_control(%cond: i1) {
  "scf.loop"() ({
  ^bb0(%outer: token):
    // expected-error @+1 {{'scf.loop' op first region argument must be a control token}}
    "scf.loop"() ({
    ^bb1(%bad: i32):
      "scf.break"(%outer) : (token) -> ()
    }) : () -> ()
    "scf.continue"(%outer) : (token) -> ()
  }) : () -> ()
  return
}
