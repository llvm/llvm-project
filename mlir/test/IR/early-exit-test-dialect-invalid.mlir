// RUN: mlir-opt %s --split-input-file --verify-diagnostics

func.func @non_index_depth(%depth: i32) {
  test.breakable_loop {
    // expected-error @+1 {{operand #0 must be index}}
    "test.dynamic_break"(%depth) : (i32) -> ()
  }
  return
}

// -----

func.func @no_reachable_loop(%depth: index) {
  // expected-error @+1 {{must be nested inside a reachable test.breakable_loop}}
  test.dynamic_break %depth
}

// -----

func.func @zero_depth() {
  %c0 = arith.constant 0 : index
  test.breakable_loop {
    // expected-error @+1 {{depth must be positive}}
    test.dynamic_continue %c0
  }
  return
}

// -----

func.func @too_deep() {
  %c2 = arith.constant 2 : index
  test.breakable_loop {
    // expected-error @+1 {{constant depth exceeds the number of reachable test.breakable_loop operations}}
    test.dynamic_break %c2
  }
  return
}

// -----

func.func @blocked_parent() {
  %c1 = arith.constant 1 : index
  test.breakable_loop {
    "test.any_cond"() ({
      // expected-error @+1 {{depth target crosses an op that does not have the PropagateControlFlowBreak trait}}
      test.dynamic_break %c1
    }) : () -> ()
    test.dynamic_continue %c1
  }
  return
}

// -----

func.func @dynamic_depth_blocked_outer(%depth: index) {
  %c1 = arith.constant 1 : index
  test.breakable_loop {
    test.single_no_terminator_custom_asm_op {
      test.breakable_loop {
        // expected-error @+1 {{dynamic depth may target across an op that does not have the PropagateControlFlowBreak trait}}
        test.dynamic_break %depth
      }
    }
    test.dynamic_continue %c1
  }
  return
}

// -----

// Constant depths select one concrete loop, so payload mismatches are still
// diagnosed against that selected target.
func.func @constant_break_payload_mismatch(%value: f32) -> i32 {
  %c1 = arith.constant 1 : index
  %result = test.breakable_loop -> i32 {
    // expected-error @+1 {{payload operand #0 has type 'f32', but target results}}
    test.dynamic_break %c1 %value : f32
  }
  return %result : i32
}

// -----

// Dynamic depths filter out incompatible targets. This is still invalid when
// filtering leaves no target that can accept the terminator payload.
func.func @dynamic_break_no_compatible_target(%depth: index, %value: f32) -> i32 {
  %result = test.breakable_loop -> i32 {
    // expected-error @+1 {{dynamic depth has no compatible test.breakable_loop target}}
    test.dynamic_break %depth %value : f32
  }
  return %result : i32
}
