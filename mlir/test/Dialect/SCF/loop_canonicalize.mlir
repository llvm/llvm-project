// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(canonicalize{test-convergence}))' -split-input-file | FileCheck %s

// CHECK-LABEL: func @fold_single_iteration_loop1
func.func @fold_single_iteration_loop1(%arg0 : index) -> index {
  // CHECK-NOT: loop
  %0 = scf.loop token(%loop) -> index {
    scf.break [%loop] %arg0 : index
  }
  return %0 : index
}

// -----

// CHECK-LABEL: func @fold_single_iteration_loop_with_propagating_control_flow
func.func @fold_single_iteration_loop_with_propagating_control_flow(%cond : i1, %arg0 : index) -> index {
  %0 = scf.loop token(%outer) -> index {
    scf.loop token(%inner) {
      scf.if %cond {
        scf.break [%outer] %arg0 : index
      }
      scf.break [%inner]
    }
  }
  return %0 : index
}

// -----

// CHECK-LABEL: func @loop_not_combine_ifs
func.func @loop_not_combine_ifs(%arg0 : i1, %arg2: i64) {
  // Verify that we don't combine ifs when terminator mismatches
  scf.loop token(%loop) {
    // CHECK: scf.if
    %res = scf.if %arg0 -> i32 {
      %v = "test.firstCodeTrue"() : () -> i32
      scf.yield %v : i32
    } else {
      %v2 = "test.firstCodeFalse"() : () -> i32
      scf.break [%loop]
    }
    // CHECK: scf.if
    %res2 = scf.if %arg0 -> i32 {
      %v = "test.secondCodeTrue"() : () -> i32
      scf.yield %v : i32
    } else {
      %v2 = "test.secondCodeFalse"() : () -> i32
      scf.continue [%loop]
    }
  }
  return
}

// -----

// CHECK-LABEL: func @loop_combine_ifs
func.func @loop_combine_ifs(%arg0 : i1, %arg2: i64) {
  // CombineIfs is yield-only, so breaking terminators remain separate.
  scf.loop token(%loop) {
    // CHECK: scf.if
    %res = scf.if %arg0 -> i32 {
      %v = "test.firstCodeTrue"() : () -> i32
      scf.yield %v : i32
    } else {
      %v2 = "test.firstCodeFalse"() : () -> i32
      scf.break [%loop]
    }
    %res2 = scf.if %arg0 -> i32 {
      %v = "test.secondCodeTrue"() : () -> i32
      scf.yield %v : i32
    } else {
      %v2 = "test.secondCodeFalse"() : () -> i32
      scf.break [%loop]
    }
  }
  return
}

// -----

// CHECK-LABEL: @do_not_merge_nested_if_with_breaking_control_flow1
func.func @do_not_merge_nested_if_with_breaking_control_flow1(%arg0: i1, %arg1: i1) {
// The outer if then terminator isn't a yield, blocking the merge.
// CHECK: scf.loop
// CHECK: scf.if
// CHECK: scf.if
  scf.loop token(%loop) {
    scf.if %arg0 {
      scf.if %arg1 {
        "test.op"() : () -> ()
        scf.yield
      }
      scf.break [%loop]
    }
  }
  return
}

// -----

// CHECK-LABEL: @do_not_merge_nested_if_with_breaking_control_flow2
func.func @do_not_merge_nested_if_with_breaking_control_flow2(%arg0: i1, %arg1: i1) {
// The outer if else block terminator isn't a yield, blocking the merge.
// CHECK: scf.loop
// CHECK: scf.if
// CHECK: scf.if
// CHECK: else
// CHECK-NEXT: scf.break
  scf.loop token(%loop) {
    scf.if %arg0 {
      scf.if %arg1 {
        "test.op"() : () -> ()
        scf.yield
      }
      scf.yield
    } else {
      scf.break [%loop]
    }
  }
  return
}

// -----

// CHECK-LABEL: @do_not_merge_nested_if_with_breaking_control_flow3
func.func @do_not_merge_nested_if_with_breaking_control_flow3(%arg0: i1, %arg1: i1) {
// The nested if then block terminator isn't a yield, blocking the merge.
// CHECK: scf.loop
// CHECK: scf.if
// CHECK: scf.if
// CHECK: test.op
// CHECK-NEXT: scf.break
  scf.loop token(%loop) {
    scf.if %arg0 {
      scf.if %arg1 {
        "test.op"() : () -> ()
        scf.break [%loop]
      }
    }
  }
  return
}

// -----

// CHECK-LABEL: @do_not_merge_nested_if_with_breaking_control_flow4
func.func @do_not_merge_nested_if_with_breaking_control_flow4(%arg0: i1, %arg1: i1) {
// The nested if else block terminator isn't a yield, blocking the merge.
// CHECK: scf.loop
// CHECK: scf.if
// CHECK: scf.if
// CHECK: else
// CHECK-NEXT: scf.break
  scf.loop token(%loop) {
    scf.if %arg0 {
      scf.if %arg1 {
        "test.op"() : () -> ()
      } else {
        scf.break [%loop]
      }
    }
  }
  return
}

// -----

// CHECK-LABEL: func @do_not_convert_if_to_select1
func.func @do_not_convert_if_to_select1(%cond: i1, %arg0 : index, %arg1 : index) -> index {
  %loop_res  = scf.loop token(%loop) -> index {
    // Inner then terminator is not a yield, blocking transform to select.
    // CHECK: scf.if
    %0 = scf.if %cond -> index {
      scf.break [%loop] %arg0 : index
    } else {
      scf.yield %arg1 : index
    }
    scf.break [%loop] %0 : index
  }
  return %loop_res : index
}

// -----

// CHECK-LABEL: func @do_not_convert_if_to_select2
func.func @do_not_convert_if_to_select2(%cond: i1, %arg0 : index, %arg1 : index) -> index {
  %loop_res  = scf.loop token(%loop) -> index {
    // Inner then terminator is not a yield, blocking transform to select.
    // CHECK: scf.if
    %0 = scf.if %cond -> index {
      scf.yield %arg0 : index
    } else {
      scf.break [%loop] %arg1 : index
    }
    scf.break [%loop] %0 : index
  }
  return %loop_res : index
}


// -----

// Verify that removing the unused results of an if with nested breaking control flow
// operation works.
// CHECK-LABEL: func @remove_unused_if_results1
func.func @remove_unused_if_results1(%cond : i1, %arg0 : index) -> index {
  // CHECK: scf.loop
  %0 = scf.loop token(%loop) -> index {
    // CHECK: %[[FOO:.*]]:3 = "test.foo"
    %foo:3 = "test.foo" () : () -> (i32, i64, index)
    // CHECK-NOT: %[[RES:.*]] = scf.if
    // CHECK: scf.if
    %res:3 = scf.if %cond -> (i32, i64, index) {
      // CHECK: scf.yield
      scf.yield %foo#0, %foo#1, %foo#2 : i32, i64, index
    } else {
      // CHECK: scf.break {{.*}} %[[FOO]]#2 : index
      scf.break [%loop] %foo#2 : index
    }
      // CHECK: "test.op"(%[[FOO]]#1)
    "test.op"(%res#1) : (i64) -> ()
  }
  return %0 : index
}

// -----

// Verify that removing the unused results of an if with nested breaking control flow
// operation works.
// CHECK-LABEL: func @simplify_if_with_breaking_controlflow_in_both_branches
func.func @simplify_if_with_breaking_controlflow_in_both_branches(%cond : i1, %cond2 : i1, %arg0 : index) -> index {
  // CHECK: scf.loop
  %0 = scf.loop token(%loop) -> index {
    // CHECK: %[[FOO:.*]] = "test.foo"
    %foo = "test.foo" () : () -> (index)
    // CHECK: scf.if
    scf.if %cond {
      // CHECK: scf.break {{.*}} %[[FOO]] : index
      scf.break [%loop] %foo : index
      // CHECK-NOT: else
    } else {
      // CHECK: %[[BAR:.*]] = "test.bar"
      %bar = "test.bar" () : () -> (index)
      // CHECK: scf.if
      scf.if %cond2 {
        // verify that this is correctly updated when inlining the parent region.
        // CHECK: scf.break {{.*}} %[[BAR]] : index
        scf.break [%loop] %bar : index
      }
      scf.continue [%loop]
    }
    "test.op"() : () -> ()
  }
  return %0 : index
}

// -----

// Verify simplification when both branches have continue.
// CHECK-LABEL: func @simplify_if_with_continue_in_both_branches
func.func @simplify_if_with_continue_in_both_branches(%cond : i1, %init : i32) {
  // CHECK: scf.loop
  scf.loop token(%loop) iter_args(%arg = %init) : i32 {
    // CHECK: scf.if
    scf.if %cond {
      // CHECK: scf.continue {{.*}} %{{.*}} : i32
      scf.continue [%loop] %arg : i32
    } else {
      scf.continue [%loop] %arg : i32
    }
    // The dead code after the if should be removed.
    // CHECK-NOT: test.op
    "test.op"() : () -> ()
    scf.continue [%loop] %arg : i32
  }
  return
}

// -----

// Verify simplification when both branches have break.
// CHECK-LABEL: func @simplify_if_with_break_in_both_branches
func.func @simplify_if_with_break_in_both_branches(%cond : i1) -> index {
  // CHECK: scf.loop
  %0 = scf.loop token(%loop) -> index {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // CHECK: scf.if
    scf.if %cond {
      // CHECK: scf.break
      scf.break [%loop] %c0 : index
    } else {
      scf.break [%loop] %c1 : index
    }
    // Dead code after the if.
    // CHECK-NOT: test.op
    "test.op"() : () -> ()
    scf.continue [%loop]
  }
  return %0 : index
}
