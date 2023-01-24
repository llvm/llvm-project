// RUN: mlir-opt %s --transform-dialect-check-uses --split-input-file --verify-diagnostics

func.func @use_after_free_branching_control_flow() {
  // expected-note @below {{allocated here}}
  %0 = transform.test_produce_param_or_forward_operand 42
  transform.test_transform_op_with_regions {
    "transform.test_branching_transform_op_terminator"() : () -> ()
  },
  {
  ^bb0:
    "transform.test_branching_transform_op_terminator"()[^bb1, ^bb2] : () -> ()
  ^bb1:
    // expected-note @below {{freed here}}
    transform.test_consume_operand_if_matches_param_or_fail %0[42]
    "transform.test_branching_transform_op_terminator"()[^bb3] : () -> ()
  ^bb2:
    "transform.test_branching_transform_op_terminator"()[^bb3] : () -> ()
  ^bb3:
    // expected-warning @below {{operand #0 may be used after free}}
    transform.sequence %0 : !pdl.operation failures(propagate) {
    ^bb0(%arg0: !pdl.operation):
    }
    "transform.test_branching_transform_op_terminator"() : () -> ()
  }
  return
}

// -----

func.func @use_after_free_in_nested_op() {
  // expected-note @below {{allocated here}}
  %0 = transform.test_produce_param_or_forward_operand 42
  // expected-note @below {{freed here}}
  transform.test_transform_op_with_regions {
    "transform.test_branching_transform_op_terminator"() : () -> ()
  },
  {
  ^bb0:
    "transform.test_branching_transform_op_terminator"()[^bb1, ^bb2] : () -> ()
  ^bb1:
    transform.test_consume_operand_if_matches_param_or_fail %0[42]
    "transform.test_branching_transform_op_terminator"()[^bb3] : () -> ()
  ^bb2:
    "transform.test_branching_transform_op_terminator"()[^bb3] : () -> ()
  ^bb3:
    "transform.test_branching_transform_op_terminator"() : () -> ()
  }
  // expected-warning @below {{operand #0 may be used after free}}
  transform.sequence %0 : !pdl.operation failures(propagate) {
    ^bb0(%arg0: !pdl.operation):
  }
  return
}

// -----

func.func @use_after_free_recursive_side_effects() {
  transform.sequence failures(propagate) {
  ^bb0(%arg0: !pdl.operation):
    // expected-note @below {{allocated here}}
    %0 = transform.sequence %arg0 : !pdl.operation -> !pdl.operation failures(propagate) attributes { ord = 1 } {
    ^bb1(%arg1: !pdl.operation):
      yield %arg1 : !pdl.operation
    }
    transform.sequence %0 : !pdl.operation failures(propagate) attributes { ord = 2 } {
    ^bb2(%arg2: !pdl.operation):
    }
    transform.sequence %0 : !pdl.operation failures(propagate) attributes { ord = 3 } {
    ^bb3(%arg3: !pdl.operation):
    }

    // `transform.sequence` has recursive side effects so it has the same "free"
    // as the child op it contains.
    // expected-note @below {{freed here}}
    transform.sequence %0 : !pdl.operation failures(propagate) attributes { ord = 4 } {
    ^bb4(%arg4: !pdl.operation):
      test_consume_operand_if_matches_param_or_fail %0[42]
    }
    // expected-warning @below {{operand #0 may be used after free}}
    transform.sequence %0 : !pdl.operation failures(propagate) attributes { ord = 5 } {
    ^bb3(%arg3: !pdl.operation):
    }
  }
  return
}

// -----

func.func @use_after_free() {
  transform.sequence failures(propagate) {
  ^bb0(%arg0: !pdl.operation):
    // expected-note @below {{allocated here}}
    %0 = transform.sequence %arg0 : !pdl.operation -> !pdl.operation failures(propagate) attributes { ord = 1 } {
    ^bb1(%arg1: !pdl.operation):
      yield %arg1 : !pdl.operation
    }
    transform.sequence %0 : !pdl.operation failures(propagate) attributes { ord = 2 } {
    ^bb2(%arg2: !pdl.operation):
    }
    transform.sequence %0 : !pdl.operation failures(propagate) attributes { ord = 3 } {
    ^bb3(%arg3: !pdl.operation):
    }

    // expected-note @below {{freed here}}
    test_consume_operand_if_matches_param_or_fail %0[42]
    // expected-warning @below {{operand #0 may be used after free}}
    transform.sequence %0 : !pdl.operation failures(propagate) attributes { ord = 5 } {
    ^bb3(%arg3: !pdl.operation):
    }
  }
  return
}

// -----

// In the case of a control flow cycle, the operation that uses the value may
// precede the one that frees it in the same block. Both operations should
// be reported as use-after-free.
func.func @use_after_free_self_cycle() {
  // expected-note @below {{allocated here}}
  %0 = transform.test_produce_param_or_forward_operand 42
  transform.test_transform_op_with_regions {
    "transform.test_branching_transform_op_terminator"() : () -> ()
  },
  {
  ^bb0:
    "transform.test_branching_transform_op_terminator"()[^bb1] : () -> ()
  ^bb1:
    // expected-warning @below {{operand #0 may be used after free}}
    transform.sequence %0 : !pdl.operation failures(propagate) {
    ^bb0(%arg0: !pdl.operation):
    }
    // expected-warning @below {{operand #0 may be used after free}}
    // expected-note @below {{freed here}}
    transform.test_consume_operand_if_matches_param_or_fail %0[42]
    "transform.test_branching_transform_op_terminator"()[^bb1, ^bb2] : () -> ()
  ^bb2:
    "transform.test_branching_transform_op_terminator"() : () -> ()
  }
  return
}


// -----

// Check that the "free" that happens in a cycle is also reported as potential
// use-after-free.
func.func @use_after_free_cycle() {
  // expected-note @below {{allocated here}}
  %0 = transform.test_produce_param_or_forward_operand 42
  transform.test_transform_op_with_regions {
    "transform.test_branching_transform_op_terminator"() : () -> ()
  },
  {
  ^bb0:
    "transform.test_branching_transform_op_terminator"()[^bb1, ^bb2] : () -> ()
  ^bb1:
    // expected-warning @below {{operand #0 may be used after free}}
    // expected-note @below {{freed here}}
    transform.test_consume_operand_if_matches_param_or_fail %0[42]
    "transform.test_branching_transform_op_terminator"()[^bb2, ^bb3] : () -> ()
  ^bb2:
    "transform.test_branching_transform_op_terminator"()[^bb1] : () -> ()
  ^bb3:
    "transform.test_branching_transform_op_terminator"() : () -> ()
  }
  return
}

// -----

// This should not crash.

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  alternatives %arg0 : !pdl.operation {
  ^bb0(%arg1: !pdl.operation):
  }
}
