// RUN: mlir-opt %s --transform-dialect-check-uses --split-input-file --verify-diagnostics

func.func @use_after_free_branching_control_flow() {
  // expected-note @below {{allocated here}}
  %0 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
  transform.test_transform_op_with_regions {
    "transform.test_branching_transform_op_terminator"() : () -> ()
  },
  {
  ^bb0:
    "transform.test_branching_transform_op_terminator"()[^bb1, ^bb2] : () -> ()
  ^bb1:
    // expected-note @below {{freed here}}
    transform.test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
    "transform.test_branching_transform_op_terminator"()[^bb3] : () -> ()
  ^bb2:
    "transform.test_branching_transform_op_terminator"()[^bb3] : () -> ()
  ^bb3:
    // expected-warning @below {{operand #0 may be used after free}}
    transform.sequence %0 : !transform.any_op failures(propagate) {
    ^bb0(%arg0: !transform.any_op):
    }
    "transform.test_branching_transform_op_terminator"() : () -> ()
  }
  return
}

// -----

func.func @use_after_free_in_nested_op() {
  // expected-note @below {{allocated here}}
  %0 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
  // expected-note @below {{freed here}}
  transform.test_transform_op_with_regions {
    "transform.test_branching_transform_op_terminator"() : () -> ()
  },
  {
  ^bb0:
    "transform.test_branching_transform_op_terminator"()[^bb1, ^bb2] : () -> ()
  ^bb1:
    transform.test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
    "transform.test_branching_transform_op_terminator"()[^bb3] : () -> ()
  ^bb2:
    "transform.test_branching_transform_op_terminator"()[^bb3] : () -> ()
  ^bb3:
    "transform.test_branching_transform_op_terminator"() : () -> ()
  }
  // expected-warning @below {{operand #0 may be used after free}}
  transform.sequence %0 : !transform.any_op failures(propagate) {
    ^bb0(%arg0: !transform.any_op):
  }
  return
}

// -----

func.func @use_after_free_recursive_side_effects() {
  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    // expected-note @below {{allocated here}}
    %0 = transform.sequence %arg0 : !transform.any_op -> !transform.any_op failures(propagate) attributes { ord = 1 } {
    ^bb1(%arg1: !transform.any_op):
      yield %arg1 : !transform.any_op
    }
    transform.sequence %0 : !transform.any_op failures(propagate) attributes { ord = 2 } {
    ^bb2(%arg2: !transform.any_op):
    }
    transform.sequence %0 : !transform.any_op failures(propagate) attributes { ord = 3 } {
    ^bb3(%arg3: !transform.any_op):
    }

    // `transform.sequence` has recursive side effects so it has the same "free"
    // as the child op it contains.
    // expected-note @below {{freed here}}
    transform.sequence %0 : !transform.any_op failures(propagate) attributes { ord = 4 } {
    ^bb4(%arg4: !transform.any_op):
      test_consume_operand_of_op_kind_or_fail %0, "transform.sequence" : !transform.any_op
    }
    // expected-warning @below {{operand #0 may be used after free}}
    transform.sequence %0 : !transform.any_op failures(propagate) attributes { ord = 5 } {
    ^bb3(%arg3: !transform.any_op):
    }
  }
  return
}

// -----

func.func @use_after_free() {
  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    // expected-note @below {{allocated here}}
    %0 = transform.sequence %arg0 : !transform.any_op -> !transform.any_op failures(propagate) attributes { ord = 1 } {
    ^bb1(%arg1: !transform.any_op):
      yield %arg1 : !transform.any_op
    }
    transform.sequence %0 : !transform.any_op failures(propagate) attributes { ord = 2 } {
    ^bb2(%arg2: !transform.any_op):
    }
    transform.sequence %0 : !transform.any_op failures(propagate) attributes { ord = 3 } {
    ^bb3(%arg3: !transform.any_op):
    }

    // expected-note @below {{freed here}}
    test_consume_operand_of_op_kind_or_fail %0, "transform.sequence" : !transform.any_op
    // expected-warning @below {{operand #0 may be used after free}}
    transform.sequence %0 : !transform.any_op failures(propagate) attributes { ord = 5 } {
    ^bb3(%arg3: !transform.any_op):
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
  %0 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
  transform.test_transform_op_with_regions {
    "transform.test_branching_transform_op_terminator"() : () -> ()
  },
  {
  ^bb0:
    "transform.test_branching_transform_op_terminator"()[^bb1] : () -> ()
  ^bb1:
    // expected-warning @below {{operand #0 may be used after free}}
    transform.sequence %0 : !transform.any_op failures(propagate) {
    ^bb0(%arg0: !transform.any_op):
    }
    // expected-warning @below {{operand #0 may be used after free}}
    // expected-note @below {{freed here}}
    transform.test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
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
  %0 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
  transform.test_transform_op_with_regions {
    "transform.test_branching_transform_op_terminator"() : () -> ()
  },
  {
  ^bb0:
    "transform.test_branching_transform_op_terminator"()[^bb1, ^bb2] : () -> ()
  ^bb1:
    // expected-warning @below {{operand #0 may be used after free}}
    // expected-note @below {{freed here}}
    transform.test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
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
^bb0(%arg0: !transform.any_op):
  alternatives %arg0 : !transform.any_op {
  ^bb0(%arg1: !transform.any_op):
  }
}
