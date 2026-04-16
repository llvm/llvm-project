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

// -----

// This should not crash.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.memref.extract_address_computations
    } : !transform.any_op
    transform.yield
  }
}

// -----

// collectFreedValues should not crash on ops that don't implement
// MemoryEffectOpInterface (e.g. pdl ops inside with_pdl_patterns).
// https://github.com/llvm/llvm-project/issues/120944

// CHECK-LABEL: func @foo
func.func @foo(%arg0: index, %arg1: index, %arg2: index) {
  scf.for %i = %arg0 to %arg1 step %arg2 {
    %0 = arith.constant 0 : i32
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @match_const : benefit(1) {
        %0 = pdl.operands
        %1 = pdl.types
        %2 = pdl.operation "arith.constant"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        pdl.rewrite %2 with "transform.dialect"
      }
      sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        %0 = transform.pdl_match @match_const in %arg1 : (!transform.any_op) -> !transform.any_op
        %1 = transform.get_parent_op %0 {op_name = "scf.for"} : (!transform.any_op) -> !transform.any_op
        alternatives %1 : !transform.any_op {
        ^bb2(%arg2: !transform.any_op):
        }
      }
    }
    transform.yield
  }
}
