// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// expected-error @below {{expects the entry block to have at least one argument}}
transform.sequence failures(propagate) {
}

// -----

// expected-error @below {{expects the first entry block argument to be of type implementing TransformHandleTypeInterface}}
transform.sequence failures(propagate) {
^bb0(%rag0: i64):
}

// -----

// expected-note @below {{nested in another possible top-level op}}
transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  // expected-error @below {{expects operands to be provided for a nested op}}
  transform.sequence failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
  }
}

// -----

// expected-error @below {{'transform.sequence' op expects trailing entry block arguments to be of type implementing TransformHandleTypeInterface or TransformParamTypeInterface}}
// expected-note @below {{argument #1 does not}}
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op, %arg1: i64):
}

// -----

// expected-error @below {{expected children ops to implement TransformOpInterface}}
transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  // expected-note @below {{op without interface}}
  arith.constant 42.0 : f32
}

// -----

// expected-error @below {{expects the types of the terminator operands to match the types of the result}}
%0 = transform.sequence -> !pdl.operation failures(propagate) {
^bb0(%arg0: !pdl.operation):
  // expected-note @below {{terminator}}
  transform.yield
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // expected-error @below {{expects the type of the block argument to match the type of the operand}}
  transform.sequence %arg0: !transform.any_op failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    transform.yield
  }
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op, %arg1: !transform.any_op, %arg2: !transform.any_op):
  // expected-error @below {{expected types to be provided for all operands}}
  transform.sequence %arg0, %arg1, %arg2 : (!transform.any_op, !transform.any_op) failures(propagate) {
  ^bb0(%arg3: !transform.any_op, %arg4: !transform.any_op, %arg5: !transform.any_op):
  }
}

// -----

%0 = "test.generate_something"() : () -> !transform.any_op
// expected-error @below {{does not expect extra operands when used as top-level}}
"transform.sequence"(%0) ({
^bb0(%arg0: !transform.any_op):
  "transform.yield"() : () -> ()
}) {failure_propagation_mode = 1 : i32, operand_segment_sizes = array<i32: 0, 1>} : (!transform.any_op) -> ()

// -----

// expected-note @below {{nested in another possible top-level op}}
transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  // expected-error @below {{expects operands to be provided for a nested op}}
  transform.sequence failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
  }
}

// -----

// expected-error @below {{expects only one non-pattern op in its body}}
transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  // expected-note @below {{first non-pattern op}}
  transform.sequence failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
  }
  // expected-note @below {{second non-pattern op}}
  transform.sequence failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
  }
}

// -----

// expected-error @below {{expects only pattern and top-level transform ops in its body}}
transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  // expected-note @below {{offending op}}
  "test.something"() : () -> ()
}

// -----

// expected-note @below {{parent operation}}
transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
   // expected-error @below {{op cannot be nested}}
  transform.with_pdl_patterns %arg0 : !pdl.operation {
  ^bb1(%arg1: !pdl.operation):
  }
}

// -----

// expected-error @below {{expects at least one region}}
"transform.test_transform_unrestricted_op_no_interface"() : () -> ()

// -----

// expected-error @below {{expects a single-block region}}
"transform.test_transform_unrestricted_op_no_interface"() ({
^bb0(%arg0: !pdl.operation):
  "test.potential_terminator"() : () -> ()
^bb1:
  "test.potential_terminator"() : () -> ()
}) : () -> ()

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  // expected-error @below {{result #0 has more than one potential consumer}}
  %0 = test_produce_param_or_forward_operand 42
  // expected-note @below {{used here as operand #0}}
  test_consume_operand_if_matches_param_or_fail %0[42]
  // expected-note @below {{used here as operand #0}}
  test_consume_operand_if_matches_param_or_fail %0[42]
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  // expected-error @below {{result #0 has more than one potential consumer}}
  %0 = test_produce_param_or_forward_operand 42
  // expected-note @below {{used here as operand #0}}
  test_consume_operand_if_matches_param_or_fail %0[42]
  // expected-note @below {{used here as operand #0}}
  transform.sequence %0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    test_consume_operand_if_matches_param_or_fail %arg1[42]
  }
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  // expected-error @below {{result #0 has more than one potential consumer}}
  %0 = test_produce_param_or_forward_operand 42
  // expected-note @below {{used here as operand #0}}
  test_consume_operand_if_matches_param_or_fail %0[42]
  transform.sequence %0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    // expected-note @below {{used here as operand #0}}
    test_consume_operand_if_matches_param_or_fail %0[42]
  }
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  // expected-error @below {{result #0 has more than one potential consumer}}
  %0 = test_produce_param_or_forward_operand 42
  // expected-note @below {{used here as operand #0}}
  test_consume_operand_if_matches_param_or_fail %0[42]
  // expected-note @below {{used here as operand #0}}
  transform.sequence %0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    transform.sequence %arg1 : !pdl.operation failures(propagate) {
    ^bb2(%arg2: !pdl.operation):
      test_consume_operand_if_matches_param_or_fail %arg2[42]
    }
  }
}

// -----

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  // expected-error @below {{expects at least one region}}
  transform.alternatives
}

// -----

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  // expected-error @below {{expects terminator operands to have the same type as results of the operation}}
  %2 = transform.alternatives %arg1 : !pdl.operation -> !pdl.operation {
  ^bb2(%arg2: !pdl.operation):
    transform.yield %arg2 : !pdl.operation
  }, {
  ^bb2(%arg2: !pdl.operation):
    // expected-note @below {{terminator}}
    transform.yield
  }
}

// -----

// expected-error @below {{expects the entry block to have at least one argument}}
transform.alternatives {
^bb0:
  transform.yield
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  // expected-error @below {{result #0 has more than one potential consumer}}
  %0 = test_produce_param_or_forward_operand 42
  // expected-note @below {{used here as operand #0}}
  transform.foreach %0 : !pdl.operation {
  ^bb1(%arg1: !pdl.operation):
    transform.test_consume_operand %arg1
  }
  // expected-note @below {{used here as operand #0}}
  transform.test_consume_operand %0
}

// -----

transform.sequence failures(suppress) {
^bb0(%arg0: !transform.any_op):
  // expected-error @below {{TransformOpInterface requires memory effects on operands to be specified}}
  // expected-note @below {{no effects specified for operand #0}}
  transform.test_required_memory_effects %arg0 : (!transform.any_op) -> !transform.any_op
}

// -----

transform.sequence failures(suppress) {
^bb0(%arg0: !transform.any_op):
  // expected-error @below {{TransformOpInterface requires 'allocate' memory effect to be specified for results}}
  // expected-note @below {{no 'allocate' effect specified for result #0}}
  transform.test_required_memory_effects %arg0 {has_operand_effect} : (!transform.any_op) -> !transform.any_op
}
