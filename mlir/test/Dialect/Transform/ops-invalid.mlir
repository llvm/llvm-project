// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// expected-error @below {{expects the entry block to have one argument of type '!pdl.operation'}}
transform.sequence {
}

// -----

// expected-note @below {{nested in another possible top-level op}}
transform.sequence {
^bb0(%arg0: !pdl.operation):
  // expected-error @below {{expects the root operation to be provided for a nested op}}
  transform.sequence {
  ^bb1(%arg1: !pdl.operation):
  }
}

// -----

// expected-error @below {{expected children ops to implement TransformOpInterface}}
transform.sequence {
^bb0(%arg0: !pdl.operation):
  // expected-note @below {{op without interface}}
  arith.constant 42.0 : f32
}

// -----

// expected-error @below {{expects the types of the terminator operands to match the types of the resul}}
%0 = transform.sequence {
^bb0(%arg0: !pdl.operation):
  // expected-note @below {{terminator}}
  transform.yield
} : !pdl.operation

// -----

// expected-note @below {{nested in another possible top-level op}}
transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  // expected-error @below {{expects the root operation to be provided for a nested op}}
  transform.sequence {
  ^bb1(%arg1: !pdl.operation):
  }
}

// -----

// expected-error @below {{expects only one non-pattern op in its body}}
transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  // expected-note @below {{first non-pattern op}}
  transform.sequence {
  ^bb1(%arg1: !pdl.operation):
  }
  // expected-note @below {{second non-pattern op}}
  transform.sequence {
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
  transform.with_pdl_patterns %arg0 {
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

transform.sequence {
^bb0(%arg0: !pdl.operation):
  // expected-error @below {{result #0 has more than one potential consumer}}
  %0 = test_produce_param_or_forward_operand 42
  // expected-note @below {{used here as operand #0}}
  test_consume_operand_if_matches_param_or_fail %0[42]
  // expected-note @below {{used here as operand #0}}
  test_consume_operand_if_matches_param_or_fail %0[42]
}

// -----

transform.sequence {
^bb0(%arg0: !pdl.operation):
  // expected-error @below {{result #0 has more than one potential consumer}}
  %0 = test_produce_param_or_forward_operand 42
  // expected-note @below {{used here as operand #0}}
  test_consume_operand_if_matches_param_or_fail %0[42]
  // expected-note @below {{used here as operand #0}}
  transform.sequence %0 {
  ^bb1(%arg1: !pdl.operation):
    test_consume_operand_if_matches_param_or_fail %arg1[42]
  }
}

// -----

transform.sequence {
^bb0(%arg0: !pdl.operation):
  // expected-error @below {{result #0 has more than one potential consumer}}
  %0 = test_produce_param_or_forward_operand 42
  // expected-note @below {{used here as operand #0}}
  test_consume_operand_if_matches_param_or_fail %0[42]
  transform.sequence %0 {
  ^bb1(%arg1: !pdl.operation):
    // expected-note @below {{used here as operand #0}}
    test_consume_operand_if_matches_param_or_fail %0[42]
  }
}

// -----

transform.sequence {
^bb0(%arg0: !pdl.operation):
  // expected-error @below {{result #0 has more than one potential consumer}}
  %0 = test_produce_param_or_forward_operand 42
  // expected-note @below {{used here as operand #0}}
  test_consume_operand_if_matches_param_or_fail %0[42]
  // expected-note @below {{used here as operand #0}}
  transform.sequence %0 {
  ^bb1(%arg1: !pdl.operation):
    transform.sequence %arg1 {
    ^bb2(%arg2: !pdl.operation):
      test_consume_operand_if_matches_param_or_fail %arg2[42]
    }
  }
}

// -----

transform.sequence {
^bb1(%arg1: !pdl.operation):
  // expected-error @below {{expects at least one region}}
  transform.alternatives
}

// -----

transform.sequence {
^bb1(%arg1: !pdl.operation):
  // expected-error @below {{expects terminator operands to have the same type as results of the operation}}
  %2 = transform.alternatives %arg1 -> !pdl.operation {
  ^bb2(%arg2: !pdl.operation):
    transform.yield %arg2 : !pdl.operation
  }, {
  ^bb2(%arg2: !pdl.operation):
    // expected-note @below {{terminator}}
    transform.yield
  }
}

// -----

// expected-error @below {{expects the entry block to have one argument of type '!pdl.operation'}}
transform.alternatives {
^bb0:
  transform.yield
}
