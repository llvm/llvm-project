// RUN: mlir-opt %s --test-transform-dialect-interpreter --split-input-file --verify-diagnostics

// expected-remark @below {{applying transformation}}
transform.test_transform_op

// -----

%0 = transform.test_produce_param_or_forward_operand 42 { foo = "bar" }
// expected-remark @below {{succeeded}}
transform.test_consume_operand_if_matches_param_or_fail %0[42]

// -----

%0 = transform.test_produce_param_or_forward_operand 42 { foo = "bar" }
// expected-error @below {{expected the operand to be associated with 21 got 42}}
transform.test_consume_operand_if_matches_param_or_fail %0[21]

// -----

// expected-error @below {{operation tracked by two handles}}
%0 = transform.test_produce_param_or_forward_operand 42
// expected-note @below {{handle}}
%1 = transform.test_produce_param_or_forward_operand from %0
// expected-note @below {{handle}}
%2 = transform.test_produce_param_or_forward_operand from %0
transform.test_consume_operand_if_matches_param_or_fail %1[42]
transform.test_consume_operand_if_matches_param_or_fail %2[42]

// -----

transform.sequence {
^bb0(%arg0: !pdl.operation):
  sequence %arg0 {
  ^bb0(%arg1: !pdl.operation):
    // expected-remark @below {{applying transformation "a"}}
    test_transform_op "a"
    // expected-remark @below {{applying transformation "b"}}
    test_transform_op "b"
    // expected-remark @below {{applying transformation "c"}}
    test_transform_op "c"
  }
  // expected-remark @below {{applying transformation "d"}}
  test_transform_op "d"
  // expected-remark @below {{applying transformation "e"}}
  test_transform_op "e"
}

// -----

transform.sequence {
^bb0(%arg0: !pdl.operation):
  %0 = test_produce_param_or_forward_operand 42
  sequence %0 {
  ^bb0(%arg1: !pdl.operation):
    // expected-remark @below {{succeeded}}
    test_consume_operand_if_matches_param_or_fail %arg1[42]
  }
}

// -----

transform.sequence {
^bb0(%arg0: !pdl.operation):
  %0 = sequence %arg0 {
  ^bb0(%arg1: !pdl.operation):
    %1 = test_produce_param_or_forward_operand 42
    yield %1 : !pdl.operation
  } : !pdl.operation
  // expected-remark @below {{succeeded}}
  test_consume_operand_if_matches_param_or_fail %0[42]
}
