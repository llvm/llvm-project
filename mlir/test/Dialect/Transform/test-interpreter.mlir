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
