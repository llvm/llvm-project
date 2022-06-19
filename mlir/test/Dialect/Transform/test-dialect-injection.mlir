// RUN: mlir-opt %s | FileCheck %s

// These ops are defined by a test extension but should be okay to roundtrip.

// CHECK: transform.test_transform_op
transform.test_transform_op

// CHECK: = transform.test_produce_param_or_forward_operand 42 {foo = "bar"}
%0 = transform.test_produce_param_or_forward_operand 42 { foo = "bar" }

// CHECK: transform.test_consume_operand_if_matches_param_or_fail %{{.*}}[42]
transform.test_consume_operand_if_matches_param_or_fail %0[42]
