// RUN: mlir-opt %s | FileCheck %s

// These types and ops are defined by a test extension but should be okay to
// roundtrip.

// CHECK: transform.test_transform_op
transform.test_transform_op

// CHECK: = transform.test_produce_self_handle_or_forward_operand {foo = "bar"}
%0 = transform.test_produce_self_handle_or_forward_operand { foo = "bar" } : () -> !transform.any_op

// CHECK: transform.test_consume_operand_of_op_kind_or_fail %{{.*}},
transform.test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op

// Ensure that the extension type is roundtripped correctly.
// CHECK: transform.cast %{{.*}} : !transform.any_op to !transform.test_dialect_op
%1 = transform.cast %0: !transform.any_op to !transform.test_dialect_op
