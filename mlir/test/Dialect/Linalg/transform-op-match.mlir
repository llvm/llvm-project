// RUN: mlir-opt %s --test-transform-dialect-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics

func.func @bar() {
  // expected-remark @below {{matched op name}}
  // expected-remark @below {{matched attr name}}
  %0 = arith.constant {my_attr} 0: i32
  // expected-remark @below {{matched op name}}
  %1 = arith.constant 1 : i32
  return
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %match_name = transform.structured.match ops{["arith.constant"]} in %arg1
    transform.test_print_remark_at_operand %match_name, "matched op name"
    transform.test_consume_operand %match_name

    %match_attr = transform.structured.match ops{["arith.constant"]} attribute{"my_attr"} in %arg1
    transform.test_print_remark_at_operand %match_attr, "matched attr name"
    transform.test_consume_operand %match_attr
  }
}
