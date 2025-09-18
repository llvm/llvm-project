// RUN: mlir-opt %s --transform-interpreter --split-input-file --verify-diagnostics

// CHECK-LABEL: @constraint_not_using_smt_ops
module attributes {transform.with_named_sequence} {
  transform.named_sequence @constraint_not_using_smt_ops(%arg0: !transform.any_op {transform.readonly}) {
    %param_as_param = transform.param.constant 42 -> !transform.param<i64>
    // expected-error@below {{ops contained in region should belong to SMT-dialect}}
    transform.smt.constrain_params(%param_as_param) : !transform.param<i64> {
      ^bb0(%param_as_smt_var: !smt.int):
      %c4 = arith.constant 4 : i32
      // This is the kind of thing one might think works:
      //arith.remsi %param_as_smt_var, %c4 : i32
    }
    transform.yield
  }
}

// -----

// CHECK-LABEL: @operands_not_one_to_one_with_vars
module attributes {transform.with_named_sequence} {
  transform.named_sequence @operands_not_one_to_one_with_vars(%arg0: !transform.any_op {transform.readonly}) {
    %param_as_param = transform.param.constant 42 -> !transform.param<i64>
    // expected-error@below {{must have the same number of block arguments as operands}}
    transform.smt.constrain_params(%param_as_param) : !transform.param<i64> {
      ^bb0(%param_as_smt_var: !smt.int, %param_as_another_smt_var: !smt.int):
    }
    transform.yield
  }
}
