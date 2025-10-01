// RUN: mlir-opt %s --transform-interpreter --split-input-file --verify-diagnostics

// CHECK-LABEL: @constraint_not_using_smt_ops
module attributes {transform.with_named_sequence} {
  transform.named_sequence @constraint_not_using_smt_ops(%arg0: !transform.any_op {transform.readonly}) {
    %param_as_param = transform.param.constant 42 -> !transform.param<i64>
    // expected-error@below {{ops contained in region should belong to SMT-dialect}}
    transform.smt.constrain_params(%param_as_param) : (!transform.param<i64>) -> () {
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
    transform.smt.constrain_params(%param_as_param) : (!transform.param<i64>) -> () {
      ^bb0(%param_as_smt_var: !smt.int, %param_as_another_smt_var: !smt.int):
    }
    transform.yield
  }
}

// -----

// CHECK-LABEL: @results_not_one_to_one_with_vars
module attributes {transform.with_named_sequence} {
  transform.named_sequence @results_not_one_to_one_with_vars(%arg0: !transform.any_op {transform.readonly}) {
    %param_as_param = transform.param.constant 42 -> !transform.param<i64>
    transform.smt.constrain_params(%param_as_param, %param_as_param) : (!transform.param<i64>, !transform.param<i64>) -> () {
      ^bb0(%param_as_smt_var: !smt.int, %param_as_another_smt_var: !smt.int):
      // expected-error@below {{expected terminator to have as many operands as the parent op has results}}
      smt.yield %param_as_smt_var : !smt.int
    }
    transform.yield
  }
}

// -----

// CHECK-LABEL: @mismatched_type_bool
module attributes {transform.with_named_sequence} {
  transform.named_sequence @mismatched_type_bool(%arg0: !transform.any_op {transform.readonly}) {
    %param_as_param = transform.param.constant 42 -> !transform.param<i64>
    // expected-error@below {{the type of block arg #0 is !smt.bool though the corresponding operand type ('!transform.param<i64>') is not wrapping i1 (i.e. bool)}}
    transform.smt.constrain_params(%param_as_param) : (!transform.param<i64>) -> (!transform.param<i64>) {
      ^bb0(%param_as_smt_var: !smt.bool):
      smt.yield %param_as_smt_var : !smt.bool
    }
    transform.yield
  }
}

// -----

// CHECK-LABEL: @mismatched_type_bitvector
module attributes {transform.with_named_sequence} {
  transform.named_sequence @mismatched_type_bitvector(%arg0: !transform.any_op {transform.readonly}) {
    %param_as_param = transform.param.constant 42 -> !transform.param<i64>
    // expected-error@below {{the type of block arg #0 is '!smt.bv<8>' though the corresponding operand type ('!transform.param<i64>') is not wrapping an integer type of the same bitwidth}}
    transform.smt.constrain_params(%param_as_param) : (!transform.param<i64>) -> (!transform.param<i64>) {
      ^bb0(%param_as_smt_var: !smt.bv<8>):
      smt.yield %param_as_smt_var : !smt.bv<8>
    }
    transform.yield
  }
}
