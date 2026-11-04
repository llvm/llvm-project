// RUN: mlir-opt %s --transform-interpreter --split-input-file --verify-diagnostics

// CHECK-LABEL: @incorrect terminator
module attributes {transform.with_named_sequence} {
  transform.named_sequence @operands_not_one_to_one_with_vars(%arg0: !transform.any_op {transform.readonly}) {
    %param_as_param = transform.param.constant 42 -> !transform.param<i64>
    // expected-error@below {{op expected 'smt.yield' as terminator}}
    transform.smt.constrain_params(%param_as_param) : (!transform.param<i64>) -> () {
      ^bb0(%param_as_smt_var: !smt.int):
      transform.yield
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

// CHECK-LABEL: @non_smt_type_block_args
module attributes {transform.with_named_sequence} {
  transform.named_sequence @non_smt_type_block_args(%arg0: !transform.any_op {transform.readonly}) {
    %param_as_param = transform.param.constant 42 -> !transform.param<i8>
    // expected-error@below {{the type of block arg #0 is expected to be either a !smt.bool, a !smt.int, or a !smt.bv}}
    transform.smt.constrain_params(%param_as_param) : (!transform.param<i8>) -> (!transform.param<i8>) {
      ^bb0(%param_as_smt_var: !transform.param<i8>):
      smt.yield %param_as_smt_var : !transform.param<i8>
    }
    transform.yield
  }
}


// -----

// CHECK-LABEL: @mismatched_arg_type_bool
module attributes {transform.with_named_sequence} {
  transform.named_sequence @mismatched_arg_type_bool(%arg0: !transform.any_op {transform.readonly}) {
    %param_as_param = transform.param.constant 42 -> !transform.param<i64>
    // expected-error@below {{the type of block arg #0 is !smt.bool though the corresponding operand type ('!transform.param<i64>') is not wrapping i1}}
    transform.smt.constrain_params(%param_as_param) : (!transform.param<i64>) -> (!transform.param<i64>) {
      ^bb0(%param_as_smt_var: !smt.bool):
      smt.yield %param_as_smt_var : !smt.bool
    }
    transform.yield
  }
}

// -----

// CHECK-LABEL: @mismatched_arg_type_bitvector
module attributes {transform.with_named_sequence} {
  transform.named_sequence @mismatched_arg_type_bitvector(%arg0: !transform.any_op {transform.readonly}) {
    %param_as_param = transform.param.constant 42 -> !transform.param<i64>
    // expected-error@below {{the type of block arg #0 is '!smt.bv<8>' though the corresponding operand type ('!transform.param<i64>') is not wrapping an integer type of the same bitwidth}}
    transform.smt.constrain_params(%param_as_param) : (!transform.param<i64>) -> (!transform.param<i64>) {
      ^bb0(%param_as_smt_var: !smt.bv<8>):
      smt.yield %param_as_smt_var : !smt.bv<8>
    }
    transform.yield
  }
}

// -----

// CHECK-LABEL: @mismatched_result_type_bool
module attributes {transform.with_named_sequence} {
  transform.named_sequence @mismatched_result_type_bool(%arg0: !transform.any_op {transform.readonly}) {
    %param_as_param = transform.param.constant 1 -> !transform.param<i1>
    transform.smt.constrain_params(%param_as_param) : (!transform.param<i1>) -> (!transform.param<i64>) {
      ^bb0(%param_as_smt_var: !smt.bool):
      // expected-error@below {{the type of terminator operand #0 is !smt.bool though the corresponding result type ('!transform.param<i64>') is not wrapping i1}}
      smt.yield %param_as_smt_var : !smt.bool
    }
    transform.yield
  }
}

// -----

// CHECK-LABEL: @mismatched_result_type_bitvector
module attributes {transform.with_named_sequence} {
  transform.named_sequence @mismatched_result_type_bitvector(%arg0: !transform.any_op {transform.readonly}) {
    %param_as_param = transform.param.constant 42 -> !transform.param<i8>
    transform.smt.constrain_params(%param_as_param) : (!transform.param<i8>) -> (!transform.param<i64>) {
      ^bb0(%param_as_smt_var: !smt.bv<8>):
      // expected-error@below {{the type of terminator operand #0 is '!smt.bv<8>' though the corresponding result type ('!transform.param<i64>') is not wrapping an integer type of the same bitwidth}}
      smt.yield %param_as_smt_var : !smt.bv<8>
    }
    transform.yield
  }
}
