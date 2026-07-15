// RUN: mlir-opt %s --split-input-file | FileCheck %s

// CHECK-LABEL: @schedule_with_constrained_param
module attributes {transform.with_named_sequence} {
  transform.named_sequence @schedule_with_constrained_param(%arg0: !transform.any_op {transform.readonly}) {
    // CHECK: %[[PARAM_AS_PARAM:.*]] = transform.param.constant
    %param_as_param = transform.param.constant 42 -> !transform.param<i64>

    // CHECK: transform.smt.constrain_params(%[[PARAM_AS_PARAM]])
    transform.smt.constrain_params(%param_as_param) : (!transform.param<i64>) -> () {
      // CHECK: ^bb{{.*}}(%[[PARAM_AS_SMT_SYMB:.*]]: !smt.int):
      ^bb0(%param_as_smt_var: !smt.int):
      // CHECK: %[[C0:.*]] = smt.int.constant 0
      %c0 = smt.int.constant 0
      // CHECK: %[[C43:.*]] = smt.int.constant 43
      %c43 = smt.int.constant 43
      // CHECK: %[[LOWER_BOUND:.*]] = smt.int.cmp le %[[C0]], %[[PARAM_AS_SMT_SYMB]]
      %lower_bound = smt.int.cmp le %c0, %param_as_smt_var
      // CHECK: smt.assert %[[LOWER_BOUND]]
      smt.assert %lower_bound
      // CHECK: %[[UPPER_BOUND:.*]] = smt.int.cmp le %[[PARAM_AS_SMT_SYMB]], %[[C43]]
      %upper_bound = smt.int.cmp le %param_as_smt_var, %c43
      // CHECK: smt.assert %[[UPPER_BOUND]]
      smt.assert %upper_bound
    }
    // NB: from here can rely on that 0 <= %param_as_param <= 43, even if its
    //     definition changes.
    transform.yield
  }
}

// -----

// CHECK-LABEL: @schedule_with_constraint_on_multiple_params_returning_computed_value
module attributes {transform.with_named_sequence} {
  transform.named_sequence @schedule_with_constraint_on_multiple_params_returning_computed_value(%arg0: !transform.any_op {transform.readonly}) {
    // CHECK: %[[PARAM_A:.*]] = transform.param.constant
    %param_a = transform.param.constant 4 -> !transform.param<i64>
    // CHECK: %[[PARAM_B:.*]] = transform.param.constant
    %param_b = transform.param.constant 32 -> !transform.param<i64>

    // CHECK: transform.smt.constrain_params(%[[PARAM_A]], %[[PARAM_B]])
    %divisor = transform.smt.constrain_params(%param_a, %param_b) : (!transform.param<i64>, !transform.param<i64>) -> (!transform.param<i64>) {
      // CHECK: ^bb{{.*}}(%[[VAR_A:.*]]: !smt.int, %[[VAR_B:.*]]: !smt.int):
      ^bb0(%var_a: !smt.int, %var_b: !smt.int):
      // CHECK: %[[DIV:.*]] = smt.int.div %[[VAR_B]], %[[VAR_A]]
      %divisor = smt.int.div %var_b, %var_a
      // CHECK: %[[C0:.*]] = smt.int.constant 0
      %c0 = smt.int.constant 0
      // CHECK: %[[REMAINDER:.*]] = smt.int.mod %[[VAR_B]], %[[VAR_A]]
      %remainder = smt.int.mod %var_b, %var_a
      // CHECK: %[[EQ:.*]] = smt.eq %[[REMAINDER]], %[[C0]]
      %eq = smt.eq %remainder, %c0 : !smt.int
      // CHECK: smt.assert %[[EQ]]
      smt.assert %eq
      // CHECK: smt.yield %[[DIV]]
      smt.yield %divisor : !smt.int
    }
    // NB: from here can rely on that %param_a is a divisor of %param_b and
    //     that the relevant factor, 8, got associated to %divisor.
    transform.yield
  }
}

// -----

// CHECK-LABEL: @schedule_with_param_as_a_bool
module attributes {transform.with_named_sequence} {
  transform.named_sequence @schedule_with_param_as_a_bool(%arg0: !transform.any_op {transform.readonly}) {
    // CHECK: %[[PARAM_AS_PARAM:.*]] = transform.param.constant
    %param_as_param = transform.param.constant true -> !transform.param<i1>

    // CHECK: transform.smt.constrain_params(%[[PARAM_AS_PARAM]])
    transform.smt.constrain_params(%param_as_param) : (!transform.param<i1>) -> () {
      // CHECK: ^bb{{.*}}(%[[PARAM_AS_SMT_VAR:.*]]: !smt.bool):
      ^bb0(%param_as_smt_var: !smt.bool):
      // CHECK: %[[C0:.*]] = smt.int.constant 0
      %c0 = smt.int.constant 0
      // CHECK: %[[C1:.*]] = smt.int.constant 1
      %c1 = smt.int.constant 1
      // CHECK: %[[FALSEHOOD:.*]] = smt.eq %[[C0]], %[[C1]]
      %falsehood = smt.eq %c0, %c1 : !smt.int
      // CHECK: %[[TRUE_IFF_PARAM_IS:.*]] = smt.or %[[PARAM_AS_SMT_VAR]], %[[FALSEHOOD]]
      %true_iff_param_is = smt.or %param_as_smt_var, %falsehood
      // CHECK: smt.assert %[[TRUE_IFF_PARAM_IS]]
      smt.assert %true_iff_param_is
    }
    // NB: from here can rely on that %param_as_param holds true, even if its
    //     definition changes.
    transform.yield
  }
}
