// RUN: mlir-opt %s --transform-interpreter --split-input-file --verify-diagnostics

// Silenceable diagnostics suppressed.
module attributes { transform.with_named_sequence } {
  func.func @test_loop_peeling_not_beneficial() {
    %lb = arith.constant 0 : index
    %ub = arith.constant 40 : index
    %step = arith.constant 5 : index
    scf.for %i = %lb to %ub step %step {
      arith.addi %i, %i : index
    }
    return
  }

  transform.named_sequence @peel(%arg0: !transform.op<"scf.for"> {transform.consumed}) {
    transform.loop.peel %arg0 : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
  transform.named_sequence @match_for(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %arg0 ["scf.for"] : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.sequence  %root : !transform.any_op failures(suppress) {
    ^bb0(%arg0: !transform.any_op):
      transform.foreach_match in %arg0
          @match_for -> @peel
          : (!transform.any_op) -> !transform.any_op
      transform.yield
    }
    transform.yield
  }
}

// -----

// Silenceable diagnostics propagated.
module attributes { transform.with_named_sequence } {
  func.func @test_loop_peeling_not_beneficial() {
    %lb = arith.constant 0 : index
    %ub = arith.constant 40 : index
    %step = arith.constant 5 : index
    // expected-note @below {{when applied to this matching payload}}
    scf.for %i = %lb to %ub step %step {
      arith.addi %i, %i : index
    }
    return
  }

  // expected-note @below {{failed to peel the last iteration}}
  transform.named_sequence @peel(%arg0: !transform.op<"scf.for"> {transform.consumed}) {
    transform.loop.peel %arg0 : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
  transform.named_sequence @match_for(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %arg0 ["scf.for"] : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }
  transform.named_sequence @main_suppress(%root: !transform.any_op) {
    transform.sequence  %root : !transform.any_op failures(suppress) {
    ^bb0(%arg0: !transform.any_op):
      transform.foreach_match in %arg0
          @match_for -> @peel
          : (!transform.any_op) -> !transform.any_op
      transform.yield
    }
    transform.yield
  }
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.sequence  %root : !transform.any_op failures(propagate) {
    ^bb0(%arg0: !transform.any_op):
      // expected-error @below {{actions failed}}
      transform.foreach_match in %arg0
          @match_for -> @peel
          : (!transform.any_op) -> !transform.any_op
      transform.yield
    }
    transform.yield
  }
}

// -----

// expected-remark @below {{op from within the matcher}}
module attributes { transform.with_named_sequence } {
  // expected-remark @below {{returned root}}
  func.func @foo() {
    return
  }

  transform.named_sequence @match_fail(
      %op: !transform.any_op {transform.readonly},
      %root: !transform.any_op {transform.readonly},
      %param: !transform.param<i64> {transform.readonly}) -> (!transform.any_op, !transform.param<i64>) {
    transform.test_succeed_if_operand_of_op_kind %op, "test.impossible_to_match" : !transform.any_op
    transform.yield %root, %param : !transform.any_op, !transform.param<i64>
  }

  transform.named_sequence @match_succeed(
      %op: !transform.any_op {transform.readonly},
      %root: !transform.any_op {transform.readonly},
      %param: !transform.param<i64> {transform.readonly}) -> (!transform.any_op, !transform.param<i64>) {
    transform.debug.emit_remark_at %root, "op from within the matcher" : !transform.any_op
    // expected-remark @below {{param from within the matcher 42}}
    transform.debug.emit_param_as_remark %param, "param from within the matcher" : !transform.param<i64>
    transform.yield %root, %param : !transform.any_op, !transform.param<i64>
  }

  transform.named_sequence @return(
      %root: !transform.any_op {transform.readonly},
      %param: !transform.param<i64> {transform.readonly}) -> (!transform.param<i64>, !transform.param<i64>, !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.yield %param, %param, %func : !transform.param<i64>, !transform.param<i64>, !transform.any_op
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %param = transform.param.constant 42 : i64 -> !transform.param<i64>
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    %func2, %yielded:3 = transform.foreach_match restrict_root in %func, %root, %param
      @match_fail -> @return,
      @match_succeed -> @return
      : (!transform.any_op, !transform.any_op, !transform.param<i64>) -> (!transform.any_op, !transform.param<i64>, !transform.param<i64>, !transform.any_op)
    transform.debug.emit_remark_at %yielded#2, "returned root" : !transform.any_op
    // expected-remark @below {{42 : i64, 42 : i64}}
    transform.debug.emit_param_as_remark %yielded#0: !transform.param<i64>
    %num_roots = transform.num_associations %yielded#2 : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below {{2 : i64}}
    transform.debug.emit_param_as_remark %num_roots : !transform.param<i64>
    transform.yield
  }
}

// -----

module attributes { transform.with_named_sequence } {
  func.func private @foo()
  func.func private @bar()

  transform.named_sequence @match(
      %op: !transform.any_op {transform.readonly},
      %func: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.yield %func : !transform.any_op
  }

  transform.named_sequence @return(
      %func: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.yield %func : !transform.any_op
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    %func2, %yielded = transform.foreach_match flatten_results restrict_root in %func, %func
      @match -> @return
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %num = transform.num_associations %yielded : (!transform.any_op) -> !transform.param<i64>
    // 2 funcs are yielded for each of the 2 funcs = 4:
    // expected-remark @below {{4 : i64}}
    transform.debug.emit_param_as_remark %num : !transform.param<i64>
    transform.yield
  }
}

// -----


module attributes { transform.with_named_sequence } {
  func.func private @foo()
  func.func private @bar()

  transform.named_sequence @match(
      %op: !transform.any_op {transform.readonly},
      %func: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.yield %func : !transform.any_op
  }

  transform.named_sequence @return(
      %func: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.yield %func : !transform.any_op
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{action @return has results associated with multiple payload entities, but flattening was not requested}}
    %func2, %yielded = transform.foreach_match restrict_root in %func, %func
      @match -> @return
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %num = transform.num_associations %yielded : (!transform.any_op) -> !transform.param<i64>
    transform.yield
  }
}
