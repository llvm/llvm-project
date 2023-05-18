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
