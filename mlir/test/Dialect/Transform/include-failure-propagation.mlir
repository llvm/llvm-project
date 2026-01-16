// RUN: mlir-opt %s --transform-interpreter -allow-unregistered-dialect --verify-diagnostics

module attributes { transform.with_named_sequence } {
  // Callee returns a silenceable failure when given a module instead of func.func.
  transform.named_sequence @callee(%root: !transform.any_op {transform.consumed}) -> (!transform.any_op) {
    transform.test_consume_operand_of_op_kind_or_fail %root, "func.func" : !transform.any_op
    transform.yield %root : !transform.any_op
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %res = transform.sequence %root : !transform.any_op -> !transform.any_op failures(suppress) {
    ^bb0(%arg0: !transform.any_op):
      // This include returns a silenceable failure; it must not remap results.
      %included = transform.include @callee failures(propagate) (%arg0) : (!transform.any_op) -> (!transform.any_op)
      transform.yield %included : !transform.any_op
    }

    %count = transform.num_associations %res : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below {{0}}
    transform.debug.emit_param_as_remark %count : !transform.param<i64>

    // If the include incorrectly forwarded mappings on failure, this would run
    // and produce an unexpected remark under --verify-diagnostics.
    transform.foreach %res : !transform.any_op {
    ^bb0(%it: !transform.any_op):
      transform.debug.emit_remark_at %it, "include result unexpectedly populated" : !transform.any_op
    }
    transform.yield
  }
}

// -----

module {
  func.func @payload() {
    return
  }
}
