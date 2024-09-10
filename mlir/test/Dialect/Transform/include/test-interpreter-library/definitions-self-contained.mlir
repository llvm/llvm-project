// RUN: mlir-opt %s
// No need to check anything else than parsing here, this is being used by another test as data.

module attributes {transform.with_named_sequence} {
  transform.named_sequence private @private_helper(%arg0: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %arg0, "message" : !transform.any_op
    transform.yield
  }

  // These ops collide with ops from the other module before or after renaming.
  transform.named_sequence private @colliding(%arg0: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %arg0, "external colliding (without suffix)" : !transform.any_op
    transform.yield
  }
  transform.named_sequence private @colliding_0(%arg0: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %arg0, "external colliding_0" : !transform.any_op
    transform.yield
  }
  transform.named_sequence private @colliding_2(%arg0: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %arg0, "external colliding_2" : !transform.any_op
    transform.yield
  }
  transform.named_sequence private @colliding_3(%arg0: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %arg0, "external colliding_3" : !transform.any_op
    transform.yield
  }
  transform.named_sequence private @colliding_4(%arg0: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %arg0, "external colliding_4" : !transform.any_op
    transform.yield
  }
  transform.named_sequence @colliding_5(%arg0: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %arg0, "external colliding_5" : !transform.any_op
    transform.yield
  }

  transform.named_sequence @print_message(%arg0: !transform.any_op {transform.readonly}) {
    transform.include @private_helper failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.yield
  }

  transform.named_sequence @consuming(%arg0: !transform.any_op {transform.consumed}) {
    transform.test_consume_operand %arg0 : !transform.any_op
    transform.yield
  }

  transform.named_sequence @unannotated(%arg0: !transform.any_op) {
    transform.debug.emit_remark_at %arg0, "unannotated" : !transform.any_op
    transform.yield
  }

  transform.named_sequence @symbol_user(%arg0: !transform.any_op {transform.readonly}) {
    transform.include @colliding failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.include @colliding_0 failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.include @colliding_2 failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.include @colliding_3 failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.include @colliding_4 failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.include @colliding_5 failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.yield
  }
}
