// RUN: mlir-opt %s --test-transform-dialect-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics

transform.with_pdl_patterns {
^bb0(%arg0: !transform.any_op):
  sequence %arg0 : !transform.any_op failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    %0 = pdl_match @some in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %0, "matched" : !transform.any_op
  }

  pdl.pattern @some : benefit(1) {
    %0 = pdl.operation "test.some_op"
    pdl.rewrite %0 with "transform.dialect"
  }

  pdl.pattern @other : benefit(1) {
    %0 = pdl.operation "test.other_op"
    pdl.rewrite %0 with "transform.dialect"
  }
}

// expected-remark @below {{matched}}
"test.some_op"() : () -> ()
"test.other_op"() : () -> ()
// expected-remark @below {{matched}}
"test.some_op"() : () -> ()


// -----

transform.with_pdl_patterns {
^bb0(%arg0: !transform.any_op):
  sequence %arg0 : !transform.any_op failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    %0 = pdl_match @some in %arg1 : (!transform.any_op) -> !transform.any_op
  }

  pdl.pattern @some : benefit(1) {
    %0 = pdl.operation "test.some_op"
    pdl.apply_native_constraint "verbose_constraint"(%0 : !pdl.operation)
    pdl.rewrite %0 with "transform.dialect"
  }
}

// expected-warning @below {{from PDL constraint}}
"test.some_op"() : () -> ()
"test.other_op"() : () -> ()
