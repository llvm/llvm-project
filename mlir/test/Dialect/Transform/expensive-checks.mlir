// RUN: mlir-opt --test-transform-dialect-interpreter='enable-expensive-checks=1' --split-input-file --verify-diagnostics %s

// expected-note @below {{ancestor payload op}}
func.func @func() {
  // expected-note @below {{nested payload op}}
  return
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @return : benefit(1) {
    %0 = operands
    %1 = types
    %2 = operation "func.return"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
    rewrite %2 with "transform.dialect"
  }

  sequence %arg0 failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    // expected-note @below {{handle to invalidated ops}}
    %0 = pdl_match @return in %arg1
    %1 = get_closest_isolated_parent %0
    // expected-note @below {{invalidated by this transform op that consumes its operand #0}}
    test_consume_operand %1
    // expected-error @below {{op uses a handle invalidated by a previously executed transform op}}
    test_print_remark_at_operand %0, "remark"
  }
}

// -----

func.func @func1() {
  // expected-note @below {{repeated target op}}
  return
}
func.func private @func2()

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @func : benefit(1) {
    %0 = operands
    %1 = types
    %2 = operation "func.func"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
    rewrite %2 with "transform.dialect"
  }
  pdl.pattern @return : benefit(1) {
    %0 = operands
    %1 = types
    %2 = operation "func.return"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
    rewrite %2 with "transform.dialect"
  }

  sequence %arg0 failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %0 = pdl_match @func in %arg1
    %1 = pdl_match @return in %arg1
    %2 = replicate num(%0) %1
    // expected-error @below {{a handle passed as operand #0 and consumed by this operation points to a payload operation more than once}}
    test_consume_operand %2
    test_print_remark_at_operand %0, "remark"
  }
}


// -----

// expected-note @below {{ancestor payload op}}
// expected-note @below {{nested payload op}}
module {

  transform.sequence failures(propagate) {
  ^bb0(%0: !pdl.operation):
    %1 = transform.test_copy_payload %0
    // expected-note @below {{handle to invalidated ops}}
    %2 = transform.test_copy_payload %0
    // expected-note @below {{invalidated by this transform op that consumes its operand #0}}
    transform.test_consume_operand %1
    // expected-error @below {{op uses a handle invalidated by a previously executed transform op}}
    transform.test_consume_operand %2
  }
}

// -----

// expected-note @below {{ancestor payload op}}
// expected-note @below {{nested payload op}}
module {

  transform.sequence failures(propagate) {
  ^bb0(%0: !pdl.operation):
    %1 = transform.test_copy_payload %0
    // expected-note @below {{handle to invalidated ops}}
    %2 = transform.test_copy_payload %0
    // Consuming two handles in the same operation is invalid if they point
    // to overlapping sets of payload IR ops.
    //
    // expected-error @below {{op uses a handle invalidated by a previously executed transform op}}
    // expected-note @below {{invalidated by this transform op that consumes its operand #0 and invalidates handles}}
    transform.test_consume_operand %1, %2
  }
}
