// RUN: mlir-opt --test-transform-dialect-interpreter='enable-expensive-checks=1' --split-input-file --verify-diagnostics %s

// expected-note @below {{ancestor op}}
func.func @func() {
  // expected-note @below {{nested op}}
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

  sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    // expected-note @below {{other handle}}
    %0 = pdl_match @return in %arg1
    %1 = get_closest_isolated_parent %0
    // expected-error @below {{invalidated the handle to payload operations nested in the payload operation associated with its operand #0}}
    test_consume_operand %1
    test_print_remark_at_operand %0, "remark"
  }
}
