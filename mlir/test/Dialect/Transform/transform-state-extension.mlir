// RUN: mlir-opt %s -test-transform-dialect-interpreter -verify-diagnostics -split-input-file

// expected-note @below {{associated payload op}}
module {
  transform.sequence failures(propagate) {
  ^bb0(%arg0: !pdl.operation):
    // expected-remark @below {{extension absent}}
    test_check_if_test_extension_present %arg0
    test_add_test_extension "A"
    // expected-remark @below {{extension present, A}}
    test_check_if_test_extension_present %arg0
    test_remove_test_extension
    // expected-remark @below {{extension absent}}
    test_check_if_test_extension_present %arg0
  }
}

// -----

// expected-note @below {{associated payload op}}
module {
  transform.sequence failures(propagate) {
  ^bb0(%arg0: !pdl.operation):
    test_add_test_extension "A"
    test_remove_test_extension
    test_add_test_extension "B"
    // expected-remark @below {{extension present, B}}
    test_check_if_test_extension_present %arg0
  }
}

// -----

// expected-note @below {{associated payload op}}
module {
  transform.sequence failures(propagate) {
  ^bb0(%arg0: !pdl.operation):
    test_add_test_extension "A"
    // expected-remark @below {{extension present, A}}
    test_check_if_test_extension_present %arg0
    // expected-note @below {{associated payload op}}
    test_remap_operand_to_self %arg0
    // expected-remark @below {{extension present, A}}
    test_check_if_test_extension_present %arg0
  }
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  test_add_test_extension "A"
   // This is okay because we are replacing the top-level module operation
   // (0 results) with this operation that has _more_ (1) results.
  %dummy = test_remap_operand_to_self %arg0 : !pdl.operation
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  test_add_test_extension "A"
  %dummy = test_remap_operand_to_self %arg0 : !pdl.operation
  // This is still okay. Even though we are replacing the previous
  // operation with (1 result) with this operation that has less (0) results,
  // there is no handle to the result, hence no issue with value handle update.
  test_remap_operand_to_self %dummy
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  test_add_test_extension "A"
  // expected-error @below {{cannot replace an op with another op producing fewer results while tracking handles}}
  %dummy = test_remap_operand_to_self %arg0 : !pdl.operation
  %valuehandle = transform.get_result %dummy[0] : (!pdl.operation) -> !transform.any_value
  test_remap_operand_to_self %dummy
}

// -----

module {
  transform.sequence failures(suppress) {
  ^bb0(%arg0: !pdl.operation):
    // expected-error @below {{TestTransformStateExtension missing}}
    test_remap_operand_to_self %arg0
  }
}
