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
