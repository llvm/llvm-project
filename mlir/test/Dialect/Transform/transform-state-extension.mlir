// RUN: mlir-opt %s -transform-interpreter -verify-diagnostics -split-input-file

// expected-note @below {{associated payload op}}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-remark @below {{extension absent}}
    transform.test_check_if_test_extension_present %arg0 : !transform.any_op
    transform.test_add_test_extension "A"
    // expected-remark @below {{extension present, A}}
    transform.test_check_if_test_extension_present %arg0 : !transform.any_op
    transform.test_remove_test_extension
    // expected-remark @below {{extension absent}}
    transform.test_check_if_test_extension_present %arg0 : !transform.any_op
    transform.yield
  }
}

// -----

// expected-note @below {{associated payload op}}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.test_add_test_extension "A"
    transform.test_remove_test_extension
    transform.test_add_test_extension "B"
    // expected-remark @below {{extension present, B}}
    transform.test_check_if_test_extension_present %arg0 : !transform.any_op
    transform.yield
  }
}

// -----

// expected-note @below {{associated payload op}}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.test_add_test_extension "A"
    // expected-remark @below {{extension present, A}}
    transform.test_check_if_test_extension_present %arg0 : !transform.any_op
    // expected-note @below {{associated payload op}}
    transform.test_remap_operand_to_self %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-remark @below {{extension present, A}}
    transform.test_check_if_test_extension_present %arg0 : !transform.any_op
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.test_add_test_extension "A"
     // This is okay because we are replacing the top-level module operation
     // (0 results) with this operation that has _more_ (1) results.
    %dummy = transform.test_remap_operand_to_self %arg0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.test_add_test_extension "A"
    %dummy = transform.test_remap_operand_to_self %arg0 : (!transform.any_op) -> !transform.any_op
    // This is still okay. Even though we are replacing the previous
    // operation with (1 result) with this operation that has less (0) results,
    // there is no handle to the result, hence no issue with value handle update.
    transform.test_remap_operand_to_self %dummy : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.test_add_test_extension "A"
    // expected-error @below {{cannot replace an op with another op producing fewer results while tracking handles}}
    %dummy = transform.test_remap_operand_to_self %arg0 : (!transform.any_op) -> !transform.any_op
    %valuehandle = transform.get_result %dummy[0] : (!transform.any_op) -> !transform.any_value
    transform.test_remap_operand_to_self %dummy : (!transform.any_op) -> ()
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-error @below {{TestTransformStateExtension missing}}
    transform.test_remap_operand_to_self %arg0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
