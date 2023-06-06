// RUN: mlir-opt %s --test-transform-dialect-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @update_tracked_op_mapping()
//       CHECK:   "test.container"() ({
//       CHECK:     %0 = "test.foo"() {annotated} : () -> i32
//       CHECK:   }) : () -> ()
func.func @update_tracked_op_mapping() {
  "test.container"() ({
    %0 = "test.foo"() {replace_with_new_op = "test.foo"} : () -> (i32)
  }) : () -> ()
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["test.container"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %1 = transform.structured.match ops{["test.foo"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns ["transform.test"] to %0 {} : !transform.any_op
  // Add an attribute to %1, which is now mapped to a new op.
  transform.annotate %1 "annotated" : !transform.any_op
}

// -----

// CHECK-LABEL: func @update_tracked_op_mapping_region()
//       CHECK:   "test.container"() ({
//       CHECK:     %0 = "test.foo"() {annotated} : () -> i32
//       CHECK:   }) : () -> ()
func.func @update_tracked_op_mapping_region() {
  "test.container"() ({
    %0 = "test.foo"() {replace_with_new_op = "test.foo"} : () -> (i32)
  }) : () -> ()
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["test.container"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %1 = transform.structured.match ops{["test.foo"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns [] to %0 {
    transform.apply_patterns.transform.test_patterns
  } : !transform.any_op
  // Add an attribute to %1, which is now mapped to a new op.
  transform.annotate %1 "annotated" : !transform.any_op
}

// -----

func.func @replacement_op_not_found() {
  "test.container"() ({
    // expected-note @below {{[0] replaced op}}
    // expected-note @below {{[0] replacement value 0}}
    %0 = "test.foo"() {replace_with_new_op = "test.bar"} : () -> (i32)
  }) : () -> ()
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["test.container"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %1 = transform.structured.match ops{["test.foo"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  // expected-error @below {{tracking listener failed to find replacement op}}
  transform.apply_patterns ["transform.test"] to %0 {} : !transform.any_op
  // %1 must be used in some way. If no replacement payload op could be found,
  // an error is thrown only if the handle is not dead.
  transform.annotate %1 "annotated" : !transform.any_op
}

// -----

// CHECK-LABEL: func @replacement_op_for_dead_handle_not_found()
//       CHECK:   "test.container"() ({
//       CHECK:     %0 = "test.bar"() : () -> i32
//       CHECK:   }) : () -> ()
func.func @replacement_op_for_dead_handle_not_found() {
  "test.container"() ({
    %0 = "test.foo"() {replace_with_new_op = "test.bar"} : () -> (i32)
  }) : () -> ()
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["test.container"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %1 = transform.structured.match ops{["test.foo"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  // No error because %1 is dead.
  transform.apply_patterns ["transform.test"] to %0 {} : !transform.any_op
}

// -----

// CHECK-LABEL: func @replacement_op_not_found_silenced()
//       CHECK:   "test.container"() ({
//       CHECK:     %0 = "test.bar"() : () -> i32
//       CHECK:   }) : () -> ()
func.func @replacement_op_not_found_silenced() {
  "test.container"() ({
    %0 = "test.foo"() {replace_with_new_op = "test.bar"} : () -> (i32)
  }) : () -> ()
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["test.container"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %1 = transform.structured.match ops{["test.foo"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns ["transform.test"] to %0 {} {fail_on_payload_replacement_not_found = false}: !transform.any_op
  transform.annotate %1 "annotated" : !transform.any_op
}

// -----

// CHECK-LABEL: func @patterns_apply_only_to_target_body()
//       CHECK:   %0 = "test.foo"() {replace_with_new_op = "test.bar"} : () -> i32
func.func @patterns_apply_only_to_target_body() {
  %0 = "test.foo"() {replace_with_new_op = "test.bar"} : () -> (i32)
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
%0 = transform.structured.match ops{["test.foo"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns ["transform.test"] to %0 {} : !transform.any_op
}

// -----

// CHECK-LABEL: func @erase_tracked_op()
//       CHECK:   "test.container"() ({
//  CHECK-NEXT:   ^bb0:
//  CHECK-NEXT:   }) : () -> ()
func.func @erase_tracked_op() {
  "test.container"() ({
    // expected-remark @below {{matched op}}
    %0 = "test.erase_op"() {replace_with_new_op = "test.foo"} : () -> (i32)
  }) : () -> ()
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["test.container"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %1 = transform.structured.match ops{["test.erase_op"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  transform.test_print_remark_at_operand %1, "matched op" : !transform.any_op
  transform.apply_patterns ["transform.test"] to %0 {} : !transform.any_op
  transform.test_print_remark_at_operand %1, "op was deleted" : !transform.any_op
}

// -----

// CHECK-LABEL: func @canonicalization(
//       CHECK:   %[[c5:.*]] = arith.constant 5 : index
//       CHECK:   return %[[c5]]
func.func @canonicalization(%t: tensor<5xf32>) -> index {
  %c0 = arith.constant 0 : index
  // expected-remark @below {{op was replaced}}
  %dim = tensor.dim %t, %c0 : tensor<5xf32>
  return %dim : index
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["tensor.dim"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns [] to %1 {
    transform.apply_patterns.canonicalization
  } : !transform.any_op
  transform.test_print_remark_at_operand %0, "op was replaced" : !transform.any_op
}
