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
  transform.apply_patterns to %0 {
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
  transform.apply_patterns to %0 {
    transform.apply_patterns.transform.test_patterns
  } : !transform.any_op
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
  transform.apply_patterns to %0 {
    transform.apply_patterns.transform.test_patterns
  } : !transform.any_op
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
  transform.apply_patterns to %0 {
    transform.apply_patterns.transform.test_patterns
  } {transform.silence_tracking_failures} : !transform.any_op
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
  transform.apply_patterns to %0 {
    transform.apply_patterns.transform.test_patterns
  } : !transform.any_op
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
  transform.apply_patterns to %0 {
    transform.apply_patterns.transform.test_patterns
  } : !transform.any_op
  // No marker should be printed.
  transform.test_print_remark_at_operand %1, "op was deleted" : !transform.any_op
}

// -----

// CHECK-LABEL: func @erase_tracked_op_in_named_sequence()
//       CHECK:   "test.container"() ({
//  CHECK-NEXT:   ^bb0:
//  CHECK-NEXT:   }) : () -> ()
module {
  func.func @erase_tracked_op_in_named_sequence() {
    "test.container"() ({
      // expected-remark @below {{matched op}}
      %0 = "test.erase_op"() {replace_with_new_op = "test.foo"} : () -> (i32)
    }) : () -> ()
    return
  }

  module attributes { transform.with_named_sequence } {
    transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> () {
      transform.apply_patterns to %arg0 {
        transform.apply_patterns.transform.test_patterns
      } : !transform.any_op
      transform.yield
    }

    transform.sequence failures(propagate) {
    ^bb1(%arg1: !transform.any_op):
      %0 = transform.structured.match ops{["test.container"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.match ops{["test.erase_op"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      transform.test_print_remark_at_operand %1, "matched op" : !transform.any_op
      include @foo failures(propagate) (%0) : (!transform.any_op) -> ()
      // No marker should be printed.
      transform.test_print_remark_at_operand %1, "op was deleted" : !transform.any_op
    }
  }
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
  transform.apply_patterns to %1 {
    transform.apply_patterns.canonicalization
  } : !transform.any_op
  transform.test_print_remark_at_operand %0, "op was replaced" : !transform.any_op
}

// -----

// expected-note @below{{target payload op}}
module {
  func.func @invalid_pattern_application_to_transform_ir() {
    return
  }

  module {
    transform.sequence failures(propagate) {
    ^bb1(%arg1: !transform.any_op):
      // expected-error @below {{cannot apply transform to itself (or one of its ancestors)}}
      transform.apply_patterns to %arg1 {
        transform.apply_patterns.canonicalization
      } : !transform.any_op
    }
  }
}

// -----

// CHECK-LABEL: func @canonicalization_and_cse(
//   CHECK-NOT:   memref.subview
//   CHECK-NOT:   memref.copy
func.func @canonicalization_and_cse(%m: memref<5xf32>) {
  %c2 = arith.constant 2 : index
  %s0 = memref.subview %m[1] [2] [1] : memref<5xf32> to memref<2xf32, strided<[1], offset: 1>>
  %s1 = memref.subview %m[1] [%c2] [1] : memref<5xf32> to memref<?xf32, strided<[1], offset: 1>>
  memref.copy %s0, %s1 : memref<2xf32, strided<[1], offset: 1>> to memref<?xf32, strided<[1], offset: 1>>
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %1 {
    transform.apply_patterns.canonicalization
  } {apply_cse} : !transform.any_op
}

// -----

// CHECK-LABEL: func @full_dialect_conversion
//  CHECK-NEXT:   %[[m:.*]] = "test.new_op"() : () -> memref<5xf32>
//  CHECK-NEXT:   %[[cast:.*]] = builtin.unrealized_conversion_cast %0 : memref<5xf32> to tensor<5xf32>
//  CHECK-NEXT:   return %[[cast]]
func.func @full_dialect_conversion() -> tensor<5xf32> {
  %0 = "test.foo"() {replace_with_new_op = "test.bar"} : () -> (tensor<5xf32>)
  return %0 : tensor<5xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  transform.apply_conversion_patterns to %0 {
    transform.apply_conversion_patterns.transform.test_conversion_patterns
  } with type_converter {
    transform.apply_conversion_patterns.transform.test_type_converter
  } {legal_ops = ["func.func", "func.return", "test.new_op"]}
      : !transform.any_op
}

// -----

// Full dialect conversion fails because test.bar is not replaced and not legal.

// expected-note @below{{target op}}
func.func @full_dialect_conversion_failed() -> tensor<5xf32> {
  %0 = "test.foo"() {replace_with_new_op = "test.bar"} : () -> (tensor<5xf32>)
  // expected-error @below{{failed to legalize operation 'test.bar'}}
  "test.bar"() : () -> ()
  return %0 : tensor<5xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  // expected-error @below{{dialect conversion failed}}
  transform.apply_conversion_patterns to %0 {
    transform.apply_conversion_patterns.transform.test_conversion_patterns
  } with type_converter {
    transform.apply_conversion_patterns.transform.test_type_converter
  } {legal_ops = ["func.func", "func.return", "test.new_op"]}
      : !transform.any_op
}

// -----

// Partial dialect conversion succeeds because test.bar is not explicitly
// illegal.

// CHECK-LABEL: func @partial_dialect_conversion
//  CHECK-NEXT:   %[[m:.*]] = "test.new_op"() : () -> memref<5xf32>
//  CHECK-NEXT:   %[[cast:.*]] = builtin.unrealized_conversion_cast %0 : memref<5xf32> to tensor<5xf32>
//  CHECK-NEXT:   "test.bar"
//  CHECK-NEXT:   return %[[cast]]
func.func @partial_dialect_conversion() -> tensor<5xf32> {
  %0 = "test.foo"() {replace_with_new_op = "test.bar"} : () -> (tensor<5xf32>)
  "test.bar"() : () -> ()
  return %0 : tensor<5xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  transform.apply_conversion_patterns to %0 {
    transform.apply_conversion_patterns.transform.test_conversion_patterns
  } with type_converter {
    transform.apply_conversion_patterns.transform.test_type_converter
  } {legal_ops = ["func.func", "func.return", "test.new_op"],
     partial_conversion} : !transform.any_op
}

// -----

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  // expected-error @below{{pattern descriptor does not specify type converter and apply_conversion_patterns op has no default type converter}}
  transform.apply_conversion_patterns to %0 {
    // expected-note @below{{pattern descriptor op}}
    transform.apply_conversion_patterns.transform.test_conversion_patterns
  } {illegal_ops = ["test.foo"]} : !transform.any_op
}

// -----

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  transform.apply_conversion_patterns to %0 {
    // expected-error @below{{expected LLVMTypeConverter}}
    transform.apply_conversion_patterns.dialect_to_llvm "test"
  } with type_converter {
    transform.apply_conversion_patterns.transform.test_type_converter
  } {illegal_ops = ["test.foo"],
     legal_ops = ["func.func", "func.return", "test.new_op"]}
      : !transform.any_op
}

// -----

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  transform.apply_conversion_patterns to %0 {
    // expected-error @below{{unknown dialect or dialect not loaded: this_dialect_does_not_exist}}
    transform.apply_conversion_patterns.dialect_to_llvm "this_dialect_does_not_exist"
  } with type_converter {
    transform.apply_conversion_patterns.memref.memref_to_llvm_type_converter
  } {illegal_ops = ["test.foo"],
     legal_ops = ["func.func", "func.return", "test.new_op"]}
      : !transform.any_op
}

// -----

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  transform.apply_conversion_patterns to %0 {
    // expected-error @below{{dialect does not implement ConvertToLLVMPatternInterface or extension was not loaded: transform}}
    transform.apply_conversion_patterns.dialect_to_llvm "transform"
  } with type_converter {
    transform.apply_conversion_patterns.memref.memref_to_llvm_type_converter
  } {illegal_ops = ["test.foo"],
     legal_ops = ["func.func", "func.return", "test.new_op"]}
      : !transform.any_op
}
