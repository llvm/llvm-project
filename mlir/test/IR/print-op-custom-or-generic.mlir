// # RUN: mlir-opt %s -split-input-file | FileCheck %s
// # RUN: mlir-opt %s -mlir-print-op-generic -split-input-file  | FileCheck %s --check-prefix=GENERIC

// Check that `printCustomOrGenericOp` and `printGenericOp` print the right
// assembly format. For operations without custom format, both should print the
// generic format.

// CHECK-LABEL: func @op_with_custom_printer
// CHECK-GENERIC-LABEL: "func"()
func.func @op_with_custom_printer() {
  %x = test.string_attr_pretty_name
  // CHECK: %x = test.string_attr_pretty_name
  // GENERIC: %0 = "test.string_attr_pretty_name"()
  return
  // CHECK: return
  // GENERIC: "func.return"()
}

// -----

// CHECK-LABEL: func @op_without_custom_printer
// CHECK-GENERIC: "func"()
func.func @op_without_custom_printer() {
  // CHECK: "test.result_type_with_trait"() : () -> !test.test_type_with_trait
  // GENERIC: "test.result_type_with_trait"() : () -> !test.test_type_with_trait
  "test.result_type_with_trait"() : () -> !test.test_type_with_trait
  return
}
