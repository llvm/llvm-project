// # RUN: mlir-opt %s | FileCheck %s
// # RUN: mlir-opt %s --mlir-print-op-generic  | FileCheck %s --check-prefix=GENERIC

// CHECK-LABEL: func @pretty_names
// CHECK-GENERIC: "func"()
func @pretty_names() {
  %x = test.string_attr_pretty_name
  // CHECK: %x = test.string_attr_pretty_name
  // GENERIC: %0 = "test.string_attr_pretty_name"()
  return
  // CHECK: return
  // GENERIC: "std.return"()
}
