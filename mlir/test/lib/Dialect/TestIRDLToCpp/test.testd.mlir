// RUN: mlir-opt %s | FileCheck %s
// CHECK:       module {
// CHECK-LABEL:   [[v1: *]] = "test_irdl_to_cpp.bar"
module {
  %0 = "test_irdl_to_cpp.bar"() : () -> !test_irdl_to_cpp.foo
}
