// RUN: mlir-opt %s | FileCheck %s
// CHECK:       module {
// CHECK-NEXT:   [[v1:[^ ]+]] = "test_irdl_to_cpp.bar"() : () -> !test_irdl_to_cpp.foo
module {
  %0 = "test_irdl_to_cpp.bar"() : () -> !test_irdl_to_cpp.foo
}
