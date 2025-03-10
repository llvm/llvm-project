// RUN: mlir-opt %s | FileCheck %s
module {
  // CHECK: %[[v0:[^ ]*]] = "test_irdl_to_cpp.bar"() : () -> !test_irdl_to_cpp.foo
  %0 = "test_irdl_to_cpp.bar"() : () -> !test_irdl_to_cpp.foo
}