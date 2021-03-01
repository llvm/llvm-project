// RUN: mlir-opt --split-input-file --tosa-to-standard %s -verify-diagnostics -o -| FileCheck %s

// CHECK-LABEL: func @const_test
func @const_test() -> (tensor<i32>) {
  // CHECK: [[C3:%.+]] = constant dense<3> : tensor<i32>
  %0 = "tosa.const"() {value = dense<3> : tensor<i32>} : () -> tensor<i32>

  // CHECK: return [[C3]]
  return %0 : tensor<i32>
}
