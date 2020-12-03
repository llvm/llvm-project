// RUN: mlir-opt --test-constant-fold %s | FileCheck %s

// CHECK-LABEL: func @test_const
func @test_const(%arg0 : index) -> tensor<4xi32> {
  // CHECK: "tosa.const"
  %0 = "tosa.const"() {value = dense<[3, 0, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  return %0 : tensor<4xi32>
}
