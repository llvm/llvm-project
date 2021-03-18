// RUN: mlir-opt --split-input-file --tosa-to-standard %s -verify-diagnostics -o -| FileCheck %s

// CHECK-LABEL: func @const_test
func @const_test() -> (tensor<i32>) {
  // CHECK: [[C3:%.+]] = constant dense<3> : tensor<i32>
  %0 = "tosa.const"() {value = dense<3> : tensor<i32>} : () -> tensor<i32>

  // CHECK: return [[C3]]
  return %0 : tensor<i32>
}

// ----

func @slice(%arg0: tensor<6xf32>) ->() {
  // CHECK: [[SLICE:%.+]] = subtensor %arg0[2] [1] [1]
  %0 = "tosa.slice"(%arg0) {start = [2], size = [1]} : (tensor<6xf32>)  -> (tensor<1xf32>)
  return
}
