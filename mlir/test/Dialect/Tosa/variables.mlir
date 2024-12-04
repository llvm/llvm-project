// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s


// -----
// CHECK-LABEL:   @test_variable_scalar(
// CHECK-SAME:                        %[[ADD_VAL:.*]]: tensor<f32>) {
func.func @test_variable_scalar(%arg0: tensor<f32>) -> () {
  // CHECK:           tosa.variable 1 = dense<3.140000e+00> : tensor<f32>
  tosa.variable 1 = dense<3.14> : tensor<f32>
  // CHECK:           %[[VAR_1:.*]] = tosa.variable.read 1 : tensor<f32>
  %0 = tosa.variable.read 1 : tensor<f32>
  // CHECK:           %[[RESULT_ADD:.*]] = tosa.add %[[ADD_VAL]], %[[VAR_1]] : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %1 = "tosa.add"(%arg0, %0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK:           tosa.variable.write 1, %[[RESULT_ADD]] : tensor<f32>
  tosa.variable.write 1, %1 : tensor<f32>
  return
}

// -----
// CHECK-LABEL:   @test_variable_tensor(
// CHECK-SAME:                        %[[ADD_VAL:.*]]: tensor<2x4x8xi32>) {
func.func @test_variable_tensor(%arg0: tensor<2x4x8xi32>) -> () {
  // CHECK:           tosa.variable 1 = dense<-1> : tensor<2x4x8xi32>
  tosa.variable 1 = dense<-1> : tensor<2x4x8xi32>
  // CHECK:           %[[VAL_1:.*]] = tosa.variable.read 1 : tensor<2x4x8xi32>
  %0 = tosa.variable.read 1 : tensor<2x4x8xi32>
  // CHECK:           %[[RESULT_ADD:.*]] = tosa.add %[[ADD_VAL]], %[[VAL_1]] : (tensor<2x4x8xi32>, tensor<2x4x8xi32>) -> tensor<2x4x8xi32>
  %1 = "tosa.add"(%arg0, %0) : (tensor<2x4x8xi32>, tensor<2x4x8xi32>) -> tensor<2x4x8xi32>
  // CHECK:           tosa.variable.write 1, %[[RESULT_ADD]] : tensor<2x4x8xi32>
  tosa.variable.write 1, %1 : tensor<2x4x8xi32>
  return
}