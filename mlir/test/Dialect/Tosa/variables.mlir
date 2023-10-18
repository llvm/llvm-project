// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s


// -----
// CHECK-LABEL:   @test_variable_scalar(
// CHECK-SAME:                        %[[ADD_VAL:.*]]: tensor<f32>) {
func.func @test_variable_scalar(%arg0: tensor<f32>) -> () {
  // CHECK:           tosa.variable @stored_var = dense<3.140000e+00> : tensor<f32>
  tosa.variable @stored_var = dense<3.14> : tensor<f32>
  // CHECK:           %[[STORED_VAL:.*]] = tosa.variable.read @stored_var : tensor<f32>
  %0 = tosa.variable.read @stored_var : tensor<f32>
  // CHECK:           %[[RESULT_ADD:.*]] = tosa.add %[[ADD_VAL]], %[[STORED_VAL]] : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %1 = "tosa.add"(%arg0, %0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK:           tosa.variable.write @stored_var, %[[RESULT_ADD]] : tensor<f32>
  tosa.variable.write @stored_var, %1 : tensor<f32>
  return
}

// -----
// CHECK-LABEL:   @test_variable_tensor(
// CHECK-SAME:                        %[[ADD_VAL:.*]]: tensor<2x4x8xi32>) {
func.func @test_variable_tensor(%arg0: tensor<2x4x8xi32>) -> () {
  // CHECK:           tosa.variable @stored_var = dense<-1> : tensor<2x4x8xi32>
  tosa.variable @stored_var = dense<-1> : tensor<2x4x8xi32>
  // CHECK:           %[[STORED_VAL:.*]] = tosa.variable.read @stored_var : tensor<2x4x8xi32>
  %0 = tosa.variable.read @stored_var : tensor<2x4x8xi32>
  // CHECK:           %[[RESULT_ADD:.*]] = tosa.add %[[ADD_VAL]], %[[STORED_VAL]] : (tensor<2x4x8xi32>, tensor<2x4x8xi32>) -> tensor<2x4x8xi32>
  %1 = "tosa.add"(%arg0, %0) : (tensor<2x4x8xi32>, tensor<2x4x8xi32>) -> tensor<2x4x8xi32>
  // CHECK:           tosa.variable.write @stored_var, %[[RESULT_ADD]] : tensor<2x4x8xi32>
  tosa.variable.write @stored_var, %1 : tensor<2x4x8xi32>
  return
}
