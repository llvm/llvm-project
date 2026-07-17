// RUN: mlir-opt %s --split-input-file | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --split-input-file --mlir-print-op-generic | mlir-opt | FileCheck %s


// -----

module {
  // CHECK: tosa.variable @stored_var = dense<3.140000e+00> : tensor<f32>
  tosa.variable @stored_var = dense<3.14> : tensor<f32>

  // CHECK-LABEL: @test_variable_scalar(
  // CHECK-SAME: %[[ADD_VAL:.*]]: tensor<f32>) {
  func.func @test_variable_scalar(%arg0: tensor<f32>) -> () {
    // CHECK: %[[STORED_VAL:.*]] = tosa.variable_read @stored_var : tensor<f32>
    %0 = tosa.variable_read @stored_var : tensor<f32>
    // CHECK: %[[RESULT_ADD:.*]] = tosa.add %[[ADD_VAL]], %[[STORED_VAL]] : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %1 = "tosa.add"(%arg0, %0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    // CHECK: tosa.variable_write @stored_var, %[[RESULT_ADD]] : tensor<f32>
    tosa.variable_write @stored_var, %1 : tensor<f32>
    return
  }
}


// -----

module {
  // CHECK: tosa.variable @stored_var = dense<-1> : tensor<2x4x8xi32>
  tosa.variable @stored_var = dense<-1> : tensor<2x4x8xi32>

  // CHECK-LABEL: @test_variable_tensor(
  // CHECK-SAME: %[[ADD_VAL:.*]]: tensor<2x4x8xi32>) {
  func.func @test_variable_tensor(%arg0: tensor<2x4x8xi32>) -> () {
    // CHECK: %[[STORED_VAL:.*]] = tosa.variable_read @stored_var : tensor<2x4x8xi32>
    %0 = tosa.variable_read @stored_var : tensor<2x4x8xi32>
    // CHECK: %[[RESULT_ADD:.*]] = tosa.add %[[ADD_VAL]], %[[STORED_VAL]] : (tensor<2x4x8xi32>, tensor<2x4x8xi32>) -> tensor<2x4x8xi32>
    %1 = "tosa.add"(%arg0, %0) : (tensor<2x4x8xi32>, tensor<2x4x8xi32>) -> tensor<2x4x8xi32>
    // CHECK: tosa.variable_write @stored_var, %[[RESULT_ADD]] : tensor<2x4x8xi32>
    tosa.variable_write @stored_var, %1 : tensor<2x4x8xi32>
    return
  }
}

// -----

module {
  // CHECK: tosa.variable @stored_var : tensor<f32>
  tosa.variable @stored_var : tensor<f32>

  // CHECK-LABEL: @test_variable_scalar_no_initial_value(
  // CHECK-SAME: %[[ADD_VAL:.*]]: tensor<f32>) {
  func.func @test_variable_scalar_no_initial_value(%arg0: tensor<f32>) -> () {
    // CHECK: %[[STORED_VAL:.*]] = tosa.variable_read @stored_var : tensor<f32>
    %0 = tosa.variable_read @stored_var : tensor<f32>
    // CHECK: %[[RESULT_ADD:.*]] = tosa.add %[[ADD_VAL]], %[[STORED_VAL]] : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %1 = "tosa.add"(%arg0, %0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    // CHECK: tosa.variable_write @stored_var, %[[RESULT_ADD]] : tensor<f32>
    tosa.variable_write @stored_var, %1 : tensor<f32>
    return
  }
}

// -----

module {
  // CHECK: tosa.variable @stored_var : tensor<2x4x8xi32>
  tosa.variable @stored_var : tensor<2x4x8xi32>

  // CHECK-LABEL: @test_variable_tensor_no_initial_value(
  // CHECK-SAME: %[[ADD_VAL:.*]]: tensor<2x4x8xi32>) {
  func.func @test_variable_tensor_no_initial_value(%arg0: tensor<2x4x8xi32>) -> () {
    // CHECK: %[[STORED_VAL:.*]] = tosa.variable_read @stored_var : tensor<2x4x8xi32>
    %0 = tosa.variable_read @stored_var : tensor<2x4x8xi32>
    // CHECK: %[[RESULT_ADD:.*]] = tosa.add %[[ADD_VAL]], %[[STORED_VAL]] : (tensor<2x4x8xi32>, tensor<2x4x8xi32>) -> tensor<2x4x8xi32>
    %1 = "tosa.add"(%arg0, %0) : (tensor<2x4x8xi32>, tensor<2x4x8xi32>) -> tensor<2x4x8xi32>
    // CHECK: tosa.variable_write @stored_var, %[[RESULT_ADD]] : tensor<2x4x8xi32>
    tosa.variable_write @stored_var, %1 : tensor<2x4x8xi32>
    return
  }
}


// -----

module {
  // CHECK: tosa.variable @stored_var : tensor<2x?x8xi32>
  tosa.variable @stored_var : tensor<2x?x8xi32>

  // CHECK-LABEL: @test_variable_tensor_with_unknowns(
  // CHECK-SAME: %[[ADD_VAL:.*]]: tensor<2x4x8xi32>) {
  func.func @test_variable_tensor_with_unknowns(%arg0: tensor<2x4x8xi32>) -> () {
    // CHECK: %[[STORED_VAL:.*]] = tosa.variable_read @stored_var : tensor<2x4x8xi32>
    %0 = tosa.variable_read @stored_var : tensor<2x4x8xi32>
    // CHECK: %[[RESULT_ADD:.*]] = tosa.add %[[ADD_VAL]], %[[STORED_VAL]] : (tensor<2x4x8xi32>, tensor<2x4x8xi32>) -> tensor<2x4x8xi32>
    %1 = "tosa.add"(%arg0, %0) : (tensor<2x4x8xi32>, tensor<2x4x8xi32>) -> tensor<2x4x8xi32>
    // CHECK: tosa.variable_write @stored_var, %[[RESULT_ADD]] : tensor<2x4x8xi32>
    tosa.variable_write @stored_var, %1 : tensor<2x4x8xi32>
    return
  }
}
