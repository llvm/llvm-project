// RUN: mlir-opt --tosa-to-mlprogram %s -split-input-file -o -| FileCheck %s

module {
  // CHECK: ml_program.global private mutable @var_x(dense<7.000000e+00> : tensor<1xf32>) : tensor<1xf32>
  tosa.variable @var_x = dense<7.000000e+00> : tensor<1xf32>
  func.func @test_stateful_ops(%arg0: tensor<1xf32>) -> (tensor<1xf32>) {
    // CHECK: ml_program.global_store @var_x = %arg0 : tensor<1xf32>
    tosa.variable_write @var_x, %arg0 : tensor<1xf32>
    // CHECK: %[[LOAD:.+]] = ml_program.global_load @var_x : tensor<1xf32>
    %0 = tosa.variable_read @var_x : tensor<1xf32>
    return %0 : tensor<1xf32>
  }
}

// -----

module {
  // CHECK: ml_program.global private mutable @var_x : tensor<f32>
  tosa.variable @var_x : tensor<f32>
  func.func @test_stateful_ops(%arg0: tensor<f32>) -> (tensor<f32>) {
    // CHECK: ml_program.global_store @var_x = %arg0 : tensor<f32>
    tosa.variable_write @var_x, %arg0 : tensor<f32>
    // CHECK: %[[LOAD:.+]] = ml_program.global_load @var_x : tensor<f32>
    %0 = tosa.variable_read @var_x : tensor<f32>
    return %0 : tensor<f32>
  }
}