// RUN: mlir-opt --tosa-to-mlprogram %s -o -| FileCheck %s

module {
  // CHECK: ml_program.global private mutable @"1"(dense<7.000000e+00> : tensor<1xf32>) : tensor<1xf32>
  tosa.variable 1 = dense<7.000000e+00> : tensor<1xf32>
  func.func @test_stateful_ops(%arg0: tensor<1xf32>) -> (tensor<1xf32>) {
    // CHECK: ml_program.global_store @"1" = %arg0 : tensor<1xf32>
    tosa.variable.write 1, %arg0 : tensor<1xf32>
    // CHECK: %[[LOAD:.+]] = ml_program.global_load @"1" : tensor<1xf32>
    %0 = tosa.variable.read 1 : tensor<1xf32>
    return %0 : tensor<1xf32>
  }
}