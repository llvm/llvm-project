// RUN: mlir-opt %s -split-input-file -tosa-validate | FileCheck %s

// --- First case: without TOSA Operation (#173370 bug reproduction) ---

// CHECK-LABEL: func.func @test_validation_pass_init
func.func @test_validation_pass_init(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  // CHECK: math.asin
  %0 = math.asin %arg0 : tensor<1xf32>
  return %0 : tensor<1xf32>
}

// -----

// --- Second case: with TOSA Operation ---

// CHECK-LABEL: func.func @test_tosa_ops
func.func @test_tosa_ops(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32> {
  // CHECK: tosa.add
  %0 = tosa.add %arg0, %arg1 : (tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
  return %0 : tensor<1x2x3x4xf32>
}
