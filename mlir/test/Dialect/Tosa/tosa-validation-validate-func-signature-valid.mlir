// RUN: mlir-opt %s -split-input-file -verify-diagnostics -tosa-attach-target="specification_version=1.1.draft profiles=pro_int,pro_fp extensions=shape level=8k" -tosa-validate="strict-op-spec-alignment validate-function-signature" | FileCheck %s

// CHECK-LABEL: test_valid_arguments
func.func @test_valid_arguments(%arg0: tensor<1x2x3x4xi8>, %arg1: tensor<1x2x3xf32>) {
  return
}

// -----

// CHECK-LABEL: test_valid_results
func.func @test_valid_results() -> (tensor<1x2x3x4xi8>, tensor<1x2x3xf32>) {
  %0 = arith.constant dense<0> : tensor<1x2x3x4xi8>
  %1 = arith.constant dense<0.0> : tensor<1x2x3xf32>
  return %0, %1 : tensor<1x2x3x4xi8>, tensor<1x2x3xf32>
}
