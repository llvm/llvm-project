// RUN: mlir-opt %s -one-shot-bufferize="test-analysis-only analysis-heuristic=bottom-up-from-terminators" -split-input-file | FileCheck %s

// CHECK-LABEL: func @simple_test(
func.func @simple_test(%lb: index, %ub: index, %step: index, %f1: f32, %f2: f32) -> (tensor<5xf32>, tensor<5xf32>) {
  %c0 = arith.constant 0 : index
  %p = arith.constant 0.0 : f32

  // Make sure that ops that feed into region terminators bufferize in-place
  // (if possible).
  // Note: This test case fails to bufferize with a "top-down" or "bottom-up"
  // heuristic.

  %0 = tensor.empty() : tensor<5xf32>
  %1 = scf.for %iv = %lb to %ub step %step iter_args(%t = %0) -> (tensor<5xf32>) {
    // CHECK: linalg.fill {__inplace_operands_attr__ = ["none", "false"]}
    %2 = linalg.fill ins(%f1 : f32) outs(%t : tensor<5xf32>) -> tensor<5xf32>
    // CHECK: linalg.fill {__inplace_operands_attr__ = ["none", "true"]}
    %3 = linalg.fill ins(%f2 : f32) outs(%t : tensor<5xf32>) -> tensor<5xf32>
    %4 = vector.transfer_read %2[%c0], %p : tensor<5xf32>, vector<5xf32>
    vector.print %4 : vector<5xf32>
    scf.yield %3 : tensor<5xf32>
  }

  %5 = tensor.empty() : tensor<5xf32>
  %6 = scf.for %iv = %lb to %ub step %step iter_args(%t = %0) -> (tensor<5xf32>) {
    // CHECK: linalg.fill {__inplace_operands_attr__ = ["none", "true"]}
    %7 = linalg.fill ins(%f1 : f32) outs(%t : tensor<5xf32>) -> tensor<5xf32>
    // CHECK: linalg.fill {__inplace_operands_attr__ = ["none", "false"]}
    %8 = linalg.fill ins(%f2 : f32) outs(%t : tensor<5xf32>) -> tensor<5xf32>
    %9 = vector.transfer_read %8[%c0], %p : tensor<5xf32>, vector<5xf32>
    vector.print %9 : vector<5xf32>
    scf.yield %7 : tensor<5xf32>
  }

  return %1, %6 : tensor<5xf32>, tensor<5xf32>
}
