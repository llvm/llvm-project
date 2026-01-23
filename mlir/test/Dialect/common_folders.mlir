// RUN: mlir-opt %s --test-fold-type-converting-op --split-input-file | FileCheck %s

// CHECK-LABEL: @test_fold_unary_op_f32_to_si32(
func.func @test_fold_unary_op_f32_to_si32() -> tensor<4x2xsi32> {
  // CHECK-NEXT: %[[POSITIVE_ONE:.*]] = arith.constant dense<1> : tensor<4x2xsi32>
  // CHECK-NEXT: return %[[POSITIVE_ONE]] : tensor<4x2xsi32>
  %operand = arith.constant dense<5.1> : tensor<4x2xf32>
  %sign = test.sign %operand : (tensor<4x2xf32>) -> tensor<4x2xsi32>
  return %sign : tensor<4x2xsi32>
}

// -----

// CHECK-LABEL: @test_fold_binary_op_f32_to_i1(
func.func @test_fold_binary_op_f32_to_i1() -> tensor<8xi1> {
  // CHECK-NEXT: %[[FALSE:.*]] = arith.constant dense<false> : tensor<8xi1>
  // CHECK-NEXT: return %[[FALSE]] : tensor<8xi1>
  %lhs = arith.constant dense<5.1> : tensor<8xf32>
  %rhs = arith.constant dense<4.2> : tensor<8xf32>
  %less_than = test.less_than %lhs, %rhs : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xi1>
  return %less_than : tensor<8xi1>
}
