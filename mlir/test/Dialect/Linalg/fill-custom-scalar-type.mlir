// RUN: mlir-opt %s | FileCheck %s
// This test verifies that linalg.fill and linalg.generic accept non-builtin
// scalar types (e.g., custom dialect types) as operands.

// CHECK-LABEL: @fill_non_builtin_scalar_type
func.func @fill_non_builtin_scalar_type(%src: !test.test_type, %dst: tensor<4x!test.test_type>) -> tensor<4x!test.test_type> {
  // CHECK: linalg.fill
  %result = linalg.fill ins(%src : !test.test_type) outs(%dst : tensor<4x!test.test_type>) -> tensor<4x!test.test_type>
  return %result : tensor<4x!test.test_type>
}
