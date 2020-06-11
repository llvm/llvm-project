// RUN: mlir-opt -split-input-file %s -verify-diagnostics

// CHECK-LABEL: test_index_cast_shape_error
func @test_index_cast_shape_error(%arg0 : tensor<index>) -> tensor<2xi64> {
  // expected-error @+1 {{operand type 'tensor<index>' and result type 'tensor<2xi64>' are cast incompatible}}
  %0 = index_cast %arg0 : tensor<index> to tensor<2xi64>
  return %0 : tensor<2xi64>
}

// -----

// CHECK-LABEL: test_index_cast_tensor_error
func @test_index_cast_tensor_error(%arg0 : tensor<index>) -> i64 {
  // expected-error @+1 {{operand type 'tensor<index>' and result type 'i64' are cast incompatible}}
  %0 = index_cast %arg0 : tensor<index> to i64
  return %0 : i64
}
