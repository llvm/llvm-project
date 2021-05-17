// RUN: mlir-opt -split-input-file %s -verify-diagnostics

func @test_index_cast_shape_error(%arg0 : tensor<index>) -> tensor<2xi64> {
  // expected-error @+1 {{all non-scalar operands/results must have the same shape and base type}}
  %0 = index_cast %arg0 : tensor<index> to tensor<2xi64>
  return %0 : tensor<2xi64>
}

// -----

func @test_index_cast_tensor_error(%arg0 : tensor<index>) -> i64 {
  // expected-error @+1 {{if an operand is non-scalar, then there must be at least one non-scalar result}}
  %0 = index_cast %arg0 : tensor<index> to i64
  return %0 : i64
}

// -----

func @non_signless_constant() {
  // expected-error @+1 {{requires integer result types to be signless}}
  %0 = constant 0 : ui32
  return
}

// -----

func @non_signless_constant() {
  // expected-error @+1 {{requires integer result types to be signless}}
  %0 = constant 0 : si32
  return
}

// -----

func @unsupported_attribute() {
  // expected-error @+1 {{unsupported 'value' attribute: "" : index}}
  %0 = constant "" : index
  return
}
