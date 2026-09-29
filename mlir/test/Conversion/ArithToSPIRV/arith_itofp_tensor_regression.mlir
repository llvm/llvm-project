// RUN: mlir-opt -split-input-file -convert-arith-to-spirv -verify-diagnostics %s

// -----

// Regression test: arith.uitofp on tensor types should not crash
// (https://github.com/llvm/llvm-project/issues/226380).
func.func @uitofp_tensor_no_crash(%arg0: tensor<1xi32>) -> tensor<1xf32> {
  // expected-error @+1 {{failed to legalize operation 'arith.uitofp'}}
  %0 = arith.uitofp %arg0 : tensor<1xi32> to tensor<1xf32>
  return %0 : tensor<1xf32>
}

// -----

// Regression test: arith.sitofp on tensor types should not crash
// (https://github.com/llvm/llvm-project/issues/226380).
func.func @sitofp_tensor_no_crash(%arg0: tensor<1xi32>) -> tensor<1xf32> {
  // expected-error @+1 {{failed to legalize operation 'arith.sitofp'}}
  %0 = arith.sitofp %arg0 : tensor<1xi32> to tensor<1xf32>
  return %0 : tensor<1xf32>
}

// -----

// Dynamically shaped tensors must not crash uitofp lowering.
func.func @uitofp_dynamic_tensor_no_crash(%arg0: tensor<?xi16>) -> tensor<?xf32> {
  // expected-error @+1 {{failed to legalize operation 'arith.uitofp'}}
  %0 = arith.uitofp %arg0 : tensor<?xi16> to tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// Multi-dimensional tensors must not crash sitofp lowering.
func.func @sitofp_multidim_tensor_no_crash(%arg0: tensor<2x2xi16>) -> tensor<2x2xf32> {
  // expected-error @+1 {{failed to legalize operation 'arith.sitofp'}}
  %0 = arith.sitofp %arg0 : tensor<2x2xi16> to tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

