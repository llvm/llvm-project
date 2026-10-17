// RUN: mlir-opt -convert-spirv-to-llvm -verify-diagnostics %s

// `llvm.intr.expect` only accepts scalar integers, so vector operands are
// rejected.

spirv.func @expect_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) "None" {
  // expected-error@+1 {{failed to legalize operation 'spirv.KHR.Expect' that was explicitly marked illegal}}
  %0 = spirv.KHR.Expect %arg0, %arg1 : vector<4xi32>
  spirv.Return
}
