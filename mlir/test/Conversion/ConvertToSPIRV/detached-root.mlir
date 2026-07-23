// RUN: mlir-opt --no-implicit-module --convert-func-to-spirv -verify-diagnostics %s

// Verify that applying --convert-func-to-spirv to a detached top-level op
// (via --no-implicit-module) produces a clear diagnostic rather than crashing.
// The pass should refuse to operate on an op that has no parent block.
//
// Regression test for https://github.com/llvm/llvm-project/issues/60491

// expected-error@below {{'convert-func-to-spirv' pass requires the target operation to be nested in a block}}
func.func private @foo()
