// RUN: mlir-opt --no-implicit-module --convert-func-to-spirv -verify-diagnostics %s

// Verify that converting a top-level func.func op via --no-implicit-module
// fails gracefully without crashing. The dialect conversion infrastructure
// must detect that the root operation has no parent block and emit an error.
// Regression test for https://github.com/llvm/llvm-project/issues/60491

// expected-error@below {{dialect conversion requires that the root operation is nested within a block when it is replaced or erased by the conversion}}
func.func private @foo()
