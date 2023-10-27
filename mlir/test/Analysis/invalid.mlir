// RUN: mlir-opt -split-input-file -pass-pipeline='builtin.module(func.func(test-foo-analysis))'  %s -verify-diagnostics

// -----

// expected-error @+1 {{expected at least one block in the region}}
func.func private @no_block_func_declaration() -> ()

