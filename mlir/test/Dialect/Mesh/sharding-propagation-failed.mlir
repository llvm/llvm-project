// RUN: mlir-opt --pass-pipeline="builtin.module(func.func(sharding-propagation))" %s -verify-diagnostics

// expected-error @+1 {{'func.func' op only one block is supported!}}
func.func private @no_block_function(i64)
