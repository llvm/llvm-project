// RUN: mlir-opt %s -convert-std-to-llvm -verify-diagnostics -split-input-file

// Should not crash on unsupported types in function signatures.
func private @unsupported_signature() -> tensor<10 x i32>

// -----

func private @partially_supported_signature() -> (vector<10 x i32>, tensor<10 x i32>)
