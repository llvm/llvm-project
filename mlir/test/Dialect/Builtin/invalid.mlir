// RUN: mlir-opt %s -split-input-file -verify-diagnostics

//===----------------------------------------------------------------------===//
// UnrealizedConversionCastOp
//===----------------------------------------------------------------------===//

// expected-error@+1 {{expected at least one result for cast operation}}
"builtin.unrealized_conversion_cast"() : () -> ()

// -----

//===----------------------------------------------------------------------===//
// VectorType
//===----------------------------------------------------------------------===//

// expected-error@+1 {{missing ']' closing scalable dimension}}
func.func @scalable_vector_arg(%arg0: vector<[4xf32>) { }

// -----

// expected-error@+1 {{missing ']' closing scalable dimension}}
func.func @scalable_vector_arg(%arg0: vector<[4x4]xf32>) { }
