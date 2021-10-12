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

// expected-error@+1 {{missing ']' closing set of scalable dimensions}}
func @scalable_vector_arg(%arg0: vector<[4xf32>) { }

// -----
