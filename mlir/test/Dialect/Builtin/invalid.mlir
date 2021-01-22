// RUN: mlir-opt %s -split-input-file -verify-diagnostics

//===----------------------------------------------------------------------===//
// UnrealizedConversionCastOp
//===----------------------------------------------------------------------===//

// expected-error@+1 {{expected at least one result for cast operation}}
"unrealized_conversion_cast"() : () -> ()

// -----

