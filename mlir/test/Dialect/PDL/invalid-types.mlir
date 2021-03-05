// RUN: mlir-opt %s -split-input-file -verify-diagnostics

//===----------------------------------------------------------------------===//
// pdl::RangeType
//===----------------------------------------------------------------------===//

// expected-error@+2 {{element of pdl.range cannot be another range, but got'!pdl.range<value>'}}
// expected-error@+1 {{invalid 'pdl' type}}
#invalid_element = !pdl.range<range<value>>
