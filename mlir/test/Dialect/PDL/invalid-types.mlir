// RUN: mlir-opt %s -split-input-file -verify-diagnostics

//===----------------------------------------------------------------------===//
// pdl::RangeType
//===----------------------------------------------------------------------===//

// expected-error@below {{element of pdl.range cannot be another range, but got'!pdl.range<value>'}}
#invalid_element = !pdl.range<range<value>>
