// This file contains various failure test cases related to the structure of
// the string section.

//===--------------------------------------------------------------------===//
// Count
//===--------------------------------------------------------------------===//

// RUN: not mlir-opt %S/invalid-string_section-count.mlirbc 2>&1 | FileCheck %s --check-prefix=COUNT
// COUNT: attempting to parse a byte at the end of the bytecode

//===--------------------------------------------------------------------===//
// Invalid String
//===--------------------------------------------------------------------===//

// RUN: not mlir-opt %S/invalid-string_section-no_string.mlirbc 2>&1 | FileCheck %s --check-prefix=NO_STRING
// NO_STRING: attempting to parse a byte at the end of the bytecode

// RUN: not mlir-opt %S/invalid-string_section-large_string.mlirbc 2>&1 | FileCheck %s --check-prefix=LARGE_STRING
// LARGE_STRING: string size exceeds the available data size

//===--------------------------------------------------------------------===//
// Trailing data
//===--------------------------------------------------------------------===//

// RUN: not mlir-opt %S/invalid-string_section-trailing_data.mlirbc 2>&1 | FileCheck %s --check-prefix=TRAILING_DATA
// TRAILING_DATA: unexpected trailing data between the offsets for strings and their data
