// This file contains various failure test cases related to the structure of
// the attribute/type offset section.

//===--------------------------------------------------------------------===//
// Offset
//===--------------------------------------------------------------------===//

// RUN: not mlir-opt %S/invalid-attr_type_offset_section-large_offset.mlirbc 2>&1 | FileCheck %s --check-prefix=LARGE_OFFSET
// LARGE_OFFSET: Attribute or Type entry offset points past the end of section

//===--------------------------------------------------------------------===//
// Trailing Data
//===--------------------------------------------------------------------===//

// RUN: not mlir-opt %S/invalid-attr_type_offset_section-trailing_data.mlirbc 2>&1 | FileCheck %s --check-prefix=TRAILING_DATA
// TRAILING_DATA: unexpected trailing data in the Attribute/Type offset section
