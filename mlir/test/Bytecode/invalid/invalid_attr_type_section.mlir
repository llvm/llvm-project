// This file contains various failure test cases related to the structure of
// the attribute/type offset section.

// Bytecode currently does not support big-endian platforms
// UNSUPPORTED: s390x-

//===--------------------------------------------------------------------===//
// Index
//===--------------------------------------------------------------------===//

// RUN: not mlir-opt %S/invalid-attr_type_section-index.mlirbc 2>&1 | FileCheck %s --check-prefix=INDEX
// INDEX: invalid Attribute index: 3

//===--------------------------------------------------------------------===//
// Trailing Data
//===--------------------------------------------------------------------===//

// RUN: not mlir-opt %S/invalid-attr_type_section-trailing_data.mlirbc 2>&1 | FileCheck %s --check-prefix=TRAILING_DATA
// TRAILING_DATA: trailing characters found after Attribute assembly format: trailing
