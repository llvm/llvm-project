// This file contains various failure test cases related to the structure of
// the dialect section.

//===--------------------------------------------------------------------===//
// Dialect Name
//===--------------------------------------------------------------------===//

// RUN: not mlir-opt %S/invalid-dialect_section-dialect_string.mlirbc 2>&1 | FileCheck %s --check-prefix=DIALECT_STR
// DIALECT_STR: invalid string index: 15

//===--------------------------------------------------------------------===//
// OpName
//===--------------------------------------------------------------------===//

// RUN: not mlir-opt %S/invalid-dialect_section-opname_dialect.mlirbc 2>&1 | FileCheck %s --check-prefix=OPNAME_DIALECT
// OPNAME_DIALECT: invalid dialect index: 7

// RUN: not mlir-opt %S/invalid-dialect_section-opname_string.mlirbc 2>&1 | FileCheck %s --check-prefix=OPNAME_STR
// OPNAME_STR: invalid string index: 31
