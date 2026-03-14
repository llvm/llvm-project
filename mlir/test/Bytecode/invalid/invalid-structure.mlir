// This file contains various failure test cases related to the structure of
// a bytecode file.

//===--------------------------------------------------------------------===//
// Version
//===--------------------------------------------------------------------===//

// RUN: not mlir-opt %S/invalid-structure-version.mlirbc 2>&1 | FileCheck %s --check-prefix=VERSION
// VERSION: bytecode version 127 is newer than the current version {{[0-9]+}}

//===--------------------------------------------------------------------===//
// Producer
//===--------------------------------------------------------------------===//

// RUN: not mlir-opt %S/invalid-structure-producer.mlirbc 2>&1 | FileCheck %s --check-prefix=PRODUCER
// PRODUCER: malformed null-terminated string, no null character found

//===--------------------------------------------------------------------===//
// Section
//===--------------------------------------------------------------------===//

//===--------------------------------------------------------------------===//
// Missing

// RUN: not mlir-opt %S/invalid-structure-section-missing.mlirbc 2>&1 | FileCheck %s --check-prefix=SECTION_MISSING
// SECTION_MISSING: missing data for top-level section: String (0)

//===--------------------------------------------------------------------===//
// ID

// RUN: not mlir-opt %S/invalid-structure-section-id-unknown.mlirbc 2>&1 | FileCheck %s --check-prefix=SECTION_ID_UNKNOWN
// SECTION_ID_UNKNOWN: invalid section ID: 127

//===--------------------------------------------------------------------===//
// Length

// RUN: not mlir-opt %S/invalid-structure-section-length.mlirbc 2>&1 | FileCheck %s --check-prefix=SECTION_LENGTH
// SECTION_LENGTH: attempting to parse a byte at the end of the bytecode

//===--------------------------------------------------------------------===//
// Duplicate

// RUN: not mlir-opt %S/invalid-structure-section-duplicate.mlirbc 2>&1 | FileCheck %s --check-prefix=SECTION_DUPLICATE
// SECTION_DUPLICATE: duplicate top-level section: String (0)
