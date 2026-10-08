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

//===--------------------------------------------------------------------===//
// Group Count
//===--------------------------------------------------------------------===//

// The dialect groups contain more entries than the declared number of
// attributes.
// RUN: not mlir-opt %S/invalid-attr_type_offset_section-attr_group_count.mlirbc 2>&1 | FileCheck %s --check-prefix=ATTR_GROUP_COUNT
// ATTR_GROUP_COUNT: Attribute or Type dialect group entries exceed the declared number of entries (1)

// The dialect groups contain more entries than the declared number of types.
// RUN: not mlir-opt %S/invalid-attr_type_offset_section-type_group_count.mlirbc 2>&1 | FileCheck %s --check-prefix=TYPE_GROUP_COUNT
// TYPE_GROUP_COUNT: Attribute or Type dialect group entries exceed the declared number of entries (1)
