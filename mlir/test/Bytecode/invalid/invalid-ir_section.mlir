// This file contains various failure test cases related to the structure of
// the IR section.

//===--------------------------------------------------------------------===//
// Operations
//===--------------------------------------------------------------------===//

//===--------------------------------------------------------------------===//
// Name

// RUN: not mlir-opt %S/invalid-ir_section-opname.mlirbc -allow-unregistered-dialect 2>&1 | FileCheck %s --check-prefix=OP_NAME
// OP_NAME: invalid operation name index: 14

//===--------------------------------------------------------------------===//
// Loc

// RUN: not mlir-opt %S/invalid-ir_section-loc.mlirbc -allow-unregistered-dialect 2>&1 | FileCheck %s --check-prefix=OP_LOC
// OP_LOC: expected attribute of type: {{.*}}, but got: {attra = 10 : i64, attrb = #bytecode.attr}

//===--------------------------------------------------------------------===//
// Attr

// RUN: not mlir-opt %S/invalid-ir_section-attr.mlirbc -allow-unregistered-dialect 2>&1 | FileCheck %s --check-prefix=OP_ATTR
// OP_ATTR: expected attribute value

//===--------------------------------------------------------------------===//
// Operands

// RUN: not mlir-opt %S/invalid-ir_section-operands.mlirbc -allow-unregistered-dialect 2>&1 | FileCheck %s --check-prefix=OP_OPERANDS
// OP_OPERANDS: expected attribute value

// RUN: not mlir-opt %S/invalid-ir_section-forwardref.mlirbc -allow-unregistered-dialect 2>&1 | FileCheck %s --check-prefix=FORWARD_REF
// FORWARD_REF: expected attribute value

//===--------------------------------------------------------------------===//
// Results

// RUN: not mlir-opt %S/invalid-ir_section-results.mlirbc -allow-unregistered-dialect 2>&1 | FileCheck %s --check-prefix=OP_RESULTS
// OP_RESULTS: expected attribute value

//===--------------------------------------------------------------------===//
// Successors

// RUN: not mlir-opt %S/invalid-ir_section-successors.mlirbc -allow-unregistered-dialect 2>&1 | FileCheck %s --check-prefix=OP_SUCCESSORS
// OP_SUCCESSORS: expected attribute value
