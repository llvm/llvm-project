// This file contains various failure test cases related to the structure of
// the IR section.

// Bytecode currently does not support big-endian platforms
// UNSUPPORTED: target=s390x-{{.*}}

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
// OP_ATTR: expected attribute of type: {{.*}}, but got: loc(unknown)

//===--------------------------------------------------------------------===//
// Operands

// RUN: not mlir-opt %S/invalid-ir_section-operands.mlirbc -allow-unregistered-dialect 2>&1 | FileCheck %s --check-prefix=OP_OPERANDS
// OP_OPERANDS: invalid value index: 6

// RUN: not mlir-opt %S/invalid-ir_section-forwardref.mlirbc -allow-unregistered-dialect 2>&1 | FileCheck %s --check-prefix=FORWARD_REF
// FORWARD_REF: not all forward unresolved forward operand references

//===--------------------------------------------------------------------===//
// Results

// RUN: not mlir-opt %S/invalid-ir_section-results.mlirbc -allow-unregistered-dialect 2>&1 | FileCheck %s --check-prefix=OP_RESULTS
// OP_RESULTS: value index range was outside of the expected range for the parent region, got [3, 6), but the maximum index was 2

//===--------------------------------------------------------------------===//
// Successors

// RUN: not mlir-opt %S/invalid-ir_section-successors.mlirbc -allow-unregistered-dialect 2>&1 | FileCheck %s --check-prefix=OP_SUCCESSORS
// OP_SUCCESSORS: invalid successor index: 3
