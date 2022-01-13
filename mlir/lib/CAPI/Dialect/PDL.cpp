//===- PDL.cpp - C Interface for PDL dialect ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/PDL.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"

using namespace mlir;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(PDL, pdl, pdl::PDLDialect)

//===---------------------------------------------------------------------===//
// PDLType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAPDLType(MlirType type) {
  return unwrap(type).isa<pdl::PDLType>();
}

//===---------------------------------------------------------------------===//
// AttributeType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAPDLAttributeType(MlirType type) {
  return unwrap(type).isa<pdl::AttributeType>();
}

MlirType mlirPDLAttributeTypeGet(MlirContext ctx) {
  return wrap(pdl::AttributeType::get(unwrap(ctx)));
}

//===---------------------------------------------------------------------===//
// OperationType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAPDLOperationType(MlirType type) {
  return unwrap(type).isa<pdl::OperationType>();
}

MlirType mlirPDLOperationTypeGet(MlirContext ctx) {
  return wrap(pdl::OperationType::get(unwrap(ctx)));
}

//===---------------------------------------------------------------------===//
// RangeType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAPDLRangeType(MlirType type) {
  return unwrap(type).isa<pdl::RangeType>();
}

MlirType mlirPDLRangeTypeGet(MlirType elementType) {
  return wrap(pdl::RangeType::get(unwrap(elementType)));
}

MlirType mlirPDLRangeTypeGetElementType(MlirType type) {
  return wrap(unwrap(type).cast<pdl::RangeType>().getElementType());
}

//===---------------------------------------------------------------------===//
// TypeType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAPDLTypeType(MlirType type) {
  return unwrap(type).isa<pdl::TypeType>();
}

MlirType mlirPDLTypeTypeGet(MlirContext ctx) {
  return wrap(pdl::TypeType::get(unwrap(ctx)));
}

//===---------------------------------------------------------------------===//
// ValueType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAPDLValueType(MlirType type) {
  return unwrap(type).isa<pdl::ValueType>();
}

MlirType mlirPDLValueTypeGet(MlirContext ctx) {
  return wrap(pdl::ValueType::get(unwrap(ctx)));
}
