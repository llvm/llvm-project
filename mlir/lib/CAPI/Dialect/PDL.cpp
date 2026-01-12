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
  return isa<pdl::PDLType>(unwrap(type));
}

//===---------------------------------------------------------------------===//
// AttributeType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAPDLAttributeType(MlirType type) {
  return isa<pdl::AttributeType>(unwrap(type));
}

MlirTypeID mlirPDLAttributeTypeGetTypeID(void) {
  return wrap(pdl::AttributeType::getTypeID());
}

MlirType mlirPDLAttributeTypeGet(MlirContext ctx) {
  return wrap(pdl::AttributeType::get(unwrap(ctx)));
}

MlirStringRef mlirPDLAttributeTypeGetName(void) {
  return wrap(pdl::AttributeType::name);
}

//===---------------------------------------------------------------------===//
// OperationType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAPDLOperationType(MlirType type) {
  return isa<pdl::OperationType>(unwrap(type));
}

MlirTypeID mlirPDLOperationTypeGetTypeID(void) {
  return wrap(pdl::OperationType::getTypeID());
}

MlirType mlirPDLOperationTypeGet(MlirContext ctx) {
  return wrap(pdl::OperationType::get(unwrap(ctx)));
}

MlirStringRef mlirPDLOperationTypeGetName(void) {
  return wrap(pdl::OperationType::name);
}

//===---------------------------------------------------------------------===//
// RangeType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAPDLRangeType(MlirType type) {
  return isa<pdl::RangeType>(unwrap(type));
}

MlirTypeID mlirPDLRangeTypeGetTypeID(void) {
  return wrap(pdl::RangeType::getTypeID());
}

MlirType mlirPDLRangeTypeGet(MlirType elementType) {
  return wrap(pdl::RangeType::get(unwrap(elementType)));
}

MlirStringRef mlirPDLRangeTypeGetName(void) {
  return wrap(pdl::RangeType::name);
}

MlirType mlirPDLRangeTypeGetElementType(MlirType type) {
  return wrap(cast<pdl::RangeType>(unwrap(type)).getElementType());
}

//===---------------------------------------------------------------------===//
// TypeType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAPDLTypeType(MlirType type) {
  return isa<pdl::TypeType>(unwrap(type));
}

MlirTypeID mlirPDLTypeTypeGetTypeID(void) {
  return wrap(pdl::TypeType::getTypeID());
}

MlirType mlirPDLTypeTypeGet(MlirContext ctx) {
  return wrap(pdl::TypeType::get(unwrap(ctx)));
}

MlirStringRef mlirPDLTypeTypeGetName(void) { return wrap(pdl::TypeType::name); }

//===---------------------------------------------------------------------===//
// ValueType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAPDLValueType(MlirType type) {
  return isa<pdl::ValueType>(unwrap(type));
}

MlirTypeID mlirPDLValueTypeGetTypeID(void) {
  return wrap(pdl::ValueType::getTypeID());
}

MlirType mlirPDLValueTypeGet(MlirContext ctx) {
  return wrap(pdl::ValueType::get(unwrap(ctx)));
}

MlirStringRef mlirPDLValueTypeGetName(void) {
  return wrap(pdl::ValueType::name);
}
