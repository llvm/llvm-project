//===- EmitC.cpp - C Interface for EmitC dialect --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/EmitC.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"

using namespace mlir;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(EmitC, emitc, mlir::emitc::EmitCDialect)

// Ensure the C-API enums are uint64_t-castable to C++ equivalents.
static_assert(static_cast<uint64_t>(MLIR_EMITC_CMP_PREDICATE_EQ) ==
                      static_cast<uint64_t>(emitc::CmpPredicate::eq) &&
                  static_cast<uint64_t>(MLIR_EMITC_CMP_PREDICATE_NE) ==
                      static_cast<uint64_t>(emitc::CmpPredicate::ne) &&
                  static_cast<uint64_t>(MLIR_EMITC_CMP_PREDICATE_LT) ==
                      static_cast<uint64_t>(emitc::CmpPredicate::lt) &&
                  static_cast<uint64_t>(MLIR_EMITC_CMP_PREDICATE_LE) ==
                      static_cast<uint64_t>(emitc::CmpPredicate::le) &&
                  static_cast<uint64_t>(MLIR_EMITC_CMP_PREDICATE_GT) ==
                      static_cast<uint64_t>(emitc::CmpPredicate::gt) &&
                  static_cast<uint64_t>(MLIR_EMITC_CMP_PREDICATE_GE) ==
                      static_cast<uint64_t>(emitc::CmpPredicate::ge) &&
                  static_cast<uint64_t>(MLIR_EMITC_CMP_PREDICATE_THREE_WAY) ==
                      static_cast<uint64_t>(emitc::CmpPredicate::three_way),
              "MlirEmitCCmpPredicate (C-API) and CmpPredicate (C++) mismatch");

//===---------------------------------------------------------------------===//
// ArrayType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAEmitCArrayType(MlirType type) {
  return isa<emitc::ArrayType>(unwrap(type));
}

MlirTypeID mlirEmitCArrayTypeGetTypeID(void) {
  return wrap(emitc::ArrayType::getTypeID());
}

MlirType mlirEmitCArrayTypeGet(intptr_t nDims, int64_t *shape,
                               MlirType elementType) {
  return wrap(
      emitc::ArrayType::get(llvm::ArrayRef(shape, nDims), unwrap(elementType)));
}

//===---------------------------------------------------------------------===//
// LValueType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAEmitCLValueType(MlirType type) {
  return isa<emitc::LValueType>(unwrap(type));
}

MlirTypeID mlirEmitCLValueTypeGetTypeID(void) {
  return wrap(emitc::LValueType::getTypeID());
}

MlirType mlirEmitCLValueTypeGet(MlirType valueType) {
  return wrap(emitc::LValueType::get(unwrap(valueType)));
}

//===---------------------------------------------------------------------===//
// OpaqueType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAEmitCOpaqueType(MlirType type) {
  return isa<emitc::OpaqueType>(unwrap(type));
}

MlirTypeID mlirEmitCOpaqueTypeGetTypeID(void) {
  return wrap(emitc::OpaqueType::getTypeID());
}

MlirType mlirEmitCOpaqueTypeGet(MlirContext ctx, MlirStringRef value) {
  return wrap(emitc::OpaqueType::get(unwrap(ctx), unwrap(value)));
}

//===---------------------------------------------------------------------===//
// PointerType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAEmitCPointerType(MlirType type) {
  return isa<emitc::PointerType>(unwrap(type));
}

MlirTypeID mlirEmitCPointerTypeGetTypeID(void) {
  return wrap(emitc::PointerType::getTypeID());
}

MlirType mlirEmitCPointerTypeGet(MlirType pointee) {
  return wrap(emitc::PointerType::get(unwrap(pointee)));
}

//===---------------------------------------------------------------------===//
// PtrDiffTType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAEmitCPtrDiffTType(MlirType type) {
  return isa<emitc::PtrDiffTType>(unwrap(type));
}

MlirTypeID mlirEmitCPtrDiffTTypeGetTypeID(void) {
  return wrap(emitc::PtrDiffTType::getTypeID());
}

MlirType mlirEmitCPtrDiffTTypeGet(MlirContext ctx) {
  return wrap(emitc::PtrDiffTType::get(unwrap(ctx)));
}

//===---------------------------------------------------------------------===//
// SignedSizeTType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAEmitCSignedSizeTType(MlirType type) {
  return isa<emitc::SignedSizeTType>(unwrap(type));
}

MlirTypeID mlirEmitCSignedSizeTTypeGetTypeID(void) {
  return wrap(emitc::SignedSizeTType::getTypeID());
}

MlirType mlirEmitCSignedSizeTTypeGet(MlirContext ctx) {
  return wrap(emitc::SignedSizeTType::get(unwrap(ctx)));
}

//===---------------------------------------------------------------------===//
// SizeTType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAEmitCSizeTType(MlirType type) {
  return isa<emitc::SizeTType>(unwrap(type));
}

MlirTypeID mlirEmitCSizeTTypeGetTypeID(void) {
  return wrap(emitc::SizeTType::getTypeID());
}

MlirType mlirEmitCSizeTTypeGet(MlirContext ctx) {
  return wrap(emitc::SizeTType::get(unwrap(ctx)));
}

//===----------------------------------------------------------------------===//
// CmpPredicate attribute.
//===----------------------------------------------------------------------===//

bool mlirAttributeIsAEmitCCmpPredicate(MlirAttribute attr) {
  return llvm::isa<emitc::CmpPredicateAttr>(unwrap(attr));
}

MlirAttribute mlirEmitCCmpPredicateAttrGet(MlirContext ctx,
                                           MlirEmitCCmpPredicate val) {
  return wrap((Attribute)emitc::CmpPredicateAttr::get(
      unwrap(ctx), static_cast<emitc::CmpPredicate>(val)));
}

MlirEmitCCmpPredicate mlirEmitCCmpPredicateAttrGetValue(MlirAttribute attr) {
  return static_cast<MlirEmitCCmpPredicate>(
      llvm::cast<emitc::CmpPredicateAttr>(unwrap(attr)).getValue());
}

MlirTypeID mlirEmitCCmpPredicateAttrGetTypeID(void) {
  return wrap(emitc::CmpPredicateAttr::getTypeID());
}

//===----------------------------------------------------------------------===//
// Opaque attribute.
//===----------------------------------------------------------------------===//

bool mlirAttributeIsAEmitCOpaque(MlirAttribute attr) {
  return llvm::isa<emitc::OpaqueAttr>(unwrap(attr));
}

MlirAttribute mlirEmitCOpaqueAttrGet(MlirContext ctx, MlirStringRef value) {
  return wrap((Attribute)emitc::OpaqueAttr::get(unwrap(ctx), unwrap(value)));
}

MlirStringRef mlirEmitCOpaqueAttrGetValue(MlirAttribute attr) {
  return wrap(llvm::cast<emitc::OpaqueAttr>(unwrap(attr)).getValue());
}

MlirTypeID mlirEmitCOpaqueAttrGetTypeID(void) {
  return wrap(emitc::OpaqueAttr::getTypeID());
}
