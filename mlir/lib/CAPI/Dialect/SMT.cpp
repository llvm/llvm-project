//===- SMT.cpp - C interface for the SMT dialect --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/SMT.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/SMT/IR/SMTAttributes.h"
#include "mlir/Dialect/SMT/IR/SMTDialect.h"
#include "mlir/Dialect/SMT/IR/SMTTypes.h"

using namespace mlir;
using namespace smt;

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(SMT, smt, mlir::smt::SMTDialect)

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

bool mlirSMTTypeIsAnyNonFuncSMTValueType(MlirType type) {
  return isAnyNonFuncSMTValueType(unwrap(type));
}

bool mlirSMTTypeIsAnySMTValueType(MlirType type) {
  return isAnySMTValueType(unwrap(type));
}

bool mlirSMTTypeIsAArray(MlirType type) { return isa<ArrayType>(unwrap(type)); }

MlirType mlirSMTTypeGetArray(MlirContext ctx, MlirType domainType,
                             MlirType rangeType) {
  return wrap(
      ArrayType::get(unwrap(ctx), unwrap(domainType), unwrap(rangeType)));
}

bool mlirSMTTypeIsABitVector(MlirType type) {
  return isa<BitVectorType>(unwrap(type));
}

MlirType mlirSMTTypeGetBitVector(MlirContext ctx, int32_t width) {
  return wrap(BitVectorType::get(unwrap(ctx), width));
}

bool mlirSMTTypeIsABool(MlirType type) { return isa<BoolType>(unwrap(type)); }

MlirType mlirSMTTypeGetBool(MlirContext ctx) {
  return wrap(BoolType::get(unwrap(ctx)));
}

bool mlirSMTTypeIsAInt(MlirType type) { return isa<IntType>(unwrap(type)); }

MlirType mlirSMTTypeGetInt(MlirContext ctx) {
  return wrap(IntType::get(unwrap(ctx)));
}

bool mlirSMTTypeIsASMTFunc(MlirType type) {
  return isa<SMTFuncType>(unwrap(type));
}

MlirType mlirSMTTypeGetSMTFunc(MlirContext ctx, size_t numberOfDomainTypes,
                               const MlirType *domainTypes,
                               MlirType rangeType) {
  SmallVector<Type> domainTypesVec;
  domainTypesVec.reserve(numberOfDomainTypes);

  for (size_t i = 0; i < numberOfDomainTypes; i++)
    domainTypesVec.push_back(unwrap(domainTypes[i]));

  return wrap(SMTFuncType::get(unwrap(ctx), domainTypesVec, unwrap(rangeType)));
}

bool mlirSMTTypeIsASort(MlirType type) { return isa<SortType>(unwrap(type)); }

MlirType mlirSMTTypeGetSort(MlirContext ctx, MlirIdentifier identifier,
                            size_t numberOfSortParams,
                            const MlirType *sortParams) {
  SmallVector<Type> sortParamsVec;
  sortParamsVec.reserve(numberOfSortParams);

  for (size_t i = 0; i < numberOfSortParams; i++)
    sortParamsVec.push_back(unwrap(sortParams[i]));

  return wrap(SortType::get(unwrap(ctx), unwrap(identifier), sortParamsVec));
}

//===----------------------------------------------------------------------===//
// Attribute API.
//===----------------------------------------------------------------------===//

bool mlirSMTAttrCheckBVCmpPredicate(MlirContext ctx, MlirStringRef str) {
  return symbolizeBVCmpPredicate(unwrap(str)).has_value();
}

bool mlirSMTAttrCheckIntPredicate(MlirContext ctx, MlirStringRef str) {
  return symbolizeIntPredicate(unwrap(str)).has_value();
}

bool mlirSMTAttrIsASMTAttribute(MlirAttribute attr) {
  return isa<BitVectorAttr, BVCmpPredicateAttr, IntPredicateAttr>(unwrap(attr));
}

MlirAttribute mlirSMTAttrGetBitVector(MlirContext ctx, uint64_t value,
                                      unsigned width) {
  return wrap(BitVectorAttr::get(unwrap(ctx), value, width));
}

MlirAttribute mlirSMTAttrGetBVCmpPredicate(MlirContext ctx, MlirStringRef str) {
  auto predicate = symbolizeBVCmpPredicate(unwrap(str));
  assert(predicate.has_value() && "invalid predicate");

  return wrap(BVCmpPredicateAttr::get(unwrap(ctx), predicate.value()));
}

MlirAttribute mlirSMTAttrGetIntPredicate(MlirContext ctx, MlirStringRef str) {
  auto predicate = symbolizeIntPredicate(unwrap(str));
  assert(predicate.has_value() && "invalid predicate");

  return wrap(IntPredicateAttr::get(unwrap(ctx), predicate.value()));
}
