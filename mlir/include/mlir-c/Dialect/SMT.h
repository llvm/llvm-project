//===- SMT.h - C interface for the SMT dialect --------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_DIALECT_SMT_H
#define MLIR_C_DIALECT_SMT_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(SMT, smt);

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

/// Checks if the given type is any non-func SMT value type.
MLIR_CAPI_EXPORTED bool smtTypeIsAnyNonFuncSMTValueType(MlirType type);

/// Checks if the given type is any SMT value type.
MLIR_CAPI_EXPORTED bool smtTypeIsAnySMTValueType(MlirType type);

/// Checks if the given type is a smt::ArrayType.
MLIR_CAPI_EXPORTED bool smtTypeIsAArray(MlirType type);

/// Creates an array type with the given domain and range types.
MLIR_CAPI_EXPORTED MlirType smtTypeGetArray(MlirContext ctx,
                                            MlirType domainType,
                                            MlirType rangeType);

/// Checks if the given type is a smt::BitVectorType.
MLIR_CAPI_EXPORTED bool smtTypeIsABitVector(MlirType type);

/// Creates a smt::BitVectorType with the given width.
MLIR_CAPI_EXPORTED MlirType smtTypeGetBitVector(MlirContext ctx, int32_t width);

/// Checks if the given type is a smt::BoolType.
MLIR_CAPI_EXPORTED bool smtTypeIsABool(MlirType type);

/// Creates a smt::BoolType.
MLIR_CAPI_EXPORTED MlirType smtTypeGetBool(MlirContext ctx);

/// Checks if the given type is a smt::IntType.
MLIR_CAPI_EXPORTED bool smtTypeIsAInt(MlirType type);

/// Creates a smt::IntType.
MLIR_CAPI_EXPORTED MlirType smtTypeGetInt(MlirContext ctx);

/// Checks if the given type is a smt::FuncType.
MLIR_CAPI_EXPORTED bool smtTypeIsASMTFunc(MlirType type);

/// Creates a smt::FuncType with the given domain and range types.
MLIR_CAPI_EXPORTED MlirType smtTypeGetSMTFunc(MlirContext ctx,
                                              size_t numberOfDomainTypes,
                                              const MlirType *domainTypes,
                                              MlirType rangeType);

/// Checks if the given type is a smt::SortType.
MLIR_CAPI_EXPORTED bool smtTypeIsASort(MlirType type);

/// Creates a smt::SortType with the given identifier and sort parameters.
MLIR_CAPI_EXPORTED MlirType smtTypeGetSort(MlirContext ctx,
                                           MlirIdentifier identifier,
                                           size_t numberOfSortParams,
                                           const MlirType *sortParams);

//===----------------------------------------------------------------------===//
// Attribute API.
//===----------------------------------------------------------------------===//

/// Checks if the given string is a valid smt::BVCmpPredicate.
MLIR_CAPI_EXPORTED bool smtAttrCheckBVCmpPredicate(MlirContext ctx,
                                                   MlirStringRef str);

/// Checks if the given string is a valid smt::IntPredicate.
MLIR_CAPI_EXPORTED bool smtAttrCheckIntPredicate(MlirContext ctx,
                                                 MlirStringRef str);

/// Checks if the given attribute is a smt::SMTAttribute.
MLIR_CAPI_EXPORTED bool smtAttrIsASMTAttribute(MlirAttribute attr);

/// Creates a smt::BitVectorAttr with the given value and width.
MLIR_CAPI_EXPORTED MlirAttribute smtAttrGetBitVector(MlirContext ctx,
                                                     uint64_t value,
                                                     unsigned width);

/// Creates a smt::BVCmpPredicateAttr with the given string.
MLIR_CAPI_EXPORTED MlirAttribute smtAttrGetBVCmpPredicate(MlirContext ctx,
                                                          MlirStringRef str);

/// Creates a smt::IntPredicateAttr with the given string.
MLIR_CAPI_EXPORTED MlirAttribute smtAttrGetIntPredicate(MlirContext ctx,
                                                        MlirStringRef str);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIALECT_SMT_H
