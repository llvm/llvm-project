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
MLIR_CAPI_EXPORTED bool mlirSMTTypeIsAnyNonFuncSMTValueType(MlirType type);

/// Checks if the given type is any SMT value type.
MLIR_CAPI_EXPORTED bool mlirSMTTypeIsAnySMTValueType(MlirType type);

/// Checks if the given type is a smt::ArrayType.
MLIR_CAPI_EXPORTED bool mlirSMTTypeIsAArray(MlirType type);

/// Creates an array type with the given domain and range types.
MLIR_CAPI_EXPORTED MlirType mlirSMTTypeGetArray(MlirContext ctx,
                                                MlirType domainType,
                                                MlirType rangeType);

/// Checks if the given type is a smt::BitVectorType.
MLIR_CAPI_EXPORTED bool mlirSMTTypeIsABitVector(MlirType type);

/// Creates a smt::BitVectorType with the given width.
MLIR_CAPI_EXPORTED MlirType mlirSMTTypeGetBitVector(MlirContext ctx,
                                                    int32_t width);

/// Checks if the given type is a smt::BoolType.
MLIR_CAPI_EXPORTED bool mlirSMTTypeIsABool(MlirType type);

/// Creates a smt::BoolType.
MLIR_CAPI_EXPORTED MlirType mlirSMTTypeGetBool(MlirContext ctx);

/// Checks if the given type is a smt::IntType.
MLIR_CAPI_EXPORTED bool mlirSMTTypeIsAInt(MlirType type);

/// Creates a smt::IntType.
MLIR_CAPI_EXPORTED MlirType mlirSMTTypeGetInt(MlirContext ctx);

/// Checks if the given type is a smt::FuncType.
MLIR_CAPI_EXPORTED bool mlirSMTTypeIsASMTFunc(MlirType type);

/// Creates a smt::FuncType with the given domain and range types.
MLIR_CAPI_EXPORTED MlirType mlirSMTTypeGetSMTFunc(MlirContext ctx,
                                                  size_t numberOfDomainTypes,
                                                  const MlirType *domainTypes,
                                                  MlirType rangeType);

/// Checks if the given type is a smt::SortType.
MLIR_CAPI_EXPORTED bool mlirSMTTypeIsASort(MlirType type);

/// Creates a smt::SortType with the given identifier and sort parameters.
MLIR_CAPI_EXPORTED MlirType mlirSMTTypeGetSort(MlirContext ctx,
                                               MlirIdentifier identifier,
                                               size_t numberOfSortParams,
                                               const MlirType *sortParams);

//===----------------------------------------------------------------------===//
// Attribute API.
//===----------------------------------------------------------------------===//

/// Checks if the given string is a valid smt::BVCmpPredicate.
MLIR_CAPI_EXPORTED bool mlirSMTAttrCheckBVCmpPredicate(MlirContext ctx,
                                                       MlirStringRef str);

/// Checks if the given string is a valid smt::IntPredicate.
MLIR_CAPI_EXPORTED bool mlirSMTAttrCheckIntPredicate(MlirContext ctx,
                                                     MlirStringRef str);

/// Checks if the given attribute is a smt::SMTAttribute.
MLIR_CAPI_EXPORTED bool mlirSMTAttrIsASMTAttribute(MlirAttribute attr);

/// Creates a smt::BitVectorAttr with the given value and width.
MLIR_CAPI_EXPORTED MlirAttribute mlirSMTAttrGetBitVector(MlirContext ctx,
                                                         uint64_t value,
                                                         unsigned width);

/// Creates a smt::BVCmpPredicateAttr with the given string.
MLIR_CAPI_EXPORTED MlirAttribute
mlirSMTAttrGetBVCmpPredicate(MlirContext ctx, MlirStringRef str);

/// Creates a smt::IntPredicateAttr with the given string.
MLIR_CAPI_EXPORTED MlirAttribute mlirSMTAttrGetIntPredicate(MlirContext ctx,
                                                            MlirStringRef str);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIALECT_SMT_H
