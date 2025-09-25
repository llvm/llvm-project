//===-- mlir-c/Dialect/EmitC.h - C API for EmitC dialect ----------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_DIALECT_EmitC_H
#define MLIR_C_DIALECT_EmitC_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(EmitC, emitc);

enum MlirEmitCCmpPredicate : uint64_t {
  MLIR_EMITC_CMP_PREDICATE_EQ = 0,
  MLIR_EMITC_CMP_PREDICATE_NE = 1,
  MLIR_EMITC_CMP_PREDICATE_LT = 2,
  MLIR_EMITC_CMP_PREDICATE_LE = 3,
  MLIR_EMITC_CMP_PREDICATE_GT = 4,
  MLIR_EMITC_CMP_PREDICATE_GE = 5,
  MLIR_EMITC_CMP_PREDICATE_THREE_WAY = 6,
};

//===---------------------------------------------------------------------===//
// ArrayType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAEmitCArrayType(MlirType type);

MLIR_CAPI_EXPORTED MlirTypeID mlirEmitCArrayTypeGetTypeID(void);

MLIR_CAPI_EXPORTED MlirType mlirEmitCArrayTypeGet(intptr_t nDims,
                                                  int64_t *shape,
                                                  MlirType elementType);

//===---------------------------------------------------------------------===//
// LValueType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAEmitCLValueType(MlirType type);

MLIR_CAPI_EXPORTED MlirTypeID mlirEmitCLValueTypeGetTypeID(void);

MLIR_CAPI_EXPORTED MlirType mlirEmitCLValueTypeGet(MlirType valueType);

//===---------------------------------------------------------------------===//
// OpaqueType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAEmitCOpaqueType(MlirType type);

MLIR_CAPI_EXPORTED MlirTypeID mlirEmitCOpaqueTypeGetTypeID(void);

MLIR_CAPI_EXPORTED MlirType mlirEmitCOpaqueTypeGet(MlirContext ctx,
                                                   MlirStringRef value);

//===---------------------------------------------------------------------===//
// PointerType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAEmitCPointerType(MlirType type);

MLIR_CAPI_EXPORTED MlirTypeID mlirEmitCPointerTypeGetTypeID(void);

MLIR_CAPI_EXPORTED MlirType mlirEmitCPointerTypeGet(MlirType pointee);

//===---------------------------------------------------------------------===//
// PtrDiffTType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAEmitCPtrDiffTType(MlirType type);

MLIR_CAPI_EXPORTED MlirTypeID mlirEmitCPtrDiffTTypeGetTypeID(void);

MLIR_CAPI_EXPORTED MlirType mlirEmitCPtrDiffTTypeGet(MlirContext ctx);

//===---------------------------------------------------------------------===//
// SignedSizeTType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAEmitCSignedSizeTType(MlirType type);

MLIR_CAPI_EXPORTED MlirTypeID mlirEmitCSignedSizeTTypeGetTypeID(void);

MLIR_CAPI_EXPORTED MlirType mlirEmitCSignedSizeTTypeGet(MlirContext ctx);

//===---------------------------------------------------------------------===//
// SizeTType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAEmitCSizeTType(MlirType type);

MLIR_CAPI_EXPORTED MlirTypeID mlirEmitCSizeTTypeGetTypeID(void);

MLIR_CAPI_EXPORTED MlirType mlirEmitCSizeTTypeGet(MlirContext ctx);

//===----------------------------------------------------------------------===//
// CmpPredicate attribute.
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirAttributeIsAEmitCCmpPredicate(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute
mlirEmitCCmpPredicateAttrGet(MlirContext ctx, enum MlirEmitCCmpPredicate val);

MLIR_CAPI_EXPORTED enum MlirEmitCCmpPredicate
mlirEmitCCmpPredicateAttrGetValue(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirTypeID mlirEmitCCmpPredicateAttrGetTypeID(void);

//===----------------------------------------------------------------------===//
// Opaque attribute.
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirAttributeIsAEmitCOpaque(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute mlirEmitCOpaqueAttrGet(MlirContext ctx,
                                                        MlirStringRef value);

MLIR_CAPI_EXPORTED MlirStringRef
mlirEmitCOpaqueAttrGetValue(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirTypeID mlirEmitCOpaqueAttrGetTypeID(void);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIALECT_EmitC_H
