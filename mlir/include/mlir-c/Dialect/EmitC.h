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

//===---------------------------------------------------------------------===//
// ArrayType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAEmitCArrayType(MlirType type);

MLIR_CAPI_EXPORTED MlirTypeID mlirEmitCArrayTypeGetTypeID(void);

MLIR_CAPI_EXPORTED MlirType mlirEmitCArrayTypeGet(intptr_t nDims,
                                                  int64_t *shape,
                                                  MlirType elementType);
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

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIALECT_EmitC_H
