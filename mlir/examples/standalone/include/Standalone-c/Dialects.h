//===- Dialects.h - CAPI for dialects -----------------------------*- C -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_C_DIALECTS_H
#define STANDALONE_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Standalone, standalone);

MLIR_CAPI_EXPORTED MlirType mlirStandaloneCustomTypeGet(MlirContext ctx,
                                                        MlirStringRef value);

MLIR_CAPI_EXPORTED bool mlirStandaloneTypeIsACustomType(MlirType t);

MLIR_CAPI_EXPORTED MlirTypeID mlirStandaloneCustomTypeGetTypeID(void);

#ifdef __cplusplus
}
#endif

#endif // STANDALONE_C_DIALECTS_H
