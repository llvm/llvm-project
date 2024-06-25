//===-- mlir-c/Rewrite.h - Helpers for C API to Rewrites ----------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the registration and creation method for
// rewrite patterns.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_REWRITE_H
#define MLIR_C_REWRITE_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Config/mlir-config.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
/// Opaque type declarations (see mlir-c/IR.h for more details).
//===----------------------------------------------------------------------===//

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(MlirFrozenRewritePatternSet, void);
DEFINE_C_API_STRUCT(MlirGreedyRewriteDriverConfig, void);
DEFINE_C_API_STRUCT(MlirRewritePatternSet, void);

MLIR_CAPI_EXPORTED MlirFrozenRewritePatternSet
mlirFreezeRewritePattern(MlirRewritePatternSet op);

MLIR_CAPI_EXPORTED void
mlirFrozenRewritePatternSetDestroy(MlirFrozenRewritePatternSet op);

MLIR_CAPI_EXPORTED MlirLogicalResult mlirApplyPatternsAndFoldGreedily(
    MlirModule op, MlirFrozenRewritePatternSet patterns,
    MlirGreedyRewriteDriverConfig);

#if MLIR_ENABLE_PDL_IN_PATTERNMATCH
DEFINE_C_API_STRUCT(MlirPDLPatternModule, void);

MLIR_CAPI_EXPORTED MlirPDLPatternModule
mlirPDLPatternModuleFromModule(MlirModule op);

MLIR_CAPI_EXPORTED void mlirPDLPatternModuleDestroy(MlirPDLPatternModule op);

MLIR_CAPI_EXPORTED MlirRewritePatternSet
mlirRewritePatternSetFromPDLPatternModule(MlirPDLPatternModule op);
#endif // MLIR_ENABLE_PDL_IN_PATTERNMATCH

#undef DEFINE_C_API_STRUCT

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_REWRITE_H
