//===- Dominance.h - C API for Dominance Analysis -----------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_DOMINANCE_H
#define MLIR_C_DOMINANCE_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(MlirDominanceInfo, void);
DEFINE_C_API_STRUCT(MlirPostDominanceInfo, void);

#undef DEFINE_C_API_STRUCT

//===----------------------------------------------------------------------===//
// DominanceInfo API
//===----------------------------------------------------------------------===//

/// Creates a DominanceInfo for the given operation (typically a FuncOp or
/// ModuleOp). The caller owns the returned object and must destroy it.
MLIR_CAPI_EXPORTED MlirDominanceInfo mlirDominanceInfoCreate(MlirOperation op);

/// Destroys the given DominanceInfo.
MLIR_CAPI_EXPORTED void mlirDominanceInfoDestroy(MlirDominanceInfo info);

/// Returns true if operation A properly dominates operation B.
MLIR_CAPI_EXPORTED bool
mlirDominanceInfoProperlyDominatesOperation(MlirDominanceInfo info,
                                            MlirOperation a, MlirOperation b);

/// Returns true if operation A dominates operation B (A == B or A properly
/// dominates B).
MLIR_CAPI_EXPORTED bool
mlirDominanceInfoDominatesOperation(MlirDominanceInfo info, MlirOperation a,
                                    MlirOperation b);

/// Returns true if value A properly dominates operation B.
MLIR_CAPI_EXPORTED bool
mlirDominanceInfoValueProperlyDominates(MlirDominanceInfo info, MlirValue a,
                                        MlirOperation b);

/// Returns true if value A dominates operation B (the operation defining A is B
/// or A properly dominates B).
MLIR_CAPI_EXPORTED bool mlirDominanceInfoValueDominates(MlirDominanceInfo info,
                                                        MlirValue a,
                                                        MlirOperation b);

/// Returns true if block A properly dominates block B.
MLIR_CAPI_EXPORTED bool
mlirDominanceInfoProperlyDominatesBlock(MlirDominanceInfo info, MlirBlock a,
                                        MlirBlock b);

/// Returns true if block A dominates block B.
MLIR_CAPI_EXPORTED bool mlirDominanceInfoDominatesBlock(MlirDominanceInfo info,
                                                        MlirBlock a,
                                                        MlirBlock b);

/// Finds the nearest common dominator of blocks A and B. Returns a null block
/// if none exists.
MLIR_CAPI_EXPORTED MlirBlock mlirDominanceInfoFindNearestCommonDominator(
    MlirDominanceInfo info, MlirBlock a, MlirBlock b);

/// Returns true if the given block is reachable from the entry block of its
/// region.
MLIR_CAPI_EXPORTED bool
mlirDominanceInfoIsReachableFromEntry(MlirDominanceInfo info, MlirBlock block);

/// Invalidates all cached dominance information.
MLIR_CAPI_EXPORTED void mlirDominanceInfoInvalidate(MlirDominanceInfo info);

//===----------------------------------------------------------------------===//
// PostDominanceInfo API
//===----------------------------------------------------------------------===//

/// Creates a PostDominanceInfo for the given operation.
MLIR_CAPI_EXPORTED MlirPostDominanceInfo
mlirPostDominanceInfoCreate(MlirOperation op);

/// Destroys the given PostDominanceInfo.
MLIR_CAPI_EXPORTED void
mlirPostDominanceInfoDestroy(MlirPostDominanceInfo info);

/// Returns true if operation A properly post-dominates operation B.
MLIR_CAPI_EXPORTED bool mlirPostDominanceInfoProperlyPostDominatesOperation(
    MlirPostDominanceInfo info, MlirOperation a, MlirOperation b);

/// Returns true if operation A post-dominates operation B.
MLIR_CAPI_EXPORTED bool
mlirPostDominanceInfoPostDominatesOperation(MlirPostDominanceInfo info,
                                            MlirOperation a, MlirOperation b);

/// Returns true if block A properly post-dominates block B.
MLIR_CAPI_EXPORTED bool
mlirPostDominanceInfoProperlyPostDominatesBlock(MlirPostDominanceInfo info,
                                                MlirBlock a, MlirBlock b);

/// Returns true if block A post-dominates block B.
MLIR_CAPI_EXPORTED bool
mlirPostDominanceInfoPostDominatesBlock(MlirPostDominanceInfo info, MlirBlock a,
                                        MlirBlock b);

/// Invalidates all cached post-dominance information.
MLIR_CAPI_EXPORTED void
mlirPostDominanceInfoInvalidate(MlirPostDominanceInfo info);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DOMINANCE_H
