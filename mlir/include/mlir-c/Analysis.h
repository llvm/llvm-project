//===- Analysis.h - C API for MLIR Analysis Utilities -------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_ANALYSIS_H
#define MLIR_C_ANALYSIS_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Slice analysis
//===----------------------------------------------------------------------===//

/// Filter callback for slice analysis, corresponding to
/// `mlir::SliceOptions::filter`. Return true to keep traversing through the
/// given operation, false to treat it as a frontier and stop propagation.
typedef bool (*MlirSliceFilterCallback)(MlirOperation op, void *userData);

/// Computes the forward slice of the given operation, i.e. all its transitive
/// users, not including the operation itself. The result operations are written
/// (in slice order) into the caller-allocated `slice` buffer, up to `n`
/// entries; the total number of operations in the slice is returned (which may
/// exceed `n`). Passing `n == 0` (with `slice` ignored) queries the size.
/// `filter` may be NULL to traverse all operations; otherwise it acts as a
/// frontier (see MlirSliceFilterCallback).
MLIR_CAPI_EXPORTED intptr_t mlirGetForwardSlice(MlirOperation op,
                                                MlirSliceFilterCallback filter,
                                                void *filterUserData,
                                                intptr_t n,
                                                MlirOperation *slice);

/// Computes the backward slice of the given operation, i.e. all its transitive
/// definitions, not including the operation itself. The result operations are
/// written (in slice order) into the caller-allocated `slice` buffer, up to `n`
/// entries; the total number of operations in the slice is returned. A negative
/// return value indicates the backward slice could not be computed. Passing
/// `n == 0` (with `slice` ignored) queries the size. `filter` may be NULL to
/// traverse all operations; otherwise it acts as a frontier (see
/// MlirSliceFilterCallback).
MLIR_CAPI_EXPORTED intptr_t mlirGetBackwardSlice(MlirOperation op,
                                                 MlirSliceFilterCallback filter,
                                                 void *filterUserData,
                                                 intptr_t n,
                                                 MlirOperation *slice);

//===----------------------------------------------------------------------===//
// Topological sort
//===----------------------------------------------------------------------===//

/// Returns the blocks of the given region sorted by dominance, a stable order
/// in which a block appears after all blocks that dominate it. The result
/// blocks are written into the caller-allocated `blocks` buffer, up to `n`
/// entries; the total number of blocks in the region is returned. Passing
/// `n == 0` (with `blocks` ignored) queries the size.
MLIR_CAPI_EXPORTED intptr_t mlirRegionGetBlocksSortedByDominance(
    MlirRegion region, intptr_t n, MlirBlock *blocks);

/// Topologically sorts the `nOps` operations in `ops` (taking region semantics
/// into account) so that definitions come before uses, writing the result into
/// the caller-allocated `sorted` buffer, which must have room for `nOps`
/// entries. The input operations need not all belong to the same block.
MLIR_CAPI_EXPORTED void mlirTopologicalSort(intptr_t nOps, MlirOperation *ops,
                                            MlirOperation *sorted);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_ANALYSIS_H
