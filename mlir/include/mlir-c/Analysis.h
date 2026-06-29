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

/// Returns the number of operations in the forward slice of the given
/// operation, i.e. the number of entries mlirGetForwardSlice will write. Use
/// this to allocate the buffer passed to mlirGetForwardSlice. `filter` may be
/// NULL to traverse all operations; otherwise it acts as a frontier (see
/// MlirSliceFilterCallback).
MLIR_CAPI_EXPORTED intptr_t mlirGetForwardSliceSize(
    MlirOperation op, MlirSliceFilterCallback filter, void *filterUserData);

/// Computes the forward slice of the given operation, i.e. all its transitive
/// users, not including the operation itself. The result operations are written
/// (in slice order) into the caller-allocated `slice` buffer, which must have
/// room for mlirGetForwardSliceSize() entries. `filter` must match the one
/// passed to mlirGetForwardSliceSize and may be NULL to traverse all
/// operations; otherwise it acts as a frontier (see MlirSliceFilterCallback).
MLIR_CAPI_EXPORTED void mlirGetForwardSlice(MlirOperation op,
                                            MlirSliceFilterCallback filter,
                                            void *filterUserData,
                                            MlirOperation *slice);

/// Returns the number of operations in the backward slice of the given
/// operation, i.e. the number of entries mlirGetBackwardSlice will write, or a
/// negative value if the backward slice could not be computed. Use this to
/// allocate the buffer passed to mlirGetBackwardSlice. `filter` may be NULL to
/// traverse all operations; otherwise it acts as a frontier (see
/// MlirSliceFilterCallback).
MLIR_CAPI_EXPORTED intptr_t mlirGetBackwardSliceSize(
    MlirOperation op, MlirSliceFilterCallback filter, void *filterUserData);

/// Computes the backward slice of the given operation, i.e. all its transitive
/// definitions, not including the operation itself. The result operations are
/// written (in slice order) into the caller-allocated `slice` buffer, which
/// must have room for mlirGetBackwardSliceSize() entries. `filter` must match
/// the one passed to mlirGetBackwardSliceSize and may be NULL to traverse all
/// operations; otherwise it acts as a frontier (see MlirSliceFilterCallback).
MLIR_CAPI_EXPORTED void mlirGetBackwardSlice(MlirOperation op,
                                             MlirSliceFilterCallback filter,
                                             void *filterUserData,
                                             MlirOperation *slice);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_ANALYSIS_H
