//===- Analysis.cpp - C API for MLIR Analysis Utilities -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Analysis.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

using namespace mlir;

static SetVector<Operation *> computeForwardSlice(
    MlirOperation op, MlirSliceFilterCallback filter, void *filterUserData) {
  SetVector<Operation *> result;
  ForwardSliceOptions options;
  if (filter) {
    options.filter = [filter, filterUserData](Operation *op) {
      return filter(wrap(op), filterUserData);
    };
  }
  getForwardSlice(unwrap(op), &result, options);
  return result;
}

intptr_t mlirGetForwardSliceSize(MlirOperation op,
                                 MlirSliceFilterCallback filter,
                                 void *filterUserData) {
  return static_cast<intptr_t>(
      computeForwardSlice(op, filter, filterUserData).size());
}

void mlirGetForwardSlice(MlirOperation op, MlirSliceFilterCallback filter,
                         void *filterUserData, MlirOperation *slice) {
  SetVector<Operation *> result =
      computeForwardSlice(op, filter, filterUserData);
  for (intptr_t i = 0, e = static_cast<intptr_t>(result.size()); i < e; ++i)
    slice[i] = wrap(result[i]);
}

static LogicalResult computeBackwardSlice(MlirOperation op,
                                          MlirSliceFilterCallback filter,
                                          void *filterUserData,
                                          SetVector<Operation *> &result) {
  BackwardSliceOptions options;
  if (filter) {
    options.filter = [filter, filterUserData](Operation *op) {
      return filter(wrap(op), filterUserData);
    };
  }
  return getBackwardSlice(unwrap(op), &result, options);
}

intptr_t mlirGetBackwardSliceSize(MlirOperation op,
                                  MlirSliceFilterCallback filter,
                                  void *filterUserData) {
  SetVector<Operation *> result;
  if (failed(computeBackwardSlice(op, filter, filterUserData, result)))
    return -1;
  return static_cast<intptr_t>(result.size());
}

void mlirGetBackwardSlice(MlirOperation op, MlirSliceFilterCallback filter,
                          void *filterUserData, MlirOperation *slice) {
  SetVector<Operation *> result;
  (void)computeBackwardSlice(op, filter, filterUserData, result);
  for (intptr_t i = 0, e = static_cast<intptr_t>(result.size()); i < e; ++i)
    slice[i] = wrap(result[i]);
}

intptr_t mlirRegionGetBlocksSortedByDominanceSize(MlirRegion region) {
  return static_cast<intptr_t>(
      getBlocksSortedByDominance(*unwrap(region)).size());
}

void mlirRegionGetBlocksSortedByDominance(MlirRegion region,
                                          MlirBlock *blocks) {
  SetVector<Block *> sorted = getBlocksSortedByDominance(*unwrap(region));
  for (intptr_t i = 0, e = static_cast<intptr_t>(sorted.size()); i < e; ++i)
    blocks[i] = wrap(sorted[i]);
}

void mlirTopologicalSort(intptr_t nOps, MlirOperation *ops,
                         MlirOperation *sorted) {
  SetVector<Operation *> toSort;
  for (intptr_t i = 0; i < nOps; ++i)
    toSort.insert(unwrap(ops[i]));
  SetVector<Operation *> result = topologicalSort(toSort);
  for (intptr_t i = 0, e = static_cast<intptr_t>(result.size()); i < e; ++i)
    sorted[i] = wrap(result[i]);
}
