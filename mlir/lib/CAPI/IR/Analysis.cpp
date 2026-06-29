//===- Analysis.cpp - C API for MLIR Analysis Utilities -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Analysis.h"

#include "mlir/Analysis/SliceAnalysis.h"
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
