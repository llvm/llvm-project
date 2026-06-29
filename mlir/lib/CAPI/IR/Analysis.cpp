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

#include <algorithm>

using namespace mlir;

/// Copies up to `n` operations from `slice` into the caller-provided `result`
/// buffer (wrapping each), returning the total number of operations.
static intptr_t copySlice(const SetVector<Operation *> &slice, intptr_t n,
                          MlirOperation *result) {
  intptr_t count = static_cast<intptr_t>(slice.size());
  for (intptr_t i = 0, e = std::min(count, n); i < e; ++i)
    result[i] = wrap(slice[i]);
  return count;
}

intptr_t mlirGetForwardSlice(MlirOperation op, MlirSliceFilterCallback filter,
                             void *filterUserData, intptr_t n,
                             MlirOperation *slice) {
  SetVector<Operation *> result;
  ForwardSliceOptions options;
  if (filter)
    options.filter = [filter, filterUserData](Operation *op) {
      return filter(wrap(op), filterUserData);
    };
  getForwardSlice(unwrap(op), &result, options);
  return copySlice(result, n, slice);
}
