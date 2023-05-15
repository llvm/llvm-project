//===- MemRefUtils.cpp - Utilities to support the MemRef dialect ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for the MemRef dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir {
namespace memref {

bool isStaticShapeAndContiguousRowMajor(MemRefType type) {
  if (!type.hasStaticShape())
    return false;

  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(getStridesAndOffset(type, strides, offset)))
    return false;

  // MemRef is contiguous if outer dimensions are size-1 and inner
  // dimensions have unit strides.
  int64_t runningStride = 1;
  int64_t curDim = strides.size() - 1;
  // Finds all inner dimensions with unit strides.
  while (curDim >= 0 && strides[curDim] == runningStride) {
    runningStride *= type.getDimSize(curDim);
    --curDim;
  }

  // Check if other dimensions are size-1.
  while (curDim >= 0 && type.getDimSize(curDim) == 1) {
    --curDim;
  }

  // All dims are unit-strided or size-1.
  return curDim < 0;
}

} // namespace memref
} // namespace mlir
