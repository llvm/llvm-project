//===- VerificationUtils.cpp - Common verification utilities --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Utils/VerificationUtils.h"

using namespace mlir;

LogicalResult mlir::verifyDynamicDimensionCount(Operation *op, ShapedType type,
                                                ValueRange dynamicSizes) {
  int64_t expectedCount = type.getNumDynamicDims();
  int64_t actualCount = dynamicSizes.size();
  if (expectedCount != actualCount) {
    return op->emitOpError("incorrect number of dynamic sizes, has ")
           << actualCount << ", expected " << expectedCount;
  }
  return success();
}
