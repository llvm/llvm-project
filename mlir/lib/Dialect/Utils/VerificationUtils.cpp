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

LogicalResult mlir::verifyRanksMatch(Operation *op, ShapedType type1,
                                     ShapedType type2, StringRef name1,
                                     StringRef name2) {
  if (!type1.hasRank() || !type2.hasRank())
    return success(); // Unranked types are considered compatible

  int64_t rank1 = type1.getRank();
  int64_t rank2 = type2.getRank();
  if (rank1 != rank2) {
    return op->emitOpError()
           << name1 << " rank (" << rank1 << ") does not match " << name2
           << " rank (" << rank2 << ")";
  }
  return success();
}
