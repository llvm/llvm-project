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

LogicalResult mlir::verifyRanksMatch(Operation *op, ShapedType lhs,
                                     ShapedType rhs, StringRef lhsName,
                                     StringRef rhsName) {
  if (!lhs.hasRank() || !rhs.hasRank())
    return success(); // Unranked types are considered compatible

  int64_t rank1 = lhs.getRank();
  int64_t rank2 = rhs.getRank();
  if (rank1 != rank2) {
    return op->emitOpError()
           << lhsName << " rank (" << rank1 << ") does not match " << rhsName
           << " rank (" << rank2 << ")";
  }
  return success();
}

LogicalResult mlir::verifyElementTypesMatch(Operation *op, ShapedType lhs,
                                            ShapedType rhs, StringRef lhsName,
                                            StringRef rhsName) {
  Type lhsElementType = lhs.getElementType();
  Type rhsElementType = rhs.getElementType();
  if (lhsElementType != rhsElementType) {
    return op->emitOpError() << lhsName << " element type (" << lhsElementType
                             << ") does not match " << rhsName
                             << " element type (" << rhsElementType << ")";
  }
  return success();
}
