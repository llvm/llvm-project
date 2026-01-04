//===- VerificationUtils.cpp - Common verification utilities --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Utils/VerificationUtils.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Dynamic Dimension Verification
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Rank Verification
//===----------------------------------------------------------------------===//

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

LogicalResult mlir::verifyRankEquals(Operation *op, ShapedType type,
                                     int64_t expectedRank, StringRef typeName) {
  if (!type.hasRank())
    return success();

  int64_t actualRank = type.getRank();
  if (actualRank != expectedRank) {
    return op->emitOpError() << typeName << " must have rank " << expectedRank
                             << ", but has " << actualRank;
  }
  return success();
}

LogicalResult mlir::verifyRankInRange(Operation *op, ShapedType type,
                                      int64_t minRank, int64_t maxRank,
                                      StringRef typeName) {
  if (!type.hasRank())
    return success();

  int64_t rank = type.getRank();
  if (rank < minRank || rank > maxRank) {
    return op->emitOpError()
           << typeName << " rank must be in range [" << minRank << ", "
           << maxRank << "], but has rank " << rank;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Index/Dimension Verification
//===----------------------------------------------------------------------===//

LogicalResult mlir::verifyIndexCountMatchesRank(Operation *op, int64_t rank,
                                                size_t indexCount,
                                                StringRef indexName) {
  if (rank != static_cast<int64_t>(indexCount)) {
    return op->emitOpError("incorrect number of ")
           << indexName << ", has " << indexCount << ", expected " << rank;
  }
  return success();
}

LogicalResult mlir::verifyDimensionIndicesInRange(Operation *op,
                                                  ArrayRef<int64_t> indices,
                                                  int64_t maxDim,
                                                  StringRef context) {
  for (int64_t index : indices) {
    if (index < 0 || index >= maxDim) {
      return op->emitOpError() << context << " must be in the range [0, "
                               << (maxDim - 1) << "], but got " << index;
    }
  }
  return success();
}

LogicalResult mlir::verifyDimensionIndicesUnique(Operation *op,
                                                 ArrayRef<int64_t> indices,
                                                 StringRef context) {
  llvm::DenseSet<int64_t> seen;
  for (int64_t index : indices) {
    if (!seen.insert(index).second) {
      return op->emitOpError()
             << context << " contains duplicate index " << index;
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Shape Verification
//===----------------------------------------------------------------------===//

LogicalResult mlir::verifyAllShapesMatch(Operation *op, ValueRange values,
                                         StringRef context) {
  if (values.empty())
    return success();

  auto firstType = llvm::dyn_cast<ShapedType>(values.front().getType());
  if (!firstType || !firstType.hasRank())
    return success();

  ArrayRef<int64_t> firstShape = firstType.getShape();
  for (auto [idx, value] : llvm::enumerate(values.drop_front())) {
    auto type = llvm::dyn_cast<ShapedType>(value.getType());
    if (!type || !type.hasRank())
      continue;

    if (type.getShape() != firstShape) {
      return op->emitOpError()
             << context << " must all have the same shape, but " << context
             << "[0] has shape [" << firstShape << "] while " << context << "["
             << (idx + 1) << "] has shape [" << type.getShape() << "]";
    }
  }
  return success();
}

LogicalResult mlir::verifyShapesCompatible(Operation *op, ShapedType type1,
                                           ShapedType type2, StringRef name1,
                                           StringRef name2) {
  if (!type1.hasRank() || !type2.hasRank())
    return success();

  if (type1.getRank() != type2.getRank()) {
    return op->emitOpError()
           << name1 << " and " << name2 << " must have the same rank";
  }

  ArrayRef<int64_t> shape1 = type1.getShape();
  ArrayRef<int64_t> shape2 = type2.getShape();
  for (auto [idx, dims] : llvm::enumerate(llvm::zip(shape1, shape2))) {
    auto [dim1, dim2] = dims;
    // Dynamic dimensions are compatible with anything
    if (ShapedType::isDynamic(dim1) || ShapedType::isDynamic(dim2))
      continue;
    if (dim1 != dim2) {
      return op->emitOpError()
             << name1 << " and " << name2 << " have incompatible shapes at "
             << "dimension " << idx << ": " << dim1 << " vs " << dim2;
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Element Type Verification
//===----------------------------------------------------------------------===//

LogicalResult mlir::verifyAllElementTypesMatch(Operation *op, ValueRange values,
                                               StringRef context) {
  if (values.empty())
    return success();

  auto firstType = llvm::dyn_cast<ShapedType>(values.front().getType());
  if (!firstType)
    return success();

  Type firstElementType = firstType.getElementType();
  for (auto [idx, value] : llvm::enumerate(values.drop_front())) {
    auto type = llvm::dyn_cast<ShapedType>(value.getType());
    if (!type)
      continue;

    if (type.getElementType() != firstElementType) {
      return op->emitOpError()
             << context << " must all have the same element type, but "
             << context << "[0] has element type " << firstElementType
             << " while " << context << "[" << (idx + 1)
             << "] has element type " << type.getElementType();
    }
  }
  return success();
}

LogicalResult mlir::verifyElementTypesMatch(Operation *op, ShapedType type1,
                                            ShapedType type2, StringRef name1,
                                            StringRef name2) {
  if (type1.getElementType() != type2.getElementType()) {
    return op->emitOpError()
           << name1 << " element type (" << type1.getElementType()
           << ") does not match " << name2 << " element type ("
           << type2.getElementType() << ")";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Element Count Verification
//===----------------------------------------------------------------------===//

LogicalResult mlir::verifyElementCountsMatch(Operation *op, ShapedType type1,
                                             ShapedType type2, StringRef name1,
                                             StringRef name2) {
  if (!type1.hasStaticShape() || !type2.hasStaticShape())
    return success(); // Can't verify dynamic shapes at compile time

  int64_t count1 = type1.getNumElements();
  int64_t count2 = type2.getNumElements();
  if (count1 != count2) {
    return op->emitOpError() << name1 << " has " << count1 << " elements, but "
                             << name2 << " has " << count2 << " elements";
  }
  return success();
}
