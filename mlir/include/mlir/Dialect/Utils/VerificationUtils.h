//===- VerificationUtils.h - Common verification utilities ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines common verification utilities that can be shared
// across multiple MLIR dialects. These utilities help reduce code duplication
// for common verification patterns such as checking dynamic dimensions,
// rank matching, and index validation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_UTILS_VERIFICATIONUTILS_H
#define MLIR_DIALECT_UTILS_VERIFICATIONUTILS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// Dynamic Dimension Verification
//===----------------------------------------------------------------------===//

/// Verify that the number of dynamic size operands matches the number of
/// dynamic dimensions in the shaped type. Returns failure and emits an error
/// if the counts don't match.
LogicalResult verifyDynamicDimensionCount(Operation *op, ShapedType type,
                                          ValueRange dynamicSizes);

//===----------------------------------------------------------------------===//
// Rank Verification
//===----------------------------------------------------------------------===//

/// Verify that two shaped types have matching ranks. Returns failure and emits
/// an error if ranks don't match.
LogicalResult verifyRanksMatch(Operation *op, ShapedType type1,
                               ShapedType type2, StringRef name1,
                               StringRef name2);

/// Verify that a shaped type has the expected rank. Returns failure and emits
/// an error if the rank doesn't match.
LogicalResult verifyRankEquals(Operation *op, ShapedType type,
                               int64_t expectedRank, StringRef typeName);

/// Verify that a shaped type's rank is within the specified range [minRank,
/// maxRank]. Returns failure and emits an error if out of range.
LogicalResult verifyRankInRange(Operation *op, ShapedType type, int64_t minRank,
                                int64_t maxRank, StringRef typeName);

//===----------------------------------------------------------------------===//
// Index/Dimension Verification
//===----------------------------------------------------------------------===//

/// Verify that the number of indices matches the rank of the shaped type.
/// Returns failure and emits an error if the counts don't match.
LogicalResult verifyIndexCountMatchesRank(Operation *op, int64_t rank,
                                          size_t indexCount,
                                          StringRef indexName = "indices");

/// Verify that all dimension indices in the array are within the valid range
/// [0, maxDim). Returns failure and emits an error if any index is out of
/// range.
LogicalResult verifyDimensionIndicesInRange(Operation *op,
                                            ArrayRef<int64_t> indices,
                                            int64_t maxDim,
                                            StringRef context = "dimensions");

/// Verify that all dimension indices are unique (no duplicates). Returns
/// failure and emits an error if duplicates are found.
LogicalResult verifyDimensionIndicesUnique(Operation *op,
                                           ArrayRef<int64_t> indices,
                                           StringRef context = "dimensions");

//===----------------------------------------------------------------------===//
// Shape Verification
//===----------------------------------------------------------------------===//

/// Verify that all values in the range have the same shape. Returns failure
/// and emits an error if shapes don't match.
LogicalResult verifyAllShapesMatch(Operation *op, ValueRange values,
                                   StringRef context);

/// Verify that two shaped types have compatible shapes (same rank and matching
/// dimensions, with dynamic dimensions considered compatible with any size).
/// Returns failure and emits an error if incompatible.
LogicalResult verifyShapesCompatible(Operation *op, ShapedType type1,
                                     ShapedType type2, StringRef name1,
                                     StringRef name2);

//===----------------------------------------------------------------------===//
// Element Type Verification
//===----------------------------------------------------------------------===//

/// Verify that all values in the range have the same element type. Returns
/// failure and emits an error if element types don't match.
LogicalResult verifyAllElementTypesMatch(Operation *op, ValueRange values,
                                         StringRef context);

/// Verify that two shaped types have the same element type. Returns failure
/// and emits an error if they don't match.
LogicalResult verifyElementTypesMatch(Operation *op, ShapedType type1,
                                      ShapedType type2, StringRef name1,
                                      StringRef name2);

//===----------------------------------------------------------------------===//
// Element Count Verification
//===----------------------------------------------------------------------===//

/// Verify that two shaped types have the same total number of elements.
/// Returns failure and emits an error if element counts don't match.
/// Useful for reshape and shape_cast operations.
LogicalResult verifyElementCountsMatch(Operation *op, ShapedType type1,
                                       ShapedType type2, StringRef name1,
                                       StringRef name2);

} // namespace mlir

#endif // MLIR_DIALECT_UTILS_VERIFICATIONUTILS_H
