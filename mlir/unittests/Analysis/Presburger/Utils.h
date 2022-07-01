//===- Utils.h - Utils for Presburger Tests ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines helper functions for Presburger unittests.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_UNITTESTS_ANALYSIS_PRESBURGER_UTILS_H
#define MLIR_UNITTESTS_ANALYSIS_PRESBURGER_UTILS_H

#include "../../Dialect/Affine/Analysis/AffineStructuresParser.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/PWMAFunction.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"

#include <gtest/gtest.h>

namespace mlir {
namespace presburger {

/// Parses a IntegerPolyhedron from a StringRef. It is expected that the
/// string represents a valid IntegerSet, otherwise it will violate a gtest
/// assertion.
inline IntegerPolyhedron parsePoly(StringRef str) {
  MLIRContext context(MLIRContext::Threading::DISABLED);
  FailureOr<IntegerPolyhedron> poly = parseIntegerSetToFAC(str, &context);
  EXPECT_TRUE(succeeded(poly));
  return *poly;
}

/// Parse a list of StringRefs to IntegerRelation and combine them into a
/// PresburgerSet be using the union operation. It is expected that the strings
/// are all valid IntegerSet representation and that all of them have the same
/// number of dimensions as is specified by the numDims argument.
inline PresburgerSet
parsePresburgerSetFromPolyStrings(unsigned numDims, ArrayRef<StringRef> strs,
                                  unsigned numSymbols = 0) {
  PresburgerSet set = PresburgerSet::getEmpty(
      PresburgerSpace::getSetSpace(numDims, numSymbols));
  for (StringRef str : strs)
    set.unionInPlace(parsePoly(str));
  return set;
}

inline Matrix makeMatrix(unsigned numRow, unsigned numColumns,
                         ArrayRef<SmallVector<int64_t, 8>> matrix) {
  Matrix results(numRow, numColumns);
  assert(matrix.size() == numRow);
  for (unsigned i = 0; i < numRow; ++i) {
    assert(matrix[i].size() == numColumns &&
           "Output expression has incorrect dimensionality!");
    for (unsigned j = 0; j < numColumns; ++j)
      results(i, j) = matrix[i][j];
  }
  return results;
}

/// Construct a PWMAFunction given the dimensionalities and an array describing
/// the list of pieces. Each piece is given by a string describing the domain
/// and a 2D array that represents the output.
inline PWMAFunction parsePWMAF(
    unsigned numInputs, unsigned numOutputs,
    ArrayRef<std::pair<StringRef, SmallVector<SmallVector<int64_t, 8>, 8>>>
        data,
    unsigned numSymbols = 0) {
  static MLIRContext context;

  PWMAFunction result(PresburgerSpace::getSetSpace(
                          /*numDims=*/numInputs - numSymbols, numSymbols),
                      numOutputs);
  for (const auto &pair : data) {
    IntegerPolyhedron domain = parsePoly(pair.first);

    result.addPiece(
        domain, makeMatrix(numOutputs, domain.getNumVars() + 1, pair.second));
  }
  return result;
}

/// lhs and rhs represent non-negative integers or positive infinity. The
/// infinity case corresponds to when the Optional is empty.
inline bool infinityOrUInt64LE(Optional<uint64_t> lhs, Optional<uint64_t> rhs) {
  // No constraint.
  if (!rhs)
    return true;
  // Finite rhs provided so lhs has to be finite too.
  if (!lhs)
    return false;
  return *lhs <= *rhs;
}

/// Expect that the computed volume is a valid overapproximation of
/// the true volume `trueVolume`, while also being at least as good an
/// approximation as `resultBound`.
inline void
expectComputedVolumeIsValidOverapprox(Optional<uint64_t> computedVolume,
                                      Optional<uint64_t> trueVolume,
                                      Optional<uint64_t> resultBound) {
  assert(infinityOrUInt64LE(trueVolume, resultBound) &&
         "can't expect result to be less than the true volume");
  EXPECT_TRUE(infinityOrUInt64LE(trueVolume, computedVolume));
  EXPECT_TRUE(infinityOrUInt64LE(computedVolume, resultBound));
}

} // namespace presburger
} // namespace mlir

#endif // MLIR_UNITTESTS_ANALYSIS_PRESBURGER_UTILS_H
