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

#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/PWMAFunction.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"

#include <gtest/gtest.h>
#include <optional>

namespace mlir {
namespace presburger {

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

/// lhs and rhs represent non-negative integers or positive infinity. The
/// infinity case corresponds to when the Optional is empty.
inline bool infinityOrUInt64LE(std::optional<MPInt> lhs,
                               std::optional<MPInt> rhs) {
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
inline void expectComputedVolumeIsValidOverapprox(
    const std::optional<MPInt> &computedVolume,
    const std::optional<MPInt> &trueVolume,
    const std::optional<MPInt> &resultBound) {
  assert(infinityOrUInt64LE(trueVolume, resultBound) &&
         "can't expect result to be less than the true volume");
  EXPECT_TRUE(infinityOrUInt64LE(trueVolume, computedVolume));
  EXPECT_TRUE(infinityOrUInt64LE(computedVolume, resultBound));
}

inline void expectComputedVolumeIsValidOverapprox(
    const std::optional<MPInt> &computedVolume,
    std::optional<int64_t> trueVolume, std::optional<int64_t> resultBound) {
  expectComputedVolumeIsValidOverapprox(
      computedVolume, llvm::transformOptional(trueVolume, mpintFromInt64),
      llvm::transformOptional(resultBound, mpintFromInt64));
}

} // namespace presburger
} // namespace mlir

#endif // MLIR_UNITTESTS_ANALYSIS_PRESBURGER_UTILS_H
