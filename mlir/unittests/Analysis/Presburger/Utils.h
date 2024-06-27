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

#include "mlir/Analysis/Presburger/GeneratingFunction.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/Analysis/Presburger/PWMAFunction.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/Analysis/Presburger/QuasiPolynomial.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"

#include <gtest/gtest.h>
#include <optional>

namespace mlir {
namespace presburger {
using llvm::dynamicAPIntFromInt64;

inline IntMatrix makeIntMatrix(unsigned numRow, unsigned numColumns,
                               ArrayRef<SmallVector<int, 8>> matrix) {
  IntMatrix results(numRow, numColumns);
  assert(matrix.size() == numRow);
  for (unsigned i = 0; i < numRow; ++i) {
    assert(matrix[i].size() == numColumns &&
           "Output expression has incorrect dimensionality!");
    for (unsigned j = 0; j < numColumns; ++j)
      results(i, j) = DynamicAPInt(matrix[i][j]);
  }
  return results;
}

inline FracMatrix makeFracMatrix(unsigned numRow, unsigned numColumns,
                                 ArrayRef<SmallVector<Fraction, 8>> matrix) {
  FracMatrix results(numRow, numColumns);
  assert(matrix.size() == numRow);
  for (unsigned i = 0; i < numRow; ++i) {
    assert(matrix[i].size() == numColumns &&
           "Output expression has incorrect dimensionality!");
    for (unsigned j = 0; j < numColumns; ++j)
      results(i, j) = matrix[i][j];
  }
  return results;
}

inline void EXPECT_EQ_INT_MATRIX(IntMatrix a, IntMatrix b) {
  EXPECT_EQ(a.getNumRows(), b.getNumRows());
  EXPECT_EQ(a.getNumColumns(), b.getNumColumns());

  for (unsigned row = 0; row < a.getNumRows(); row++)
    for (unsigned col = 0; col < a.getNumColumns(); col++)
      EXPECT_EQ(a(row, col), b(row, col));
}

inline void EXPECT_EQ_FRAC_MATRIX(FracMatrix a, FracMatrix b) {
  EXPECT_EQ(a.getNumRows(), b.getNumRows());
  EXPECT_EQ(a.getNumColumns(), b.getNumColumns());

  for (unsigned row = 0; row < a.getNumRows(); row++)
    for (unsigned col = 0; col < a.getNumColumns(); col++)
      EXPECT_EQ(a(row, col), b(row, col));
}

// Check the coefficients (in order) of two generating functions.
// Note that this is not a true equality check.
inline void EXPECT_EQ_REPR_GENERATINGFUNCTION(detail::GeneratingFunction a,
                                              detail::GeneratingFunction b) {
  EXPECT_EQ(a.getNumParams(), b.getNumParams());

  SmallVector<int> aSigns = a.getSigns();
  SmallVector<int> bSigns = b.getSigns();
  EXPECT_EQ(aSigns.size(), bSigns.size());
  for (unsigned i = 0, e = aSigns.size(); i < e; i++)
    EXPECT_EQ(aSigns[i], bSigns[i]);

  std::vector<detail::ParamPoint> aNums = a.getNumerators();
  std::vector<detail::ParamPoint> bNums = b.getNumerators();
  EXPECT_EQ(aNums.size(), bNums.size());
  for (unsigned i = 0, e = aNums.size(); i < e; i++)
    EXPECT_EQ_FRAC_MATRIX(aNums[i], bNums[i]);

  std::vector<std::vector<detail::Point>> aDens = a.getDenominators();
  std::vector<std::vector<detail::Point>> bDens = b.getDenominators();
  EXPECT_EQ(aDens.size(), bDens.size());
  for (unsigned i = 0, e = aDens.size(); i < e; i++) {
    EXPECT_EQ(aDens[i].size(), bDens[i].size());
    for (unsigned j = 0, f = aDens[i].size(); j < f; j++) {
      EXPECT_EQ(aDens[i][j].size(), bDens[i][j].size());
      for (unsigned k = 0, g = aDens[i][j].size(); k < g; k++) {
        EXPECT_EQ(aDens[i][j][k], bDens[i][j][k]);
      }
    }
  }
}

// Check the coefficients (in order) of two quasipolynomials.
// Note that this is not a true equality check.
inline void EXPECT_EQ_REPR_QUASIPOLYNOMIAL(QuasiPolynomial a,
                                           QuasiPolynomial b) {
  EXPECT_EQ(a.getNumInputs(), b.getNumInputs());

  SmallVector<Fraction> aCoeffs = a.getCoefficients(),
                        bCoeffs = b.getCoefficients();
  EXPECT_EQ(aCoeffs.size(), bCoeffs.size());
  for (unsigned i = 0, e = aCoeffs.size(); i < e; i++)
    EXPECT_EQ(aCoeffs[i], bCoeffs[i]);

  std::vector<std::vector<SmallVector<Fraction>>> aAff = a.getAffine(),
                                                  bAff = b.getAffine();
  EXPECT_EQ(aAff.size(), bAff.size());
  for (unsigned i = 0, e = aAff.size(); i < e; i++) {
    EXPECT_EQ(aAff[i].size(), bAff[i].size());
    for (unsigned j = 0, f = aAff[i].size(); j < f; j++)
      for (unsigned k = 0, g = a.getNumInputs(); k <= g; k++)
        EXPECT_EQ(aAff[i][j][k], bAff[i][j][k]);
  }
}

/// lhs and rhs represent non-negative integers or positive infinity. The
/// infinity case corresponds to when the Optional is empty.
inline bool infinityOrUInt64LE(std::optional<DynamicAPInt> lhs,
                               std::optional<DynamicAPInt> rhs) {
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
    const std::optional<DynamicAPInt> &computedVolume,
    const std::optional<DynamicAPInt> &trueVolume,
    const std::optional<DynamicAPInt> &resultBound) {
  assert(infinityOrUInt64LE(trueVolume, resultBound) &&
         "can't expect result to be less than the true volume");
  EXPECT_TRUE(infinityOrUInt64LE(trueVolume, computedVolume));
  EXPECT_TRUE(infinityOrUInt64LE(computedVolume, resultBound));
}

inline void expectComputedVolumeIsValidOverapprox(
    const std::optional<DynamicAPInt> &computedVolume,
    std::optional<int64_t> trueVolume, std::optional<int64_t> resultBound) {
  expectComputedVolumeIsValidOverapprox(
      computedVolume,
      llvm::transformOptional(trueVolume, dynamicAPIntFromInt64),
      llvm::transformOptional(resultBound, dynamicAPIntFromInt64));
}

} // namespace presburger
} // namespace mlir

#endif // MLIR_UNITTESTS_ANALYSIS_PRESBURGER_UTILS_H
