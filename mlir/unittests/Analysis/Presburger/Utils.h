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
#include "mlir/Analysis/Presburger/IntegerPolyhedron.h"
#include "mlir/Analysis/Presburger/PresburgerSet.h"
#include "mlir/IR/MLIRContext.h"

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
