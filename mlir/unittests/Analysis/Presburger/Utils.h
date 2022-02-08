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
#include "mlir/IR/MLIRContext.h"

#include <gtest/gtest.h>

namespace mlir {
/// Parses a IntegerPolyhedron from a StringRef. It is expected that the
/// string represents a valid IntegerSet, otherwise it will violate a gtest
/// assertion.
static IntegerPolyhedron parsePoly(StringRef str, MLIRContext *context) {
  FailureOr<IntegerPolyhedron> poly = parseIntegerSetToFAC(str, context);
  EXPECT_TRUE(succeeded(poly));
  return *poly;
}
} // namespace mlir

#endif // MLIR_UNITTESTS_ANALYSIS_PRESBURGER_UTILS_H
