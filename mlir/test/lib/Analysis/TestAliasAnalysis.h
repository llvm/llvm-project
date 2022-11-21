//===- TestAliasAnalysis.h - MLIR Test Utility ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a common facility that can be reused for the
// testing of various aliasing analyses
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TEST_LIB_ANALYSIS_ALIASANALYSIS_H
#define MLIR_TEST_LIB_ANALYSIS_ALIASANALYSIS_H

#include "mlir/Analysis/AliasAnalysis.h"

namespace mlir {
namespace test {

/// Print the result of an alias query.
void printAliasResult(AliasResult result, Value lhs, Value rhs);
void printModRefResult(ModRefResult result, Operation *op, Value location);

struct TestAliasAnalysisBase {
  void runAliasAnalysisOnOperation(Operation *op, AliasAnalysis &aliasAnalysis);
};

struct TestAliasAnalysisModRefBase {
  void runAliasAnalysisOnOperation(Operation *op, AliasAnalysis &aliasAnalysis);
};

} // namespace test
} // namespace mlir

#endif // MLIR_TEST_LIB_ANALYSIS_ALIASANALYSIS_H
