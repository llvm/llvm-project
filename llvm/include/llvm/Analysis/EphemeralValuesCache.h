//===- llvm/Analysis/EphemeralValuesCache.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass caches ephemeral values, i.e., values that are only used by
// @llvm.assume intrinsics, for cheap access after the initial collection.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_EPHEMERALVALUESCACHE_H
#define LLVM_ANALYSIS_EPHEMERALVALUESCACHE_H

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class Function;
class AssumptionCache;
class Value;

/// A cache of ephemeral values within a function.
class EphemeralValuesCache {
  SmallPtrSet<const Value *, 32> EphValues;
  Function &F;
  AssumptionCache &AC;
  bool Collected = false;

  void collectEphemeralValues();

public:
  EphemeralValuesCache(Function &F, AssumptionCache &AC) : F(F), AC(AC) {}
  void clear() {
    EphValues.clear();
    Collected = false;
  }
  const SmallPtrSetImpl<const Value *> &ephValues() {
    if (!Collected)
      collectEphemeralValues();
    return EphValues;
  }
};

class EphemeralValuesAnalysis
    : public AnalysisInfoMixin<EphemeralValuesAnalysis> {
  friend AnalysisInfoMixin<EphemeralValuesAnalysis>;
  static AnalysisKey Key;

public:
  using Result = EphemeralValuesCache;
  Result run(Function &F, FunctionAnalysisManager &FAM);
};

} // namespace llvm

#endif // LLVM_ANALYSIS_EPHEMERALVALUESCACHE_H
