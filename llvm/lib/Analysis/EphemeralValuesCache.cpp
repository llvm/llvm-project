//===- EphemeralValuesCache.cpp - Cache collecting ephemeral values -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/EphemeralValuesCache.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CodeMetrics.h"

namespace llvm {

void EphemeralValuesCache::collectEphemeralValues() {
  CodeMetrics::collectEphemeralValues(&F, &AC, EphValues);
  Collected = true;
}

AnalysisKey EphemeralValuesAnalysis::Key;

EphemeralValuesCache
EphemeralValuesAnalysis::run(Function &F, FunctionAnalysisManager &FAM) {
  auto &AC = FAM.getResult<AssumptionAnalysis>(F);
  return EphemeralValuesCache(F, AC);
}

} // namespace llvm
