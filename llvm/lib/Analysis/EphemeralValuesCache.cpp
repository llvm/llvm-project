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

AnalysisKey EphemeralValuesAnalysis::Key;

EphemeralValuesAnalysis::Result
EphemeralValuesAnalysis::run(Function &F, FunctionAnalysisManager &FAM) {
  auto &AC = FAM.getResult<AssumptionAnalysis>(F);
  SmallPtrSet<const Value *, 32> EphValues;
  CodeMetrics::collectEphemeralValues(&F, &AC, EphValues);
  return EphValues;
}

} // namespace llvm
