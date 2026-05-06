//===--- InsightBase.cpp - LLVM Advisor ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Insights/InsightBase.h"
#include "Analysis/Insights/CallFrequency.h"
#include "Analysis/Insights/CompilationFlow.h"
#include "Analysis/Insights/DebugInfo.h"
#include "Analysis/Insights/DiagnosticDelta.h"
#include "Analysis/Insights/FunctionComplexity.h"
#include "Analysis/Insights/HeaderDepth.h"
#include "Analysis/Insights/LoopNesting.h"
#include "Analysis/Insights/MetricTrends.h"
#include "Analysis/Insights/OptimizationDelta.h"
#include "Analysis/Insights/PassImpact.h"
#include "Analysis/Insights/SectionSizes.h"

using namespace llvm;
using namespace llvm::advisor;

InsightRegistry &llvm::advisor::InsightRegistry::instance() {
  static InsightRegistry Singleton;
  return Singleton;
}

void llvm::advisor::InsightRegistry::registerInsight(InsightPtr I) {
  Insights[I->getName()] = I.get();
  Owned.push_back(std::move(I));
}

void llvm::advisor::InsightRegistry::registerBuiltinInsights() {
  InsightRegistry &R = instance();
  R.registerInsight(std::make_unique<FunctionComplexityInsight>());
  R.registerInsight(std::make_unique<PassImpactInsight>());
  R.registerInsight(std::make_unique<OptimizationDeltaInsight>());
  R.registerInsight(std::make_unique<DiagnosticDeltaInsight>());
  R.registerInsight(std::make_unique<CompilationFlowInsight>());
  R.registerInsight(std::make_unique<MetricTrendsInsight>());
  R.registerInsight(std::make_unique<LoopNestingInsight>());
  R.registerInsight(std::make_unique<SectionSizesInsight>());
  R.registerInsight(std::make_unique<CallFrequencyInsight>());
  R.registerInsight(std::make_unique<DebugInfoInsight>());
  R.registerInsight(std::make_unique<HeaderDepthInsight>());
}

Insight *llvm::advisor::InsightRegistry::get(StringRef Name) const {
  return Insights.lookup(Name);
}

SmallVector<Insight *, 16> llvm::advisor::InsightRegistry::all() const {
  SmallVector<Insight *, 16> Out;
  for (auto &Entry : Insights)
    Out.push_back(Entry.getValue());
  return Out;
}

SmallVector<Insight *, 16>
llvm::advisor::InsightRegistry::getByKind(InsightKind Kind) const {
  SmallVector<Insight *, 16> Out;
  for (auto &Entry : Insights)
    if (Entry.getValue()->getKind() == Kind)
      Out.push_back(Entry.getValue());
  return Out;
}

bool llvm::advisor::InsightRegistry::isAvailable(StringRef Name,
                                                   const InsightInput &Input) const {
  Insight *I = get(Name);
  return I && I->supportsInput(Input);
}
