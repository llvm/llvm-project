//===- ScopPass.cpp - The base class of Passes that operate on Polly IR ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the ScopPass members.
//
//===----------------------------------------------------------------------===//

#include "polly/ScopPass.h"
#include "polly/ScopInfo.h"

#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/OptimizationDiagnosticInfo.h"
#include "llvm/Analysis/ScalarEvolutionAliasAnalysis.h"
#include "llvm/Analysis/TargetTransformInfo.h"

using namespace llvm;
using namespace polly;

bool ScopPass::runOnRegion(Region *R, RGPassManager &RGM) {
  S = nullptr;

  if (skipRegion(*R))
    return false;

  if ((S = getAnalysis<ScopInfoRegionPass>().getScop()))
    return runOnScop(*S);

  return false;
}

void ScopPass::print(raw_ostream &OS, const Module *M) const {
  if (S)
    printScop(OS, *S);
}

void ScopPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<ScopInfoRegionPass>();

  AU.addPreserved<AAResultsWrapperPass>();
  AU.addPreserved<BasicAAWrapperPass>();
  AU.addPreserved<LoopInfoWrapperPass>();
  AU.addPreserved<DominatorTreeWrapperPass>();
  AU.addPreserved<GlobalsAAWrapperPass>();
  AU.addPreserved<ScopDetectionWrapperPass>();
  AU.addPreserved<ScalarEvolutionWrapperPass>();
  AU.addPreserved<SCEVAAWrapperPass>();
  AU.addPreserved<OptimizationRemarkEmitterWrapperPass>();
  AU.addPreserved<RegionInfoPass>();
  AU.addPreserved<ScopInfoRegionPass>();
  AU.addPreserved<TargetTransformInfoWrapperPass>();
}

namespace polly {
template class OwningInnerAnalysisManagerProxy<ScopAnalysisManager, Function>;
}

namespace llvm {

template class PassManager<Scop, ScopAnalysisManager,
                           ScopStandardAnalysisResults &, SPMUpdater &>;
template class InnerAnalysisManagerProxy<ScopAnalysisManager, Function>;
template class OuterAnalysisManagerProxy<FunctionAnalysisManager, Scop,
                                         ScopStandardAnalysisResults &>;

template <>
PreservedAnalyses
PassManager<Scop, ScopAnalysisManager, ScopStandardAnalysisResults &,
            SPMUpdater &>::run(Scop &S, ScopAnalysisManager &AM,
                               ScopStandardAnalysisResults &AR, SPMUpdater &U) {
  auto PA = PreservedAnalyses::all();
  for (auto &Pass : Passes) {
    auto PassPA = Pass->run(S, AM, AR, U);

    AM.invalidate(S, PassPA);
    PA.intersect(std::move(PassPA));
  }

  // All analyses for 'this' Scop have been invalidated above.
  // If ScopPasses affect break other scops they have to propagate this
  // information through the updater
  PA.preserveSet<AllAnalysesOn<Scop>>();
  return PA;
}

bool ScopAnalysisManagerFunctionProxy::Result::invalidate(
    Function &F, const PreservedAnalyses &PA,
    FunctionAnalysisManager::Invalidator &Inv) {

  // First, check whether our ScopInfo is about to be invalidated
  auto PAC = PA.getChecker<ScopAnalysisManagerFunctionProxy>();
  if (!(PAC.preserved() || PAC.preservedSet<AllAnalysesOn<Function>>()) ||
      Inv.invalidate<ScopInfoAnalysis>(F, PA) ||
      Inv.invalidate<ScalarEvolutionAnalysis>(F, PA) ||
      Inv.invalidate<LoopAnalysis>(F, PA) ||
      Inv.invalidate<DominatorTreeAnalysis>(F, PA)) {

    // As everything depends on ScopInfo, we must drop all existing results
    for (auto &S : *SI)
      if (auto *scop = S.second.get())
        if (InnerAM)
          InnerAM->clear(*scop);

    InnerAM = nullptr;
    return true; // Invalidate the proxy result as well.
  }

  bool allPreserved = PA.allAnalysesInSetPreserved<AllAnalysesOn<Scop>>();

  // Invalidate all non-preserved analyses
  // Even if all analyses were preserved, we still need to run deferred
  // invalidation
  for (auto &S : *SI) {
    Optional<PreservedAnalyses> InnerPA;
    auto *scop = S.second.get();
    if (!scop)
      continue;

    if (auto *OuterProxy =
            InnerAM->getCachedResult<FunctionAnalysisManagerScopProxy>(*scop)) {
      for (const auto &InvPair : OuterProxy->getOuterInvalidations()) {
        auto *OuterAnalysisID = InvPair.first;
        const auto &InnerAnalysisIDs = InvPair.second;

        if (Inv.invalidate(OuterAnalysisID, F, PA)) {
          if (!InnerPA)
            InnerPA = PA;
          for (auto *InnerAnalysisID : InnerAnalysisIDs)
            InnerPA->abandon(InnerAnalysisID);
        }
      }

      if (InnerPA) {
        InnerAM->invalidate(*scop, *InnerPA);
        continue;
      }
    }

    if (!allPreserved)
      InnerAM->invalidate(*scop, PA);
  }

  return false; // This proxy is still valid
}

template <>
ScopAnalysisManagerFunctionProxy::Result
ScopAnalysisManagerFunctionProxy::run(Function &F,
                                      FunctionAnalysisManager &FAM) {
  return Result(*InnerAM, FAM.getResult<ScopInfoAnalysis>(F));
}
} // namespace llvm

namespace polly {
template <>
OwningScopAnalysisManagerFunctionProxy::Result
OwningScopAnalysisManagerFunctionProxy::run(Function &F,
                                            FunctionAnalysisManager &FAM) {
  return Result(InnerAM, FAM.getResult<ScopInfoAnalysis>(F));
}
} // namespace polly
