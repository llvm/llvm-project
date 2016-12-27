//===- LoopPassManager.cpp - Loop pass management -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LoopPassManager.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionAliasAnalysis.h"
#include "llvm/IR/Dominators.h"

using namespace llvm;

// Explicit template instantiations and specialization defininitions for core
// template typedefs.
namespace llvm {
template class PassManager<Loop>;
template class AnalysisManager<Loop>;
template class InnerAnalysisManagerProxy<LoopAnalysisManager, Function>;
template class OuterAnalysisManagerProxy<FunctionAnalysisManager, Loop>;

template <>
bool LoopAnalysisManagerFunctionProxy::Result::invalidate(
    Function &F, const PreservedAnalyses &PA,
    FunctionAnalysisManager::Invalidator &Inv) {
  // If this proxy isn't marked as preserved, the set of Function objects in
  // the module may have changed. We therefore can't call
  // InnerAM->invalidate(), because any pointers to Functions it has may be
  // stale.
  auto PAC = PA.getChecker<LoopAnalysisManagerFunctionProxy>();
  if (!PAC.preserved() && !PAC.preservedSet<AllAnalysesOn<Loop>>())
    InnerAM->clear();

  // FIXME: Proper suppor for invalidation isn't yet implemented for the LPM.

  // Return false to indicate that this result is still a valid proxy.
  return false;
}
}

PreservedAnalyses llvm::getLoopPassPreservedAnalyses() {
  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<LoopAnalysis>();
  PA.preserve<ScalarEvolutionAnalysis>();
  // TODO: What we really want to do here is preserve an AA category, but that
  // concept doesn't exist yet.
  PA.preserve<AAManager>();
  PA.preserve<BasicAA>();
  PA.preserve<GlobalsAA>();
  PA.preserve<SCEVAA>();
  return PA;
}
