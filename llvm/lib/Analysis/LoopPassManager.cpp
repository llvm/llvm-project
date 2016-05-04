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

// Explicit instantiations for core typedef'ed templates.
namespace llvm {
template class PassManager<Loop>;
template class AnalysisManager<Loop>;
template class InnerAnalysisManagerProxy<LoopAnalysisManager, Function>;
template class OuterAnalysisManagerProxy<FunctionAnalysisManager, Loop>;
}

PreservedAnalyses llvm::getLoopPassPreservedAnalyses() {
  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<LoopAnalysis>();
  PA.preserve<ScalarEvolutionAnalysis>();
  // TODO: What we really want to do here is preserve an AA category, but that
  // concept doesn't exist yet.
  PA.preserve<BasicAA>();
  PA.preserve<GlobalsAA>();
  PA.preserve<SCEVAA>();
  return PA;
}
