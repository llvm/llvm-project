//- CFLAndersAliasAnalysis.cpp - Unification-based Alias Analysis ---*- C++-*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a CFL-based, summary-based alias analysis algorithm. It
// differs from CFLSteensAliasAnalysis in its inclusion-based nature while
// CFLSteensAliasAnalysis is unification-based. This pass has worse performance
// than CFLSteensAliasAnalysis (the worst case complexity of
// CFLAndersAliasAnalysis is cubic, while the worst case complexity of
// CFLSteensAliasAnalysis is almost linear), but it is able to yield more
// precise analysis result. The precision of this analysis is roughly the same
// as that of an one level context-sensitive Andersen's algorithm.
//
//===----------------------------------------------------------------------===//

// N.B. AliasAnalysis as a whole is phrased as a FunctionPass at the moment, and
// CFLAndersAA is interprocedural. This is *technically* A Bad Thing, because
// FunctionPasses are only allowed to inspect the Function that they're being
// run on. Realistically, this likely isn't a problem until we allow
// FunctionPasses to run concurrently.

#include "llvm/Analysis/CFLAndersAliasAnalysis.h"
#include "CFLGraph.h"
#include "llvm/Pass.h"

using namespace llvm;
using namespace llvm::cflaa;

#define DEBUG_TYPE "cfl-anders-aa"

CFLAndersAAResult::CFLAndersAAResult() = default;

char CFLAndersAA::PassID;

CFLAndersAAResult CFLAndersAA::run(Function &F, AnalysisManager<Function> &AM) {
  return CFLAndersAAResult();
}

char CFLAndersAAWrapperPass::ID = 0;
INITIALIZE_PASS(CFLAndersAAWrapperPass, "cfl-anders-aa",
                "Inclusion-Based CFL Alias Analysis", false, true)

ImmutablePass *llvm::createCFLAndersAAWrapperPass() {
  return new CFLAndersAAWrapperPass();
}

CFLAndersAAWrapperPass::CFLAndersAAWrapperPass() : ImmutablePass(ID) {
  initializeCFLAndersAAWrapperPassPass(*PassRegistry::getPassRegistry());
}

void CFLAndersAAWrapperPass::initializePass() {}

void CFLAndersAAWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
}
