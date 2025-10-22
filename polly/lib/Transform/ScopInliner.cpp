//===---- ScopInliner.cpp - Polyhedral based inliner ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Take a SCC and:
// 1. If it has more than one component, bail out (contains cycles)
// 2. If it has just one component, and if the function is entirely a scop,
//    inline it.
//
//===----------------------------------------------------------------------===//

#include "polly/ScopInliner.h"
#include "polly/ScopDetection.h"
#include "polly/ScopInliner.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"

#include "polly/Support/PollyDebug.h"
#define DEBUG_TYPE "polly-scop-inliner"

using namespace llvm;
using namespace polly;

namespace {

/// Inliner implementation that works with both, LPM (using SCC_t=CallGraph) and
/// NPM (using SCC_t=LazyCallGraph::SCC)
template <typename SCC_t> bool runScopInlinerImpl(Function *F, SCC_t &SCC) {
  // We do not try to inline non-trivial SCCs because this would lead to
  // "infinite" inlining if we are not careful.
  if (SCC.size() > 1)
    return false;
  assert(SCC.size() == 1 && "found empty SCC");

  // If the function is a nullptr, or the function is a declaration.
  if (!F)
    return false;
  if (F->isDeclaration()) {
    POLLY_DEBUG(dbgs() << "Skipping " << F->getName()
                       << "because it is a declaration.\n");
    return false;
  }

  PassBuilder PB;
  // Populate analysis managers and register Polly-specific analyses.
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  auto &DT = FAM.getResult<DominatorTreeAnalysis>(*F);
  auto &SE = FAM.getResult<ScalarEvolutionAnalysis>(*F);
  auto &LI = FAM.getResult<LoopAnalysis>(*F);
  auto &RI = FAM.getResult<RegionInfoAnalysis>(*F);
  auto &AA = FAM.getResult<AAManager>(*F);
  auto &ORE = FAM.getResult<OptimizationRemarkEmitterAnalysis>(*F);
  ScopDetection SD(DT, SE, LI, RI, AA, ORE);
  SD.detect(*F);

  const bool HasScopAsTopLevelRegion =
      SD.ValidRegions.contains(RI.getTopLevelRegion());

  bool Changed = false;
  if (HasScopAsTopLevelRegion) {
    POLLY_DEBUG(dbgs() << "Skipping " << F->getName()
                       << " has scop as top level region");
    F->addFnAttr(llvm::Attribute::AlwaysInline);

    ModulePassManager MPM;
    MPM.addPass(AlwaysInlinerPass());
    Module *M = F->getParent();
    assert(M && "Function has illegal module");
    PreservedAnalyses PA = MPM.run(*M, MAM);
    if (!PA.areAllPreserved())
      Changed = true;
  } else {
    POLLY_DEBUG(dbgs() << F->getName()
                       << " does NOT have scop as top level region\n");
  }

  return Changed;
}

class ScopInlinerWrapperPass final : public CallGraphSCCPass {
  using llvm::Pass::doInitialization;

public:
  static char ID;

  ScopInlinerWrapperPass() : CallGraphSCCPass(ID) {}

  bool doInitialization(CallGraph &CG) override {
    if (!polly::PollyAllowFullFunction) {
      report_fatal_error(
          "Aborting from ScopInliner because it only makes sense to run with "
          "-polly-allow-full-function. "
          "The heurtistic for ScopInliner checks that the full function is a "
          "Scop, which happens if and only if polly-allow-full-function is "
          " enabled. "
          " If not, the entry block is not included in the Scop");
    }
    return true;
  }

  bool runOnSCC(CallGraphSCC &SCC) override {
    Function *F = (*SCC.begin())->getFunction();
    return runScopInlinerImpl(F, SCC);
  };

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    CallGraphSCCPass::getAnalysisUsage(AU);
  }
};
} // namespace
char ScopInlinerWrapperPass::ID;

Pass *polly::createScopInlinerWrapperPass() {
  ScopInlinerWrapperPass *pass = new ScopInlinerWrapperPass();
  return pass;
}

INITIALIZE_PASS_BEGIN(
    ScopInlinerWrapperPass, "polly-scop-inliner",
    "inline functions based on how much of the function is a scop.", false,
    false)
INITIALIZE_PASS_END(
    ScopInlinerWrapperPass, "polly-scop-inliner",
    "inline functions based on how much of the function is a scop.", false,
    false)

polly::ScopInlinerPass::ScopInlinerPass() {
  if (!polly::PollyAllowFullFunction) {
    report_fatal_error(
        "Aborting from ScopInliner because it only makes sense to run with "
        "-polly-allow-full-function. "
        "The heurtistic for ScopInliner checks that the full function is a "
        "Scop, which happens if and only if polly-allow-full-function is "
        " enabled. "
        " If not, the entry block is not included in the Scop");
  }
}

PreservedAnalyses polly::ScopInlinerPass::run(llvm::LazyCallGraph::SCC &SCC,
                                              llvm::CGSCCAnalysisManager &AM,
                                              llvm::LazyCallGraph &CG,
                                              llvm::CGSCCUpdateResult &UR) {
  Function *F = &SCC.begin()->getFunction();
  bool Changed = runScopInlinerImpl(F, SCC);
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
