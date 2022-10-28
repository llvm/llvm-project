//===-- SCCP.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements Interprocedural Sparse Conditional Constant Propagation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/SCCP.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar/SCCP.h"
#include "llvm/Transforms/Utils/SCCPSolver.h"

using namespace llvm;

PreservedAnalyses IPSCCPPass::run(Module &M, ModuleAnalysisManager &AM) {
  const DataLayout &DL = M.getDataLayout();
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto GetTLI = [&FAM](Function &F) -> const TargetLibraryInfo & {
    return FAM.getResult<TargetLibraryAnalysis>(F);
  };
  auto getAnalysis = [&FAM](Function &F) -> AnalysisResultsForFn {
    DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);
    return {
        std::make_unique<PredicateInfo>(F, DT, FAM.getResult<AssumptionAnalysis>(F)),
        &DT, FAM.getCachedResult<PostDominatorTreeAnalysis>(F),
        nullptr};
  };

  if (!runIPSCCP(M, DL, GetTLI, getAnalysis))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<PostDominatorTreeAnalysis>();
  PA.preserve<FunctionAnalysisManagerModuleProxy>();
  return PA;
}

namespace {

//===--------------------------------------------------------------------===//
//
/// IPSCCP Class - This class implements interprocedural Sparse Conditional
/// Constant Propagation.
///
class IPSCCPLegacyPass : public ModulePass {
public:
  static char ID;

  IPSCCPLegacyPass() : ModulePass(ID) {
    initializeIPSCCPLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override {
    if (skipModule(M))
      return false;
    const DataLayout &DL = M.getDataLayout();
    auto GetTLI = [this](Function &F) -> const TargetLibraryInfo & {
      return this->getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
    };
    auto getAnalysis = [this](Function &F) -> AnalysisResultsForFn {
      DominatorTree &DT =
          this->getAnalysis<DominatorTreeWrapperPass>(F).getDomTree();
      return {
          std::make_unique<PredicateInfo>(
              F, DT,
              this->getAnalysis<AssumptionCacheTracker>().getAssumptionCache(
                  F)),
          nullptr,  // We cannot preserve the LI, DT or PDT with the legacy pass
          nullptr,  // manager, so set them to nullptr.
          nullptr};
    };

    return runIPSCCP(M, DL, GetTLI, getAnalysis);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
  }
};

} // end anonymous namespace

char IPSCCPLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(IPSCCPLegacyPass, "ipsccp",
                      "Interprocedural Sparse Conditional Constant Propagation",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(IPSCCPLegacyPass, "ipsccp",
                    "Interprocedural Sparse Conditional Constant Propagation",
                    false, false)

// createIPSCCPPass - This is the public interface to this file.
ModulePass *llvm::createIPSCCPPass() { return new IPSCCPLegacyPass(); }

PreservedAnalyses FunctionSpecializationPass::run(Module &M,
                                                  ModuleAnalysisManager &AM) {
  const DataLayout &DL = M.getDataLayout();
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto GetTLI = [&FAM](Function &F) -> TargetLibraryInfo & {
    return FAM.getResult<TargetLibraryAnalysis>(F);
  };
  auto GetTTI = [&FAM](Function &F) -> TargetTransformInfo & {
    return FAM.getResult<TargetIRAnalysis>(F);
  };
  auto GetAC = [&FAM](Function &F) -> AssumptionCache & {
    return FAM.getResult<AssumptionAnalysis>(F);
  };
  auto GetAnalysis = [&FAM](Function &F) -> AnalysisResultsForFn {
    DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);
    return {std::make_unique<PredicateInfo>(
                F, DT, FAM.getResult<AssumptionAnalysis>(F)),
            &DT, FAM.getCachedResult<PostDominatorTreeAnalysis>(F),
            &FAM.getResult<LoopAnalysis>(F)};
  };

  if (!runFunctionSpecialization(M, DL, GetTLI, GetTTI, GetAC, GetAnalysis))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<PostDominatorTreeAnalysis>();
  PA.preserve<FunctionAnalysisManagerModuleProxy>();
  return PA;
}

namespace {
struct FunctionSpecializationLegacyPass : public ModulePass {
  static char ID; // Pass identification, replacement for typeid
  FunctionSpecializationLegacyPass() : ModulePass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
  }

  bool runOnModule(Module &M) override {
    if (skipModule(M))
      return false;

    const DataLayout &DL = M.getDataLayout();
    auto GetTLI = [this](Function &F) -> TargetLibraryInfo & {
      return this->getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
    };
    auto GetTTI = [this](Function &F) -> TargetTransformInfo & {
      return this->getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    };
    auto GetAC = [this](Function &F) -> AssumptionCache & {
      return this->getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
    };

    auto GetAnalysis = [this](Function &F) -> AnalysisResultsForFn {
      DominatorTree &DT =
          this->getAnalysis<DominatorTreeWrapperPass>(F).getDomTree();
      return {
          std::make_unique<PredicateInfo>(
              F, DT,
              this->getAnalysis<AssumptionCacheTracker>().getAssumptionCache(
                  F)),
          nullptr, // We cannot preserve the LI, DT, or PDT with the legacy pass
          nullptr, // manager, so set them to nullptr.
          nullptr};
    };
    return runFunctionSpecialization(M, DL, GetTLI, GetTTI, GetAC, GetAnalysis);
  }
};
} // namespace

char FunctionSpecializationLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(
    FunctionSpecializationLegacyPass, "function-specialization",
    "Propagate constant arguments by specializing the function", false, false)

INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(FunctionSpecializationLegacyPass, "function-specialization",
                    "Propagate constant arguments by specializing the function",
                    false, false)

ModulePass *llvm::createFunctionSpecializationPass() {
  return new FunctionSpecializationLegacyPass();
}
