//===- Mem2Reg.cpp - The -mem2reg pass, a wrapper around the Utils lib ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass is a simple pass wrapper around the PromoteMemToReg function call
// exposed by the Utils library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/Mem2Reg.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "mem2reg"

STATISTIC(NumPromoted, "Number of alloca's promoted");
STATISTIC(NumNotParallelPromotable, "Number of alloca's not promotable due to "
          "Tapir instructions");

static bool promoteMemoryToRegister(Function &F, DominatorTree &DT,
                                    AssumptionCache &AC, TaskInfo &TI) {
  std::vector<AllocaInst *> Allocas;
  bool Changed = false;

  // Scan the function to get its entry block and all entry blocks of detached
  // CFG's.  We can perform this scan for entry blocks once for the function,
  // because this pass preserves the CFG.
  SmallVector<BasicBlock *, 4> EntryBlocks;
  for (Task *T : depth_first(TI.getRootTask())) {
    EntryBlocks.push_back(T->getEntry());
  }

  while (true) {
    Allocas.clear();

    // Find allocas that are safe to promote, by looking at all instructions in
    // the entry node
    for (BasicBlock *BB : EntryBlocks)
      for (BasicBlock::iterator I = BB->begin(), E = --BB->end(); I != E; ++I)
        if (AllocaInst *AI = dyn_cast<AllocaInst>(I)) {      // Is it an alloca?
          if (isAllocaPromotable(AI)) {
            if (TI.isAllocaParallelPromotable(AI))
              Allocas.push_back(AI);
            else
              ++NumNotParallelPromotable;
          }
        }

    if (Allocas.empty())
      break;

    PromoteMemToReg(Allocas, DT, &AC, &TI);
    NumPromoted += Allocas.size();
    Changed = true;
  }
  return Changed;
}

PreservedAnalyses PromotePass::run(Function &F, FunctionAnalysisManager &AM) {
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &AC = AM.getResult<AssumptionAnalysis>(F);
  auto &TI = AM.getResult<TaskAnalysis>(F);
  if (!promoteMemoryToRegister(F, DT, AC, TI))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

namespace {

struct PromoteLegacyPass : public FunctionPass {
  // Pass identification, replacement for typeid
  static char ID;

  PromoteLegacyPass() : FunctionPass(ID) {
    initializePromoteLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  // runOnFunction - To run this pass, first we calculate the alloca
  // instructions that are safe for promotion, then we promote each one.
  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    AssumptionCache &AC =
        getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
    TaskInfo &TI = getAnalysis<TaskInfoWrapperPass>().getTaskInfo();
    return promoteMemoryToRegister(F, DT, AC, TI);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<TaskInfoWrapperPass>();
    AU.setPreservesCFG();
  }
};

} // end anonymous namespace

char PromoteLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(PromoteLegacyPass, "mem2reg", "Promote Memory to "
                                                    "Register",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TaskInfoWrapperPass)
INITIALIZE_PASS_END(PromoteLegacyPass, "mem2reg", "Promote Memory to Register",
                    false, false)

// createPromoteMemoryToRegister - Provide an entry point to create this pass.
FunctionPass *llvm::createPromoteMemoryToRegisterPass() {
  return new PromoteLegacyPass();
}
