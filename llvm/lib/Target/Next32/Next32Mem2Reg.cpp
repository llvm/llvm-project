//===----- Next32Mem2Reg.cpp - Force mem2reg for Next32 backend      ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass is a simple pass wrapper around the PromoteMemToReg function call
// exposed by the Utils library.
//
//===----------------------------------------------------------------------===//

#include "Next32.h"
#include "Next32InstrInfo.h"
#include "Next32Subtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "mem2reg"

static cl::opt<bool>
    Next32Mem2RegDisabled("disable-next32-mem2reg", cl::init(false),
                          cl::desc("Disable Next32 m2mreg promotion pass"));

STATISTIC(NumPromoted, "Number of alloca's promoted");

static bool promoteMemoryToRegister(Function &F, DominatorTree &DT,
                                    AssumptionCache &AC) {
  std::vector<AllocaInst *> Allocas;
  BasicBlock &BB = F.getEntryBlock(); // Get the entry node for the function
  bool Changed = false;

  while (true) {
    Allocas.clear();

    // Find allocas that are safe to promote, by looking at all instructions in
    // the entry node
    for (BasicBlock::iterator I = BB.begin(), E = --BB.end(); I != E; ++I)
      if (AllocaInst *AI = dyn_cast<AllocaInst>(I)) // Is it an alloca?
        if (isAllocaPromotable(AI))
          Allocas.push_back(AI);

    if (Allocas.empty())
      break;

    PromoteMemToReg(Allocas, DT, &AC);
    NumPromoted += Allocas.size();
    Changed = true;
  }
  return Changed;
}

namespace llvm {
void initializeNext32PromotePassPass(PassRegistry &);
}

namespace {

struct Next32PromotePass : public FunctionPass {
  // Pass identification, replacement for typeid
  static char ID;

  Next32PromotePass() : FunctionPass(ID) {
    initializeNext32PromotePassPass(*PassRegistry::getPassRegistry());
  }

  // runOnFunction - To run this pass, first we calculate the alloca
  // instructions that are safe for promotion, then we promote each one.
  bool runOnFunction(Function &F) override {
    if (Next32Mem2RegDisabled)
      return false;

    DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    AssumptionCache &AC =
        getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
    return promoteMemoryToRegister(F, DT, AC);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.setPreservesCFG();
  }
};

} // end anonymous namespace

char Next32PromotePass::ID = 0;

INITIALIZE_PASS_BEGIN(Next32PromotePass, "next32mem2reg",
                      "Promote Memory to Register", false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(Next32PromotePass, "next32mem2reg",
                    "Promote Memory to Register", false, false)

// createPromoteMemoryToRegister - Provide an entry point to create this pass.
FunctionPass *llvm::createNext32PromotePass() {
  return new Next32PromotePass();
}
