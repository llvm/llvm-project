//===- Mem2Reg.cpp - The -mem2reg pass, a wrapper around the Utils lib ----===//
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

#define DEBUG_TYPE "mem2reg"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Transforms/Utils/UnifyFunctionExitNodes.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Instructions.h"
#include "llvm/Function.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Compiler.h"
using namespace llvm;

STATISTIC(NumPromoted, "Number of alloca's promoted");

namespace {
  struct VISIBILITY_HIDDEN PromotePass : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    PromotePass() : FunctionPass(&ID) {}

    // runOnFunction - To run this pass, first we calculate the alloca
    // instructions that are safe for promotion, then we promote each one.
    //
    virtual bool runOnFunction(Function &F);

    // getAnalysisUsage - We need dominance frontiers
    //
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<DominatorTree>();
      AU.addRequired<DominanceFrontier>();
      AU.setPreservesCFG();
      // This is a cluster of orthogonal Transforms
      AU.addPreserved<UnifyFunctionExitNodes>();
      AU.addPreservedID(LowerSwitchID);
      AU.addPreservedID(LowerInvokePassID);
      AU.addPreservedID(LowerAllocationsID);
    }
  };
}  // end of anonymous namespace

char PromotePass::ID = 0;
static RegisterPass<PromotePass> X("mem2reg", "Promote Memory to Register");

bool PromotePass::runOnFunction(Function &F) {
  std::vector<AllocaInst*> Allocas;

  BasicBlock &BB = F.getEntryBlock();  // Get the entry node for the function

  bool Changed  = false;

  DominatorTree &DT = getAnalysis<DominatorTree>();
  DominanceFrontier &DF = getAnalysis<DominanceFrontier>();

  while (1) {
    Allocas.clear();

    // Find allocas that are safe to promote, by looking at all instructions in
    // the entry node
    for (BasicBlock::iterator I = BB.begin(), E = --BB.end(); I != E; ++I)
      if (AllocaInst *AI = dyn_cast<AllocaInst>(I))       // Is it an alloca?
        if (isAllocaPromotable(AI))
          Allocas.push_back(AI);

    if (Allocas.empty()) break;

    PromoteMemToReg(Allocas, DT, DF);
    NumPromoted += Allocas.size();
    Changed = true;
  }

  return Changed;
}

// Publically exposed interface to pass...
const PassInfo *const llvm::PromoteMemoryToRegisterID = &X;
// createPromoteMemoryToRegister - Provide an entry point to create this pass.
//
FunctionPass *llvm::createPromoteMemoryToRegisterPass() {
  return new PromotePass();
}
