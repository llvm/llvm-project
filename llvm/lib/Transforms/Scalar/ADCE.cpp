//===- ADCE.cpp - Code to perform dead code elimination -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Aggressive Dead Code Elimination pass.  This pass
// optimistically assumes that all instructions are dead until proven otherwise,
// allowing it to eliminate dead computations that other DCE passes do not
// catch, particularly involving loop computations.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/ADCE.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Scalar.h"
using namespace llvm;

#define DEBUG_TYPE "adce"

STATISTIC(NumRemoved, "Number of instructions removed");

static bool aggressiveDCE(Function& F) {
  SmallPtrSet<Instruction*, 32> Alive;
  SmallVector<Instruction*, 128> Worklist;

  // Collect the set of "root" instructions that are known live.
  for (Instruction &I : instructions(F)) {
    if (isa<TerminatorInst>(I) || isa<DbgInfoIntrinsic>(I) || I.isEHPad() ||
        I.mayHaveSideEffects()) {
      Alive.insert(&I);
      Worklist.push_back(&I);
    }
  }

  // Propagate liveness backwards to operands.
  while (!Worklist.empty()) {
    Instruction *Curr = Worklist.pop_back_val();
    for (Use &OI : Curr->operands()) {
      if (Instruction *Inst = dyn_cast<Instruction>(OI))
        if (Alive.insert(Inst).second)
          Worklist.push_back(Inst);
    }
  }

  // The inverse of the live set is the dead set.  These are those instructions
  // which have no side effects and do not influence the control flow or return
  // value of the function, and may therefore be deleted safely.
  // NOTE: We reuse the Worklist vector here for memory efficiency.
  for (Instruction &I : instructions(F)) {
    if (!Alive.count(&I)) {
      Worklist.push_back(&I);
      I.dropAllReferences();
    }
  }

  for (Instruction *&I : Worklist) {
    ++NumRemoved;
    I->eraseFromParent();
  }

  return !Worklist.empty();
}

PreservedAnalyses ADCEPass::run(Function &F) {
  if (aggressiveDCE(F))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}

namespace {
struct ADCELegacyPass : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  ADCELegacyPass() : FunctionPass(ID) {
    initializeADCELegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function& F) override {
    if (skipOptnoneFunction(F))
      return false;
    return aggressiveDCE(F);
  }

  void getAnalysisUsage(AnalysisUsage& AU) const override {
    AU.setPreservesCFG();
    AU.addPreserved<GlobalsAAWrapperPass>();
  }
};
}

char ADCELegacyPass::ID = 0;
INITIALIZE_PASS(ADCELegacyPass, "adce", "Aggressive Dead Code Elimination",
                false, false)

FunctionPass *llvm::createAggressiveDCEPass() { return new ADCELegacyPass(); }
