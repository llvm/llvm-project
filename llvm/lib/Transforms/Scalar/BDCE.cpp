//===---- BDCE.cpp - Bit-tracking dead code elimination -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Bit-Tracking Dead Code Elimination pass. Some
// instructions (shifts, some ands, ors, etc.) kill some of their input bits.
// We track these dead bits and remove instructions that compute only these
// dead bits.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/DemandedBits.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Operator.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "bdce"

STATISTIC(NumRemoved, "Number of instructions removed (unused)");
STATISTIC(NumSimplified, "Number of instructions trivialized (dead bits)");

namespace {
struct BDCE : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  BDCE() : FunctionPass(ID) {
    initializeBDCEPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function& F) override;

  void getAnalysisUsage(AnalysisUsage& AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<DemandedBitsWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
  }
};
}

char BDCE::ID = 0;
INITIALIZE_PASS_BEGIN(BDCE, "bdce", "Bit-Tracking Dead Code Elimination",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(DemandedBitsWrapperPass)
INITIALIZE_PASS_END(BDCE, "bdce", "Bit-Tracking Dead Code Elimination",
                    false, false)

bool BDCE::runOnFunction(Function& F) {
  if (skipOptnoneFunction(F))
    return false;
  auto &DB = getAnalysis<DemandedBitsWrapperPass>().getDemandedBits();

  SmallVector<Instruction*, 128> Worklist;
  bool Changed = false;
  for (Instruction &I : instructions(F)) {
    if (I.getType()->isIntegerTy() &&
        !DB.getDemandedBits(&I).getBoolValue()) {
      // For live instructions that have all dead bits, first make them dead by
      // replacing all uses with something else. Then, if they don't need to
      // remain live (because they have side effects, etc.) we can remove them.
      DEBUG(dbgs() << "BDCE: Trivializing: " << I << " (all bits dead)\n");
      // FIXME: In theory we could substitute undef here instead of zero.
      // This should be reconsidered once we settle on the semantics of
      // undef, poison, etc.
      Value *Zero = ConstantInt::get(I.getType(), 0);
      ++NumSimplified;
      I.replaceAllUsesWith(Zero);
      Changed = true;
    }
    if (!DB.isInstructionDead(&I))
      continue;

    Worklist.push_back(&I);
    I.dropAllReferences();
    Changed = true;
  }

  for (Instruction *&I : Worklist) {
    ++NumRemoved;
    I->eraseFromParent();
  }

  return Changed;
}

FunctionPass *llvm::createBitTrackingDCEPass() {
  return new BDCE();
}

