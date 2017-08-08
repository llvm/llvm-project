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

#include "llvm/Transforms/Scalar/BDCE.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/DemandedBits.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Operator.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
using namespace llvm;

#define DEBUG_TYPE "bdce"

STATISTIC(NumRemoved, "Number of instructions removed (unused)");
STATISTIC(NumSimplified, "Number of instructions trivialized (dead bits)");

static bool bitTrackingDCE(Function &F, DemandedBits &DB) {
  SmallVector<Instruction*, 128> Worklist;
  bool Changed = false;
  for (Instruction &I : instructions(F)) {
    // If the instruction has side effects and no non-dbg uses,
    // skip it. This way we avoid computing known bits on an instruction
    // that will not help us.
    if (I.mayHaveSideEffects() && I.use_empty())
      continue;

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
      I.replaceNonMetadataUsesWith(Zero);
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

PreservedAnalyses BDCEPass::run(Function &F, FunctionAnalysisManager &AM) {
  auto &DB = AM.getResult<DemandedBitsAnalysis>(F);
  if (!bitTrackingDCE(F, DB))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  PA.preserve<GlobalsAA>();
  return PA;
}

namespace {
struct BDCELegacyPass : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  BDCELegacyPass() : FunctionPass(ID) {
    initializeBDCELegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;
    auto &DB = getAnalysis<DemandedBitsWrapperPass>().getDemandedBits();
    return bitTrackingDCE(F, DB);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<DemandedBitsWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
  }
};
}

char BDCELegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(BDCELegacyPass, "bdce",
                      "Bit-Tracking Dead Code Elimination", false, false)
INITIALIZE_PASS_DEPENDENCY(DemandedBitsWrapperPass)
INITIALIZE_PASS_END(BDCELegacyPass, "bdce",
                    "Bit-Tracking Dead Code Elimination", false, false)

FunctionPass *llvm::createBitTrackingDCEPass() { return new BDCELegacyPass(); }
