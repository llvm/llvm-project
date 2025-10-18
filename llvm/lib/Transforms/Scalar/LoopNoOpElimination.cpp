//===- LoopNoOpElimination.cpp - Loop No-Op Elimination Pass --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass attempts to spot and eliminate no-op operations in loop bodies.
// For example loop Vectorization may create loops like the following.
//
// vector.scevcheck:
//   %1 = add i64 %flatten.tripcount, -1
//   %2 = icmp ugt i64 %1, 4294967295
//   br i1 %2, label %scalar.ph, label %vector.ph
// vector.ph:
//    %iv = phi i64 [ 0, %vector.scevcheck], [ %iv.next, %vector.ph ]
//    %m  = and i64 %iv, 4294967295 ; 0xffff_fffe  no op
//    %p  = getelementptr inbounds <4 x i32>, ptr %A, i64 %m
//    %load = load <4 x i32>, ptr %p, align 4
//    %1 = add <4 x i32> %load,  %X
//    store <4 x i32> %1, ptr %p, align 4
//    %iv.next = add nuw i64 %iv, 4
//    %c  = icmp ult i64 %iv.next, %N
//    br i1 %c, label %vector.ph, label %exit
//  exit:
//    ret void
//
// The vectorizer creates the SCEV check block to perform
// runtime IV checks. This block can be used to determine true
// range of the the IV as entry into the vector loop is only possible
// for certain tripcount values.
//
// Currently this pass only supports spotting no-op AND operations in loop
// bodies.
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/LoopNoOpElimination.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include <iterator>
#include <optional>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "loop-noop-elim"

STATISTIC(NumEliminated, "Number of redundant instructions eliminated");

static BasicBlock *getSCEVCheckBB(Function &F) {
  for (BasicBlock &BB : F)
    if (BB.getName() == "vector.scevcheck")
      return &BB;

  return nullptr;
}

// Use vector.check block to determine if we can eliminate a bounds check on
// the IV if we know that we can only enter the vector block if the tripcount
// is within certain bounds.
static bool tryElimAndMaskOnPHI(Loop *L, Instruction *AndInstr, PHINode *IndVar,
                                ScalarEvolution *SE, Function &F) {
  Value *Op0 = AndInstr->getOperand(0);
  Value *Op1 = AndInstr->getOperand(1);

  auto *Mask = dyn_cast<ConstantInt>(Op0 == IndVar ? Op1 : Op0);
  if (!Mask)
    return false;

  auto CheckConditional = [](BranchInst *BranchI, CmpInst *CmpI,
                             unsigned ExpectedPred, BasicBlock *Header,
                             BasicBlock *PreHeader, Loop *L,
                             Value *LatchCmpV) -> bool {
    // Make sure that the conditional operator is what we
    // expect
    unsigned CmpIOpcode = CmpI->getPredicate();
    if (CmpIOpcode != ExpectedPred)
      return false;

    // Check that in the case of a true result we actually
    // branch to the loop
    Value *TrueDest = BranchI->getOperand(1);
    if (TrueDest != PreHeader && TrueDest != Header)
      return false;

    // Check that the conditional variable that is used for the
    // SCEV check is actually used in the latch compare instruction
    auto *LatchCmpInst = L->getLatchCmpInst();
    if (!LatchCmpInst)
      return false;

    if (LatchCmpInst->getOperand(0) != LatchCmpV &&
        LatchCmpInst->getOperand(1) != LatchCmpV) {
      return false;
    }

    return true;
  };

  // Determine if there's a runtime SCEV check block
  // and use that to determine if we can elim the phinode
  if (auto *SCEVCheckBB = getSCEVCheckBB(F)) {
    // Determine if the SCEV check BB branches to the loop preheader
    // or header
    BasicBlock *PreHeader = L->getLoopPreheader();
    BasicBlock *Header = L->getHeader();
    if (PreHeader && PreHeader->getUniquePredecessor() != SCEVCheckBB &&
        Header != SCEVCheckBB)
      return false;

    // We're interested in a SCEV check block with a branch instruction
    // terminator
    if (auto *BranchI = dyn_cast<BranchInst>(SCEVCheckBB->getTerminator())) {
      if (!BranchI->isConditional())
        return false;

      Value *Condition = BranchI->getCondition();
      if (auto *CmpI = dyn_cast<CmpInst>(Condition)) {
        // Check if the condition for the terminating instruction
        // is doing some comparison with a constant integer. If not
        // we can't elim our AND mask
        Value *CmpOp0 = CmpI->getOperand(0);
        Value *CmpOp1 = CmpI->getOperand(1);
        auto *CmpConstant = (dyn_cast<ConstantInt>(CmpOp0))
                                ? dyn_cast<ConstantInt>(CmpOp0)
                                : dyn_cast<ConstantInt>(CmpOp1);
        if (!CmpConstant)
          return false;

        if ((CmpConstant == CmpOp1 &&
             CheckConditional(BranchI, CmpI, CmpInst::ICMP_UGT, Header,
                              PreHeader, L, CmpOp0)) ||
            (CmpConstant == CmpOp0 &&
             CheckConditional(BranchI, CmpI, CmpInst::ICMP_ULT, Header,
                              PreHeader, L, CmpOp1))) {

          // TODO: inverse operation needs to be checked
          // We can eliminate the AND mask
          if (CmpConstant->uge(Mask->getZExtValue())) {
            AndInstr->replaceAllUsesWith(IndVar);
            return true;
          }
        }
      }
    }
  }

  return false;
}

static bool tryElimPHINodeUsers(Loop *L, PHINode *PN, ScalarEvolution *SE,
                                Function &F) {
  bool Changed = false;
  for (auto *U : PN->users()) {
    auto *I = dyn_cast<Instruction>(U);
    switch (I->getOpcode()) {
    case Instruction::And:
      if (tryElimAndMaskOnPHI(L, I, PN, SE, F)) {
        Changed |= true;
        NumEliminated++;
      }
      break;
    default:
      break;
    }
  }
  return Changed;
}

bool LoopNoOpEliminationPass::runImpl(Function &F) {
  bool Changed = false;
  for (Loop *L : *LI) {
    LoopBlocksRPO RPOT(L);
    RPOT.perform(LI);

    for (BasicBlock *BB : RPOT)
      for (Instruction &I : *BB)
        if (auto *PN = dyn_cast<PHINode>(&I))
          Changed |= tryElimPHINodeUsers(L, PN, SE, F);
  }

  return Changed;
}

PreservedAnalyses LoopNoOpEliminationPass::run(Function &F,
                                               FunctionAnalysisManager &AM) {
  LI = &AM.getResult<LoopAnalysis>(F);
  // There are no loops in the function. Return before computing other
  // expensive analyses.
  if (LI->empty())
    return PreservedAnalyses::all();
  SE = &AM.getResult<ScalarEvolutionAnalysis>(F);
  DT = &AM.getResult<DominatorTreeAnalysis>(F);
  TLI = &AM.getResult<TargetLibraryAnalysis>(F);

  if (runImpl(F))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserve<LoopAnalysis>();
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<ScalarEvolutionAnalysis>();
  PA.preserve<LoopAccessAnalysis>();

  return PA;
}
