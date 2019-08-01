//===- DivRemPairs.cpp - Hoist/decompose division and remainder -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass hoists and/or decomposes integer division and remainder
// instructions to enable CFG improvements and better codegen.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/DivRemPairs.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/DebugCounter.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BypassSlowDivision.h"

using namespace llvm;

#define DEBUG_TYPE "div-rem-pairs"
STATISTIC(NumPairs, "Number of div/rem pairs");
STATISTIC(NumHoisted, "Number of instructions hoisted");
STATISTIC(NumDecomposed, "Number of instructions decomposed");
DEBUG_COUNTER(DRPCounter, "div-rem-pairs-transform",
              "Controls transformations in div-rem-pairs pass");

/// A thin wrapper to store two values that we matched as div-rem pair.
/// We want this extra indirection to avoid dealing with RAUW'ing the map keys.
struct DivRemPairWorklistEntry {
  /// The actual udiv/sdiv instruction. Source of truth.
  AssertingVH<Instruction> DivInst;

  /// The instruction that we have matched as a remainder instruction.
  /// Should only be used as Value, don't introspect it.
  AssertingVH<Instruction> RemInst;

  DivRemPairWorklistEntry(Instruction *DivInst_, Instruction *RemInst_)
      : DivInst(DivInst_), RemInst(RemInst_) {
    assert((DivInst->getOpcode() == Instruction::UDiv ||
            DivInst->getOpcode() == Instruction::SDiv) &&
           "Not a division.");
    assert(DivInst->getType() == RemInst->getType() && "Types should match.");
    // We can't check anything else about remainder instruction,
    // it's not strictly required to be a urem/srem.
  }

  /// The type for this pair, identical for both the div and rem.
  Type *getType() const { return DivInst->getType(); }

  /// Is this pair signed or unsigned?
  bool isSigned() const { return DivInst->getOpcode() == Instruction::SDiv; }

  /// In this pair, what are the divident and divisor?
  Value *getDividend() const { return DivInst->getOperand(0); }
  Value *getDivisor() const { return DivInst->getOperand(1); }
};
using DivRemWorklistTy = SmallVector<DivRemPairWorklistEntry, 4>;

/// Find matching pairs of integer div/rem ops (they have the same numerator,
/// denominator, and signedness). Place those pairs into a worklist for further
/// processing. This indirection is needed because we have to use TrackingVH<>
/// because we will be doing RAUW, and if one of the rem instructions we change
/// happens to be an input to another div/rem in the maps, we'd have problems.
static DivRemWorklistTy getWorklist(Function &F) {
  // Insert all divide and remainder instructions into maps keyed by their
  // operands and opcode (signed or unsigned).
  DenseMap<DivRemMapKey, Instruction *> DivMap;
  // Use a MapVector for RemMap so that instructions are moved/inserted in a
  // deterministic order.
  MapVector<DivRemMapKey, Instruction *> RemMap;
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (I.getOpcode() == Instruction::SDiv)
        DivMap[DivRemMapKey(true, I.getOperand(0), I.getOperand(1))] = &I;
      else if (I.getOpcode() == Instruction::UDiv)
        DivMap[DivRemMapKey(false, I.getOperand(0), I.getOperand(1))] = &I;
      else if (I.getOpcode() == Instruction::SRem)
        RemMap[DivRemMapKey(true, I.getOperand(0), I.getOperand(1))] = &I;
      else if (I.getOpcode() == Instruction::URem)
        RemMap[DivRemMapKey(false, I.getOperand(0), I.getOperand(1))] = &I;
    }
  }

  // We'll accumulate the matching pairs of div-rem instructions here.
  DivRemWorklistTy Worklist;

  // We can iterate over either map because we are only looking for matched
  // pairs. Choose remainders for efficiency because they are usually even more
  // rare than division.
  for (auto &RemPair : RemMap) {
    // Find the matching division instruction from the division map.
    Instruction *DivInst = DivMap[RemPair.first];
    if (!DivInst)
      continue;

    // We have a matching pair of div/rem instructions.
    NumPairs++;
    Instruction *RemInst = RemPair.second;

    // Place it in the worklist.
    Worklist.emplace_back(DivInst, RemInst);
  }

  return Worklist;
}

/// Find matching pairs of integer div/rem ops (they have the same numerator,
/// denominator, and signedness). If they exist in different basic blocks, bring
/// them together by hoisting or replace the common division operation that is
/// implicit in the remainder:
/// X % Y <--> X - ((X / Y) * Y).
///
/// We can largely ignore the normal safety and cost constraints on speculation
/// of these ops when we find a matching pair. This is because we are already
/// guaranteed that any exceptions and most cost are already incurred by the
/// first member of the pair.
///
/// Note: This transform could be an oddball enhancement to EarlyCSE, GVN, or
/// SimplifyCFG, but it's split off on its own because it's different enough
/// that it doesn't quite match the stated objectives of those passes.
static bool optimizeDivRem(Function &F, const TargetTransformInfo &TTI,
                           const DominatorTree &DT) {
  bool Changed = false;

  // Get the matching pairs of div-rem instructions. We want this extra
  // indirection to avoid dealing with having to RAUW the keys of the maps.
  DivRemWorklistTy Worklist = getWorklist(F);

  // Process each entry in the worklist.
  for (DivRemPairWorklistEntry &E : Worklist) {
    bool HasDivRemOp = TTI.hasDivRemOp(E.getType(), E.isSigned());

    auto &DivInst = E.DivInst;
    auto &RemInst = E.RemInst;

    // If the target supports div+rem and the instructions are in the same block
    // already, there's nothing to do. The backend should handle this. If the
    // target does not support div+rem, then we will decompose the rem.
    if (HasDivRemOp && RemInst->getParent() == DivInst->getParent())
      continue;

    bool DivDominates = DT.dominates(DivInst, RemInst);
    if (!DivDominates && !DT.dominates(RemInst, DivInst))
      continue;

    if (!DebugCounter::shouldExecute(DRPCounter))
      continue;

    if (HasDivRemOp) {
      // The target has a single div/rem operation. Hoist the lower instruction
      // to make the matched pair visible to the backend.
      if (DivDominates)
        RemInst->moveAfter(DivInst);
      else
        DivInst->moveAfter(RemInst);
      NumHoisted++;
    } else {
      // The target does not have a single div/rem operation. Decompose the
      // remainder calculation as:
      // X % Y --> X - ((X / Y) * Y).
      Value *X = E.getDividend();
      Value *Y = E.getDivisor();
      Instruction *Mul = BinaryOperator::CreateMul(DivInst, Y);
      Instruction *Sub = BinaryOperator::CreateSub(X, Mul);

      // If the remainder dominates, then hoist the division up to that block:
      //
      // bb1:
      //   %rem = srem %x, %y
      // bb2:
      //   %div = sdiv %x, %y
      // -->
      // bb1:
      //   %div = sdiv %x, %y
      //   %mul = mul %div, %y
      //   %rem = sub %x, %mul
      //
      // If the division dominates, it's already in the right place. The mul+sub
      // will be in a different block because we don't assume that they are
      // cheap to speculatively execute:
      //
      // bb1:
      //   %div = sdiv %x, %y
      // bb2:
      //   %rem = srem %x, %y
      // -->
      // bb1:
      //   %div = sdiv %x, %y
      // bb2:
      //   %mul = mul %div, %y
      //   %rem = sub %x, %mul
      //
      // If the div and rem are in the same block, we do the same transform,
      // but any code movement would be within the same block.

      if (!DivDominates)
        DivInst->moveBefore(RemInst);
      Mul->insertAfter(RemInst);
      Sub->insertAfter(Mul);

      // Now kill the explicit remainder. We have replaced it with:
      // (sub X, (mul (div X, Y), Y)
      Sub->setName(RemInst->getName() + ".decomposed");
      Instruction *OrigRemInst = RemInst;
      // Update AssertingVH<> with new instruction so it doesn't assert.
      RemInst = Sub;
      // And replace the original instruction with the new one.
      OrigRemInst->replaceAllUsesWith(Sub);
      OrigRemInst->eraseFromParent();
      NumDecomposed++;
    }
    Changed = true;
  }

  return Changed;
}

// Pass manager boilerplate below here.

namespace {
struct DivRemPairsLegacyPass : public FunctionPass {
  static char ID;
  DivRemPairsLegacyPass() : FunctionPass(ID) {
    initializeDivRemPairsLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.setPreservesCFG();
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
    FunctionPass::getAnalysisUsage(AU);
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;
    auto &TTI = getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    return optimizeDivRem(F, TTI, DT);
  }
};
} // namespace

char DivRemPairsLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(DivRemPairsLegacyPass, "div-rem-pairs",
                      "Hoist/decompose integer division and remainder", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(DivRemPairsLegacyPass, "div-rem-pairs",
                    "Hoist/decompose integer division and remainder", false,
                    false)
FunctionPass *llvm::createDivRemPairsPass() {
  return new DivRemPairsLegacyPass();
}

PreservedAnalyses DivRemPairsPass::run(Function &F,
                                       FunctionAnalysisManager &FAM) {
  TargetTransformInfo &TTI = FAM.getResult<TargetIRAnalysis>(F);
  DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);
  if (!optimizeDivRem(F, TTI, DT))
    return PreservedAnalyses::all();
  // TODO: This pass just hoists/replaces math ops - all analyses are preserved?
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  PA.preserve<GlobalsAA>();
  return PA;
}
