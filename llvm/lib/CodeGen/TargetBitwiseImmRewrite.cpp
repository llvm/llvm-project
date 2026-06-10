//===- TargetBitwiseImmRewrite.cpp - Prefer target bitwise immediates -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Rewrite bitwise immediates using target preferences after target-independent
// IR canonicalization, but before CodeGenPrepare drops llvm.assume.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/TargetBitwiseImmRewrite.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "target-bitwise-imm-rewrite"

STATISTIC(NumBitwiseImmRewrites,
          "Number of bitwise immediates rewritten for target preference");

static bool tryRewriteBitwiseImm(BinaryOperator &BO, const TargetLowering &TLI,
                                 const DataLayout &DL, AssumptionCache &AC,
                                 DominatorTree &DT) {
  if (BO.getOpcode() != Instruction::And)
    return false;

  Type *Ty = BO.getType();
  if (!Ty->isIntegerTy())
    return false;

  auto *CI = dyn_cast<ConstantInt>(BO.getOperand(1));
  if (!CI)
    return false;

  unsigned BitWidth = Ty->getScalarSizeInBits();
  const APInt &OldImm = CI->getValue();
  if (!TLI.isPreferredBitwiseImmCandidate(BO.getOpcode(), OldImm, BitWidth))
    return false;

  KnownBits Known = computeKnownBits(BO.getOperand(0), DL, &AC, &BO, &DT);
  if (Known.Zero.isZero())
    return false;

  APInt DemandedBits = APInt::getAllOnes(BitWidth) & ~Known.Zero;
  std::optional<APInt> NewImm = TLI.getPreferredBitwiseImmForDemandedBits(
      BO.getOpcode(), OldImm, DemandedBits, BitWidth);
  if (!NewImm || *NewImm == OldImm)
    return false;

  if (NewImm->getBitWidth() != OldImm.getBitWidth())
    return false;

  APInt Diff = OldImm ^ *NewImm;
  if (!(Diff & DemandedBits).isZero())
    return false;

  BO.setOperand(1, ConstantInt::get(Ty, *NewImm));
  ++NumBitwiseImmRewrites;
  return true;
}

static bool runImpl(Function &F, const TargetMachine &TM, AssumptionCache &AC,
                    DominatorTree &DT) {
  const TargetSubtargetInfo *STI = TM.getSubtargetImpl(F);
  const TargetLowering *TLI = STI ? STI->getTargetLowering() : nullptr;
  if (!TLI)
    return false;

  bool Changed = false;
  const DataLayout &DL = F.getDataLayout();
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (auto *BO = dyn_cast<BinaryOperator>(&I))
        Changed |= tryRewriteBitwiseImm(*BO, *TLI, DL, AC, DT);

  return Changed;
}

namespace {
class TargetBitwiseImmRewrite : public FunctionPass {
public:
  static char ID;

  TargetBitwiseImmRewrite() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    auto &TM = getAnalysis<TargetPassConfig>().getTM<TargetMachine>();
    auto &AC = getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    return runImpl(F, TM, AC, DT);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addPreserved<AssumptionCacheTracker>();
    AU.addPreserved<DominatorTreeWrapperPass>();
  }
};
} // end anonymous namespace

PreservedAnalyses
TargetBitwiseImmRewritePass::run(Function &F, FunctionAnalysisManager &FAM) {
  auto &AC = FAM.getResult<AssumptionAnalysis>(F);
  auto &DT = FAM.getResult<DominatorTreeAnalysis>(F);
  if (!runImpl(F, *TM, AC, DT))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserve<AssumptionAnalysis>();
  PA.preserve<DominatorTreeAnalysis>();
  return PA;
}

char TargetBitwiseImmRewrite::ID = 0;

INITIALIZE_PASS_BEGIN(TargetBitwiseImmRewrite, DEBUG_TYPE,
                      "Rewrite target bitwise immediates", false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(TargetBitwiseImmRewrite, DEBUG_TYPE,
                    "Rewrite target bitwise immediates", false, false)

FunctionPass *llvm::createTargetBitwiseImmRewritePass() {
  return new TargetBitwiseImmRewrite();
}
