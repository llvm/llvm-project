//===------ NVPTXFoldFMA.cpp - Fold FMA --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements FMA folding for float/double type for NVPTX. It folds
// following patterns:
// 1. fadd(fmul(a, b), c) => fma(a, b, c)
// 2. fadd(c, fmul(a, b)) => fma(a, b, c)
// 3. fadd(fmul(a, b), fmul(c, d)) => fma(a, b, fmul(c, d))
// 4. fsub(fmul(a, b), c) => fma(a, b, fneg(c))
// 5. fsub(a, fmul(b, c)) => fma(fneg(b), c, a)
// 6. fsub(fmul(a, b), fmul(c, d)) => fma(a, b, fneg(fmul(c, d)))
//===----------------------------------------------------------------------===//

#include "NVPTXUtilities.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"

#define DEBUG_TYPE "nvptx-fold-fma"

using namespace llvm;

static bool foldFMA(Function &F) {
  bool Changed = false;
  SmallVector<BinaryOperator *, 16> FAddFSubInsts;

  // Collect all float/double FAdd/FSub instructions with allow-contract
  for (auto &I : instructions(F)) {
    if (auto *BI = dyn_cast<BinaryOperator>(&I)) {
      // Only FAdd and FSub are supported.
      if (BI->getOpcode() != Instruction::FAdd &&
          BI->getOpcode() != Instruction::FSub)
        continue;

      // At minimum, the instruction should have allow-contract.
      if (!BI->hasAllowContract())
        continue;

      // Only float and double are supported.
      if (!BI->getType()->isFloatTy() && !BI->getType()->isDoubleTy())
        continue;

      FAddFSubInsts.push_back(BI);
    }
  }

  auto tryFoldBinaryFMul = [](BinaryOperator *BI, Value *MulOperand,
                              Value *OtherOperand, bool IsFirstOperand,
                              bool IsFSub) -> bool {
    auto *FMul = dyn_cast<BinaryOperator>(MulOperand);
    if (!FMul || FMul->getOpcode() != Instruction::FMul || !FMul->hasOneUse() ||
        !FMul->hasAllowContract())
      return false;

    LLVM_DEBUG({
      const char *OpName = IsFSub ? "FSub" : "FAdd";
      dbgs() << "Found " << OpName << " with FMul (single use) as "
             << (IsFirstOperand ? "first" : "second") << " operand: " << *BI
             << "\n";
    });

    Value *MulOp0 = FMul->getOperand(0);
    Value *MulOp1 = FMul->getOperand(1);
    IRBuilder<> Builder(BI);
    Value *FMA = nullptr;

    if (!IsFSub) {
      // fadd(fmul(a, b), c) => fma(a, b, c)
      // fadd(c, fmul(a, b)) => fma(a, b, c)
      FMA = Builder.CreateIntrinsic(Intrinsic::fma, {BI->getType()},
                                    {MulOp0, MulOp1, OtherOperand});
    } else {
      if (IsFirstOperand) {
        // fsub(fmul(a, b), c) => fma(a, b, fneg(c))
        Value *NegOtherOp = Builder.CreateFNeg(OtherOperand);
        cast<Instruction>(NegOtherOp)->setFastMathFlags(BI->getFastMathFlags());
        FMA = Builder.CreateIntrinsic(Intrinsic::fma, {BI->getType()},
                                      {MulOp0, MulOp1, NegOtherOp});
      } else {
        // fsub(a, fmul(b, c)) => fma(fneg(b), c, a)
        Value *NegMulOp0 = Builder.CreateFNeg(MulOp0);
        cast<Instruction>(NegMulOp0)->setFastMathFlags(
            FMul->getFastMathFlags());
        FMA = Builder.CreateIntrinsic(Intrinsic::fma, {BI->getType()},
                                      {NegMulOp0, MulOp1, OtherOperand});
      }
    }

    // Combine fast-math flags from the original instructions
    auto *FMAInst = cast<Instruction>(FMA);
    FastMathFlags BinaryFMF = BI->getFastMathFlags();
    FastMathFlags FMulFMF = FMul->getFastMathFlags();
    FastMathFlags NewFMF = FastMathFlags::intersectRewrite(BinaryFMF, FMulFMF) |
                           FastMathFlags::unionValue(BinaryFMF, FMulFMF);
    FMAInst->setFastMathFlags(NewFMF);

    LLVM_DEBUG({
      const char *OpName = IsFSub ? "FSub" : "FAdd";
      dbgs() << "Replacing " << OpName << " with FMA: " << *FMA << "\n";
    });
    BI->replaceAllUsesWith(FMA);
    BI->eraseFromParent();
    FMul->eraseFromParent();
    return true;
  };

  for (auto *BI : FAddFSubInsts) {
    Value *Op0 = BI->getOperand(0);
    Value *Op1 = BI->getOperand(1);
    bool IsFSub = BI->getOpcode() == Instruction::FSub;

    if (tryFoldBinaryFMul(BI, Op0, Op1, true /*IsFirstOperand*/, IsFSub) ||
        tryFoldBinaryFMul(BI, Op1, Op0, false /*IsFirstOperand*/, IsFSub))
      Changed = true;
  }

  return Changed;
}

namespace {

struct NVPTXFoldFMA : public FunctionPass {
  static char ID;
  NVPTXFoldFMA() : FunctionPass(ID) {}
  bool runOnFunction(Function &F) override;
};

} // namespace

char NVPTXFoldFMA::ID = 0;
INITIALIZE_PASS(NVPTXFoldFMA, "nvptx-fold-fma", "NVPTX Fold FMA", false, false)

bool NVPTXFoldFMA::runOnFunction(Function &F) { return foldFMA(F); }

FunctionPass *llvm::createNVPTXFoldFMAPass() { return new NVPTXFoldFMA(); }

PreservedAnalyses NVPTXFoldFMAPass::run(Function &F,
                                        FunctionAnalysisManager &) {
  return foldFMA(F) ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
