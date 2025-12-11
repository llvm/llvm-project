//===------ NVPTXIRPeephole.cpp - NVPTX IR Peephole --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements IR-level peephole optimizations. These transformations
// run late in the NVPTX IR pass pipeline just before the instruction selection.
//
// Currently, it implements the following transformation(s):
// 1. FMA folding (float/double types):
//    Transforms FMUL+FADD/FSUB sequences into FMA intrinsics when the
//    'contract' fast-math flag is present. Supported patterns:
//    - fadd(fmul(a, b), c) => fma(a, b, c)
//    - fadd(c, fmul(a, b)) => fma(a, b, c)
//    - fadd(fmul(a, b), fmul(c, d)) => fma(a, b, fmul(c, d))
//    - fsub(fmul(a, b), c) => fma(a, b, fneg(c))
//    - fsub(a, fmul(b, c)) => fma(fneg(b), c, a)
//    - fsub(fmul(a, b), fmul(c, d)) => fma(a, b, fneg(fmul(c, d)))
//
//===----------------------------------------------------------------------===//

#include "NVPTXUtilities.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"

#define DEBUG_TYPE "nvptx-ir-peephole"

using namespace llvm;

static bool tryFoldBinaryFMul(BinaryOperator *BI) {
  Value *Op0 = BI->getOperand(0);
  Value *Op1 = BI->getOperand(1);

  auto *FMul0 = dyn_cast<BinaryOperator>(Op0);
  auto *FMul1 = dyn_cast<BinaryOperator>(Op1);

  BinaryOperator *FMul = nullptr;
  Value *OtherOperand = nullptr;
  bool IsFirstOperand = false;

  // Either Op0 or Op1 should be a valid FMul
  if (FMul0 && FMul0->getOpcode() == Instruction::FMul && FMul0->hasOneUse() &&
      FMul0->hasAllowContract()) {
    FMul = FMul0;
    OtherOperand = Op1;
    IsFirstOperand = true;
  } else if (FMul1 && FMul1->getOpcode() == Instruction::FMul &&
             FMul1->hasOneUse() && FMul1->hasAllowContract()) {
    FMul = FMul1;
    OtherOperand = Op0;
    IsFirstOperand = false;
  } else {
    return false;
  }

  bool IsFSub = BI->getOpcode() == Instruction::FSub;
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
      Value *NegOtherOp =
          Builder.CreateFNegFMF(OtherOperand, BI->getFastMathFlags());
      FMA = Builder.CreateIntrinsic(Intrinsic::fma, {BI->getType()},
                                    {MulOp0, MulOp1, NegOtherOp});
    } else {
      // fsub(a, fmul(b, c)) => fma(fneg(b), c, a)
      Value *NegMulOp0 =
          Builder.CreateFNegFMF(MulOp0, FMul->getFastMathFlags());
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
}

static bool foldFMA(Function &F) {
  bool Changed = false;

  // Iterate and process float/double FAdd/FSub instructions with allow-contract
  for (auto &I : llvm::make_early_inc_range(instructions(F))) {
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

      if (tryFoldBinaryFMul(BI))
        Changed = true;
    }
  }
  return Changed;
}

namespace {

struct NVPTXIRPeephole : public FunctionPass {
  static char ID;
  NVPTXIRPeephole() : FunctionPass(ID) {}
  bool runOnFunction(Function &F) override;
};

} // namespace

char NVPTXIRPeephole::ID = 0;
INITIALIZE_PASS(NVPTXIRPeephole, "nvptx-ir-peephole", "NVPTX IR Peephole",
                false, false)

bool NVPTXIRPeephole::runOnFunction(Function &F) { return foldFMA(F); }

FunctionPass *llvm::createNVPTXIRPeepholePass() {
  return new NVPTXIRPeephole();
}

PreservedAnalyses NVPTXIRPeepholePass::run(Function &F,
                                           FunctionAnalysisManager &) {
  if (!foldFMA(F))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
