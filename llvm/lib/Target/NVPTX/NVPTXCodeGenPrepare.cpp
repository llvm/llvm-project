//===------ NVPTXCodeGenPrepare.cpp - NVPTX CodeGen Prepare ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements IR-level optimizations. These transformations run late
// in the NVPTX IR pass pipeline just before the instruction selection.
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
// 2. Sink casts for mul.wide:
//    If all uses of a {s|z}ext casts are multiplication and output type is
//    twice input size, then NVPTX backend can emit mul.wide operation instead
//    of mul.lo. To facilitate this at ISel we need to sink the casts to the
//    user basic blocks.
//
//===----------------------------------------------------------------------===//

#include "NVPTXUtilities.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Transforms/Utils/Local.h"

#define DEBUG_TYPE "nvptx-codegen-prepare"

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

static bool trySinkCastForWideMul(Instruction *I) {
  assert((isa<ZExtInst>(I) || isa<SExtInst>(I)) &&
         "Expected only zext/sext instructions here.");

  // For *ext casts, if the output size (O) is twice the input size (I), all
  // users are multiplications, and all inputs to these multiplications are
  // demotable to size I (either *ext or a constant), then PTX can generate=
  // a wide multiplication of I * I = O. Sink the *ext instruction to each
  // multiplication block as ISel cannot currently optimize cross-block.
  auto IsDoubleIntegerExtend = [](const Instruction *I) {
    // Require integers, disallow vectors.
    Type *SrcTy = I->getOperand(0)->getType();
    Type *DstTy = I->getType();
    if (!SrcTy->isIntegerTy() || !DstTy->isIntegerTy())
      return false;

    // Only support 2x extensions for i16 and i32 types.
    unsigned SrcSize = SrcTy->getScalarSizeInBits();
    unsigned DstSize = DstTy->getScalarSizeInBits();
    return (SrcSize * 2 == DstSize && (SrcSize == 16 || SrcSize == 32));
  };

  auto IsUserMulWideCandidate = [&](const User *U) {
    auto *UI = cast<Instruction>(U);

    // For multiplications, require that the other operand be demotabale.
    if (UI->getOpcode() == Instruction::Mul) {
      const Value *OtherV = UI->getOperand(UI->getOperand(0) == I);
      unsigned OptSize = I->getOperand(0)->getType()->getScalarSizeInBits();
      bool Signed = isa<SExtInst>(I);

      if (auto *OtherI = dyn_cast<Instruction>(OtherV)) {
        if (OtherI->getOpcode() != Instruction::ZExt &&
            OtherI->getOpcode() != Instruction::SExt)
          return false;
        if (!IsDoubleIntegerExtend(OtherI))
          return false;

        // Require that the sign matches for both input extensions.
        return Signed == (OtherI->getOpcode() == Instruction::SExt);
      }

      if (auto *OtherCI = dyn_cast<ConstantInt>(OtherV)) {
        const APInt &Val = OtherCI->getValue();
        if (Signed)
          return Val.isSignedIntN(OptSize);
        return Val.isIntN(OptSize);
      }
      return false;
    }

    // GEP gets lowered as shl in NVPTX.
    if (UI->getOpcode() == Instruction::GetElementPtr)
      return true;

    // For left shifts, require that the shift amount is a constant >= 0
    // and less than the bit width.
    if (UI->getOpcode() == Instruction::Shl) {
      if (I != UI->getOperand(0))
        return false;

      if (auto *C = dyn_cast<ConstantInt>(UI->getOperand(1))) {
        APInt ShiftAmt = C->getValue();
        unsigned BitWidth = I->getType()->getScalarSizeInBits();
        if (ShiftAmt.sge(0) && ShiftAmt.slt(BitWidth))
          return true;
      }
    }
    return false;
  };

  // This check ensures we don't re-process clones of cast created by previous
  // iteration of this pass.
  bool HasAnyOutsideUsers = any_of(I->users(), [&I](const User *U) {
    return isa<Instruction>(U) &&
           cast<Instruction>(U)->getParent() != I->getParent();
  });

  if (IsDoubleIntegerExtend(I) && HasAnyOutsideUsers &&
      all_of(I->users(), IsUserMulWideCandidate))
    return sinkCastToUsers(cast<CastInst>(I));

  return false;
}

static bool sinkCastsForWideMul(Function &F) {
  bool Changed = false;

  for (auto &I : make_early_inc_range(instructions(F))) {
    if (I.getOpcode() != Instruction::ZExt &&
        I.getOpcode() != Instruction::SExt)
      continue;

    Changed |= trySinkCastForWideMul(&I);
  }

  return Changed;
}

namespace {

struct NVPTXCodeGenPrepare : public FunctionPass {
  static char ID;
  NVPTXCodeGenPrepare() : FunctionPass(ID) {}
  bool runOnFunction(Function &F) override;
};

} // namespace

char NVPTXCodeGenPrepare::ID = 0;
INITIALIZE_PASS(NVPTXCodeGenPrepare, "nvptx-codegen-prepare",
                "NVPTX CodeGen Prepare", false, false)

bool NVPTXCodeGenPrepare::runOnFunction(Function &F) {
  bool Changed = foldFMA(F);
  Changed |= sinkCastsForWideMul(F);

  return Changed;
}

FunctionPass *llvm::createNVPTXCodeGenPreparePass() {
  return new NVPTXCodeGenPrepare();
}

PreservedAnalyses NVPTXCodeGenPreparePass::run(Function &F,
                                               FunctionAnalysisManager &) {
  bool Changed = false;

  Changed |= foldFMA(F);
  Changed |= sinkCastsForWideMul(F);

  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
