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
// 2. Warp reduction folding (i32, redux.sync.add targets):
//    Transforms full-warp butterfly add reductions into redux.sync.add when
//    each lane-id bit is used exactly once. Supported pattern:
//    - add(%x, shfl.sync.bfly.i32(-1, %x, 1 << I, 31)), I = 0..4 in any order
//      => redux.sync.add(%x, -1)
//
//===----------------------------------------------------------------------===//

#include "NVPTXTargetMachine.h"
#include "NVPTXUtilities.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"

#define DEBUG_TYPE "nvptx-ir-peephole"

using namespace llvm;
using namespace llvm::PatternMatch;

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

// Fold a five-step butterfly add reduction into redux.sync.add.
static bool tryFoldWarpReduceAdd(Instruction *Sum) {
  // The chain is erased after replacing Sum. Keep users before their operands.
  SmallVector<Instruction *, 10> Chain;
  IntrinsicInst *FirstShfl = nullptr;
  Value *Src = Sum;
  unsigned SeenLanes = 0;

  for (unsigned I = 0; I < 5; ++I) {
    auto *Step = cast<Instruction>(Src);
    // Intermediate partial sums must only feed the next step.
    if (I != 0 && !Step->hasNUses(2))
      return false;

    Value *Op0, *Op1;
    if (!match(Step, m_Add(m_Value(Op0), m_Value(Op1))))
      return false;

    uint64_t LaneMask = 0;
    IntrinsicInst *Shfl = nullptr;
    for (auto [Candidate, Other] : {std::pair(Op0, Op1), std::pair(Op1, Op0)}) {
      auto *II = dyn_cast<IntrinsicInst>(Candidate);
      const APInt *LM;
      if (!II || II->getIntrinsicID() != Intrinsic::nvvm_shfl_sync_bfly_i32 ||
          II->hasOperandBundles() || !II->hasOneUse() ||
          !match(II->getArgOperand(0), m_AllOnes()) ||
          II->getArgOperand(1) != Other ||
          !match(II->getArgOperand(2), m_APInt(LM)) ||
          !match(II->getArgOperand(3), m_SpecificInt(31)))
        continue;
      Src = Other;
      LaneMask = LM->getZExtValue();
      Shfl = II;
      break;
    }
    if (!Shfl || !isPowerOf2_64(LaneMask) || LaneMask >= 32 ||
        (SeenLanes & LaneMask))
      return false;
    SeenLanes |= LaneMask;

    // Every step but the last must continue the chain.
    if (I != 4 && !isa<Instruction>(Src))
      return false;

    Chain.push_back(Step);
    Chain.push_back(Shfl);
    FirstShfl = Shfl;
  }

  LLVM_DEBUG(dbgs() << "Found full-warp butterfly add reduction: " << *Sum
                    << "\n");

  // Place the reduction at the first shuffle. The redux.sync.add wrap-around
  // matches the add operations, so nsw/nuw flags are not preserved.
  IRBuilder<> Builder(FirstShfl);
  Value *Redux =
      Builder.CreateIntrinsic(Intrinsic::nvvm_redux_sync_add, {},
                              {Src, Constant::getAllOnesValue(Src->getType())});
  Redux->takeName(Sum);
  Sum->replaceAllUsesWith(Redux);
  for (Instruction *I : Chain)
    I->eraseFromParent();
  return true;
}

static bool foldWarpReductions(Function &F, const NVPTXSubtarget &ST) {
  if (!ST.hasReduxSync())
    return false;

  // Folding erases instructions that may be later candidates.
  SmallVector<WeakVH, 8> Candidates;
  for (Instruction &I : instructions(F))
    if (match(&I, m_c_Add(m_Value(),
                          m_Intrinsic<Intrinsic::nvvm_shfl_sync_bfly_i32>())))
      Candidates.push_back(&I);

  bool Changed = false;
  for (WeakVH &V : Candidates)
    if (auto *I = cast_or_null<Instruction>(V))
      Changed |= tryFoldWarpReduceAdd(I);
  return Changed;
}

namespace {

struct NVPTXIRPeephole : public FunctionPass {
  static char ID;
  NVPTXIRPeephole() : FunctionPass(ID) {}
  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
  }
};

} // namespace

char NVPTXIRPeephole::ID = 0;
INITIALIZE_PASS_BEGIN(NVPTXIRPeephole, "nvptx-ir-peephole", "NVPTX IR Peephole",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(NVPTXIRPeephole, "nvptx-ir-peephole", "NVPTX IR Peephole",
                    false, false)

bool NVPTXIRPeephole::runOnFunction(Function &F) {
  auto &TM = getAnalysis<TargetPassConfig>().getTM<NVPTXTargetMachine>();
  const NVPTXSubtarget &ST = TM.getSubtarget<NVPTXSubtarget>(F);
  bool Changed = foldFMA(F);
  Changed |= foldWarpReductions(F, ST);
  return Changed;
}

FunctionPass *llvm::createNVPTXIRPeepholePass() {
  return new NVPTXIRPeephole();
}

PreservedAnalyses NVPTXIRPeepholePass::run(Function &F,
                                           FunctionAnalysisManager &) {
  const NVPTXSubtarget &ST = TM.getSubtarget<NVPTXSubtarget>(F);
  bool Changed = foldFMA(F);
  Changed |= foldWarpReductions(F, ST);
  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
