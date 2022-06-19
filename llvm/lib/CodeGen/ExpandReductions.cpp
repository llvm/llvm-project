//===--- ExpandReductions.cpp - Expand experimental reduction intrinsics --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements IR expansion for reduction intrinsics, allowing targets
// to enable the intrinsics until just before codegen.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/ExpandReductions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

using namespace llvm;

namespace {

unsigned getOpcode(Intrinsic::ID ID) {
  switch (ID) {
  case Intrinsic::vector_reduce_fadd:
    return Instruction::FAdd;
  case Intrinsic::vector_reduce_fmul:
    return Instruction::FMul;
  case Intrinsic::vector_reduce_add:
    return Instruction::Add;
  case Intrinsic::vector_reduce_mul:
    return Instruction::Mul;
  case Intrinsic::vector_reduce_and:
    return Instruction::And;
  case Intrinsic::vector_reduce_or:
    return Instruction::Or;
  case Intrinsic::vector_reduce_xor:
    return Instruction::Xor;
  case Intrinsic::vector_reduce_smax:
  case Intrinsic::vector_reduce_smin:
  case Intrinsic::vector_reduce_umax:
  case Intrinsic::vector_reduce_umin:
    return Instruction::ICmp;
  case Intrinsic::vector_reduce_fmax:
  case Intrinsic::vector_reduce_fmin:
    return Instruction::FCmp;
  default:
    llvm_unreachable("Unexpected ID");
  }
}

RecurKind getRK(Intrinsic::ID ID) {
  switch (ID) {
  case Intrinsic::vector_reduce_smax:
    return RecurKind::SMax;
  case Intrinsic::vector_reduce_smin:
    return RecurKind::SMin;
  case Intrinsic::vector_reduce_umax:
    return RecurKind::UMax;
  case Intrinsic::vector_reduce_umin:
    return RecurKind::UMin;
  case Intrinsic::vector_reduce_fmax:
    return RecurKind::FMax;
  case Intrinsic::vector_reduce_fmin:
    return RecurKind::FMin;
  default:
    return RecurKind::None;
  }
}

bool expandReductions(Function &F, const TargetTransformInfo *TTI) {
  bool Changed = false;
  SmallVector<IntrinsicInst *, 4> Worklist;
  for (auto &I : instructions(F)) {
    if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
      switch (II->getIntrinsicID()) {
      default: break;
      case Intrinsic::vector_reduce_fadd:
      case Intrinsic::vector_reduce_fmul:
      case Intrinsic::vector_reduce_add:
      case Intrinsic::vector_reduce_mul:
      case Intrinsic::vector_reduce_and:
      case Intrinsic::vector_reduce_or:
      case Intrinsic::vector_reduce_xor:
      case Intrinsic::vector_reduce_smax:
      case Intrinsic::vector_reduce_smin:
      case Intrinsic::vector_reduce_umax:
      case Intrinsic::vector_reduce_umin:
      case Intrinsic::vector_reduce_fmax:
      case Intrinsic::vector_reduce_fmin:
        if (TTI->shouldExpandReduction(II))
          Worklist.push_back(II);

        break;
      }
    }
  }

  for (auto *II : Worklist) {
    FastMathFlags FMF =
        isa<FPMathOperator>(II) ? II->getFastMathFlags() : FastMathFlags{};
    Intrinsic::ID ID = II->getIntrinsicID();
    RecurKind RK = getRK(ID);

    Value *Rdx = nullptr;
    IRBuilder<> Builder(II);
    IRBuilder<>::FastMathFlagGuard FMFGuard(Builder);
    Builder.setFastMathFlags(FMF);
    switch (ID) {
    default: llvm_unreachable("Unexpected intrinsic!");
    case Intrinsic::vector_reduce_fadd:
    case Intrinsic::vector_reduce_fmul: {
      // FMFs must be attached to the call, otherwise it's an ordered reduction
      // and it can't be handled by generating a shuffle sequence.
      Value *Acc = II->getArgOperand(0);
      Value *Vec = II->getArgOperand(1);
      if (!FMF.allowReassoc())
        Rdx = getOrderedReduction(Builder, Acc, Vec, getOpcode(ID), RK);
      else {
        if (!isPowerOf2_32(
                cast<FixedVectorType>(Vec->getType())->getNumElements()))
          continue;

        Rdx = getShuffleReduction(Builder, Vec, getOpcode(ID), RK);
        Rdx = Builder.CreateBinOp((Instruction::BinaryOps)getOpcode(ID),
                                  Acc, Rdx, "bin.rdx");
      }
      break;
    }
    case Intrinsic::vector_reduce_add:
    case Intrinsic::vector_reduce_mul:
    case Intrinsic::vector_reduce_and:
    case Intrinsic::vector_reduce_or:
    case Intrinsic::vector_reduce_xor:
    case Intrinsic::vector_reduce_smax:
    case Intrinsic::vector_reduce_smin:
    case Intrinsic::vector_reduce_umax:
    case Intrinsic::vector_reduce_umin: {
      Value *Vec = II->getArgOperand(0);
      if (!isPowerOf2_32(
              cast<FixedVectorType>(Vec->getType())->getNumElements()))
        continue;

      Rdx = getShuffleReduction(Builder, Vec, getOpcode(ID), RK);
      break;
    }
    case Intrinsic::vector_reduce_fmax:
    case Intrinsic::vector_reduce_fmin: {
      // We require "nnan" to use a shuffle reduction; "nsz" is implied by the
      // semantics of the reduction.
      Value *Vec = II->getArgOperand(0);
      if (!isPowerOf2_32(
              cast<FixedVectorType>(Vec->getType())->getNumElements()) ||
          !FMF.noNaNs())
        continue;

      Rdx = getShuffleReduction(Builder, Vec, getOpcode(ID), RK);
      break;
    }
    }
    II->replaceAllUsesWith(Rdx);
    II->eraseFromParent();
    Changed = true;
  }
  return Changed;
}

class ExpandReductions : public FunctionPass {
public:
  static char ID;
  ExpandReductions() : FunctionPass(ID) {
    initializeExpandReductionsPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    const auto *TTI =&getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    return expandReductions(F, TTI);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.setPreservesCFG();
  }
};
}

char ExpandReductions::ID;
INITIALIZE_PASS_BEGIN(ExpandReductions, "expand-reductions",
                      "Expand reduction intrinsics", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(ExpandReductions, "expand-reductions",
                    "Expand reduction intrinsics", false, false)

FunctionPass *llvm::createExpandReductionsPass() {
  return new ExpandReductions();
}

PreservedAnalyses ExpandReductionsPass::run(Function &F,
                                            FunctionAnalysisManager &AM) {
  const auto &TTI = AM.getResult<TargetIRAnalysis>(F);
  if (!expandReductions(F, &TTI))
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
