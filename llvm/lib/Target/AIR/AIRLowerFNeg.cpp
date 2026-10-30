//===- AIRLowerFNeg.cpp - Lower fneg to fsub ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AIRLowerFNeg.h"
#include "AIR.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "air-lower-fneg"

static bool lowerFNeg(Module &M) {
  bool Changed = false;
  for (Function &F : M)
    for (BasicBlock &BB : F)
      for (auto It = BB.begin(); It != BB.end();) {
        auto *Neg = dyn_cast<UnaryOperator>(&*It++);
        if (!Neg || Neg->getOpcode() != Instruction::FNeg)
          continue;
        IRBuilder<> B(Neg);
        Type *Ty = Neg->getType();
        Value *Sub;
        if (Ty->getScalarType()->isFloatTy() ||
            Ty->getScalarType()->isDoubleTy()) {
          Sub = B.CreateFSub(ConstantFP::getNegativeZero(Ty),
                             Neg->getOperand(0), Neg->getName());
        } else {
          // half and bfloat: the M2 Max canonicalizes a NaN operand of
          // fsub -0.0, x (0x7E00), so the sign bit is flipped in the bits.
          unsigned W = Ty->getScalarSizeInBits();
          Type *ITy = Ty->getWithNewType(B.getIntNTy(W));
          Value *Bits = B.CreateBitCast(Neg->getOperand(0), ITy);
          Sub = B.CreateBitCast(
              B.CreateXor(Bits, ConstantInt::get(ITy, APInt::getSignMask(W))),
              Ty, Neg->getName());
        }
        Neg->replaceAllUsesWith(Sub);
        Neg->eraseFromParent();
        Changed = true;
      }
  return Changed;
}

PreservedAnalyses AIRLowerFNegPass::run(Module &M,
                                          ModuleAnalysisManager &AM) {
  return lowerFNeg(M) ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

bool AIRLowerFNegLegacy::runOnModule(Module &M) { return lowerFNeg(M); }

char AIRLowerFNegLegacy::ID = 0;

INITIALIZE_PASS(AIRLowerFNegLegacy, DEBUG_TYPE, "AIR Lower FNeg", false,
                false)

ModulePass *llvm::createAIRLowerFNegLegacyPass() {
  return new AIRLowerFNegLegacy();
}
