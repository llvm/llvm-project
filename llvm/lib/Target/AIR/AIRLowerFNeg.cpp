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
        Value *Sub = B.CreateFSub(ConstantFP::getNegativeZero(Neg->getType()),
                                  Neg->getOperand(0), Neg->getName());
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
