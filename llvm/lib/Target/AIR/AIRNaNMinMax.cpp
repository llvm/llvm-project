//===- AIRNaNMinMax.cpp - NaN-propagating min/max lowering --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AIRNaNMinMax.h"
#include "AIR.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "air-nan-min-max"

static bool nanMinMax(Module &M) {
  bool Changed = false;
  SmallVector<CallInst *, 16> Calls;
  for (Function &F : M) {
    Intrinsic::ID ID = F.getIntrinsicID();
    if (ID != Intrinsic::minimum && ID != Intrinsic::maximum)
      continue;
    for (User *U : F.users())
      if (auto *CI = dyn_cast<CallInst>(U))
        if (CI->getCalledFunction() == &F)
          Calls.push_back(CI);
  }
  for (CallInst *CI : Calls) {
    IRBuilder<> B(CI->getNextNode());
    Value *A = CI->getArgOperand(0);
    Value *Bv = CI->getArgOperand(1);
    Value *IsNaN = B.CreateFCmpUNO(A, Bv, "nan_check");
    Value *NaN = ConstantFP::getNaN(CI->getType());
    Value *Sel = B.CreateSelect(IsNaN, NaN, CI, CI->getName() + ".nan");
    CI->replaceAllUsesWith(Sel);
    cast<SelectInst>(Sel)->setOperand(2, CI);
    Changed = true;
  }
  return Changed;
}

PreservedAnalyses AIRNaNMinMaxPass::run(Module &M,
                                          ModuleAnalysisManager &AM) {
  return nanMinMax(M) ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

bool AIRNaNMinMaxLegacy::runOnModule(Module &M) { return nanMinMax(M); }

char AIRNaNMinMaxLegacy::ID = 0;

INITIALIZE_PASS(AIRNaNMinMaxLegacy, DEBUG_TYPE, "AIR NaN-safe min/max",
                false, false)

ModulePass *llvm::createAIRNaNMinMaxLegacyPass() {
  return new AIRNaNMinMaxLegacy();
}
