//===- MetalNaNMinMax.cpp - NaN-propagating min/max lowering --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MetalNaNMinMax.h"
#include "Metal.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "metal-nan-min-max"

namespace {
struct Mapping {
  StringLiteral LLVMName;
  StringLiteral AIRName;
};

static constexpr Mapping kNaNMinMax[] = {
    {StringLiteral("llvm.minimum.f32"), StringLiteral("air.fmin.f32")},
    {StringLiteral("llvm.maximum.f32"), StringLiteral("air.fmax.f32")},
    {StringLiteral("llvm.minimum.f16"), StringLiteral("air.fmin.f16")},
    {StringLiteral("llvm.maximum.f16"), StringLiteral("air.fmax.f16")},
};
} // namespace

// Replace a vector llvm.minimum/maximum call with elementwise extracts,
// scalar llvm.minimum/maximum calls, and re-assembly via insertelement.
// AIR has no vector fmin/fmax intrinsic, so the correct lowering is to
// scalarize before the scalar rename + NaN-propagation loop runs.
static bool scalarizeVectorMinMax(Module &M) {
  bool Changed = false;
  Intrinsic::ID Ids[] = {Intrinsic::minimum, Intrinsic::maximum};

  for (Intrinsic::ID ID : Ids) {
    SmallVector<Function *, 4> VectorDecls;
    for (Function &F : M) {
      if (F.getIntrinsicID() != ID)
        continue;
      if (F.arg_empty())
        continue;
      if (isa<VectorType>(F.getReturnType()))
        VectorDecls.push_back(&F);
    }
    for (Function *F : VectorDecls) {
      auto *VTy = cast<FixedVectorType>(F->getReturnType());
      Type *EltTy = VTy->getElementType();
      Function *Scalar =
          Intrinsic::getOrInsertDeclaration(&M, ID, {EltTy});
      SmallVector<CallInst *, 8> Calls;
      for (User *U : F->users())
        if (auto *CI = dyn_cast<CallInst>(U))
          if (CI->getCalledFunction() == F)
            Calls.push_back(CI);
      for (CallInst *CI : Calls) {
        IRBuilder<> B(CI);
        Value *A = CI->getArgOperand(0);
        Value *Bv = CI->getArgOperand(1);
        Value *Acc = PoisonValue::get(VTy);
        for (unsigned I = 0, N = VTy->getNumElements(); I < N; ++I) {
          Value *Ae = B.CreateExtractElement(A, I);
          Value *Be = B.CreateExtractElement(Bv, I);
          Value *S = B.CreateCall(Scalar, {Ae, Be});
          Acc = B.CreateInsertElement(Acc, S, I);
        }
        CI->replaceAllUsesWith(Acc);
        CI->eraseFromParent();
        Changed = true;
      }
      if (F->use_empty())
        F->eraseFromParent();
    }
  }
  return Changed;
}

static bool nanMinMax(Module &M) {
  bool Changed = scalarizeVectorMinMax(M);

  // Rename the LLVM intrinsic declarations to their AIR counterparts.
  for (auto &Mapping : kNaNMinMax) {
    if (auto *F = M.getFunction(Mapping.LLVMName)) {
      F->setName(Mapping.AIRName);
      Changed = true;
    }
  }

  // Wrap each renamed call in a NaN-propagation select so the semantics of
  // the original llvm.minimum/maximum are preserved.
  for (Function &F : M) {
    for (BasicBlock &BB : F) {
      for (auto It = BB.begin(); It != BB.end();) {
        auto *CI = dyn_cast<CallInst>(&*It++);
        if (!CI || !CI->getCalledFunction())
          continue;
        StringRef Name = CI->getCalledFunction()->getName();
        bool IsMinMax = false;
        for (auto &Mapping : kNaNMinMax)
          if (Name == Mapping.AIRName) {
            IsMinMax = true;
            break;
          }
        if (!IsMinMax)
          continue;

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
    }
  }
  return Changed;
}

PreservedAnalyses MetalNaNMinMaxPass::run(Module &M,
                                          ModuleAnalysisManager &AM) {
  return nanMinMax(M) ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

bool MetalNaNMinMaxLegacy::runOnModule(Module &M) { return nanMinMax(M); }

char MetalNaNMinMaxLegacy::ID = 0;

INITIALIZE_PASS(MetalNaNMinMaxLegacy, DEBUG_TYPE, "Metal NaN-safe min/max",
                false, false)

ModulePass *llvm::createMetalNaNMinMaxLegacyPass() {
  return new MetalNaNMinMaxLegacy();
}
