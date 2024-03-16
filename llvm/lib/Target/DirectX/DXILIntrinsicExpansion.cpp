//===- DXILIntrinsicExpansion.cpp - Prepare LLVM Module for DXIL encoding--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains DXIL intrinsic expansions for those that don't have
//  opcodes in DirectX Intermediate Language (DXIL).
//===----------------------------------------------------------------------===//

#include "DXILIntrinsicExpansion.h"
#include "DirectX.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsDirectX.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "dxil-intrinsic-expansion"

using namespace llvm;

static bool isIntrinsicExpansion(Function &F) {
  switch (F.getIntrinsicID()) {
  case Intrinsic::exp:
  case Intrinsic::dx_any:
  case Intrinsic::dx_clamp:
  case Intrinsic::dx_uclamp:
  case Intrinsic::dx_lerp:
  case Intrinsic::dx_rcp:
    return true;
  }
  return false;
}

static bool expandExpIntrinsic(CallInst *Orig) {
  Value *X = Orig->getOperand(0);
  IRBuilder<> Builder(Orig->getParent());
  Builder.SetInsertPoint(Orig);
  Type *Ty = X->getType();
  Type *EltTy = Ty->getScalarType();
  Constant *Log2eConst =
      Ty->isVectorTy() ? ConstantVector::getSplat(
                             ElementCount::getFixed(
                                 cast<FixedVectorType>(Ty)->getNumElements()),
                             ConstantFP::get(EltTy, numbers::log2e))
                       : ConstantFP::get(EltTy, numbers::log2e);
  Value *NewX = Builder.CreateFMul(Log2eConst, X);
  auto *Exp2Call =
      Builder.CreateIntrinsic(Ty, Intrinsic::exp2, {NewX}, nullptr, "dx.exp2");
  Exp2Call->setTailCall(Orig->isTailCall());
  Exp2Call->setAttributes(Orig->getAttributes());
  Orig->replaceAllUsesWith(Exp2Call);
  Orig->eraseFromParent();
  return true;
}

static bool expandAnyIntrinsic(CallInst *Orig) {
  Value *X = Orig->getOperand(0);
  IRBuilder<> Builder(Orig->getParent());
  Builder.SetInsertPoint(Orig);
  Type *Ty = X->getType();
  Type *EltTy = Ty->getScalarType();

  if (!Ty->isVectorTy()) {
    Value *Cond = EltTy->isFloatingPointTy()
                      ? Builder.CreateFCmpUNE(X, ConstantFP::get(EltTy, 0))
                      : Builder.CreateICmpNE(X, ConstantInt::get(EltTy, 0));
    Orig->replaceAllUsesWith(Cond);
  } else {
    auto *XVec = dyn_cast<FixedVectorType>(Ty);
    Value *Cond =
        EltTy->isFloatingPointTy()
            ? Builder.CreateFCmpUNE(
                  X, ConstantVector::getSplat(
                         ElementCount::getFixed(XVec->getNumElements()),
                         ConstantFP::get(EltTy, 0)))
            : Builder.CreateICmpNE(
                  X, ConstantVector::getSplat(
                         ElementCount::getFixed(XVec->getNumElements()),
                         ConstantInt::get(EltTy, 0)));
    Value *Result = Builder.CreateExtractElement(Cond, (uint64_t)0);
    for (unsigned I = 1; I < XVec->getNumElements(); I++) {
      Value *Elt = Builder.CreateExtractElement(Cond, I);
      Result = Builder.CreateOr(Result, Elt);
    }
    Orig->replaceAllUsesWith(Result);
  }
  Orig->eraseFromParent();
  return true;
}

static bool expandLerpIntrinsic(CallInst *Orig) {
  Value *X = Orig->getOperand(0);
  Value *Y = Orig->getOperand(1);
  Value *S = Orig->getOperand(2);
  IRBuilder<> Builder(Orig->getParent());
  Builder.SetInsertPoint(Orig);
  auto *V = Builder.CreateFSub(Y, X);
  V = Builder.CreateFMul(S, V);
  auto *Result = Builder.CreateFAdd(X, V, "dx.lerp");
  Orig->replaceAllUsesWith(Result);
  Orig->eraseFromParent();
  return true;
}

static bool expandRcpIntrinsic(CallInst *Orig) {
  Value *X = Orig->getOperand(0);
  IRBuilder<> Builder(Orig->getParent());
  Builder.SetInsertPoint(Orig);
  Type *Ty = X->getType();
  Type *EltTy = Ty->getScalarType();
  Constant *One =
      Ty->isVectorTy()
          ? ConstantVector::getSplat(
                ElementCount::getFixed(
                    dyn_cast<FixedVectorType>(Ty)->getNumElements()),
                ConstantFP::get(EltTy, 1.0))
          : ConstantFP::get(EltTy, 1.0);
  auto *Result = Builder.CreateFDiv(One, X, "dx.rcp");
  Orig->replaceAllUsesWith(Result);
  Orig->eraseFromParent();
  return true;
}

static Intrinsic::ID getMaxForClamp(Type *ElemTy,
                                    Intrinsic::ID ClampIntrinsic) {
  if (ClampIntrinsic == Intrinsic::dx_uclamp)
    return Intrinsic::umax;
  assert(ClampIntrinsic == Intrinsic::dx_clamp);
  if (ElemTy->isVectorTy())
    ElemTy = ElemTy->getScalarType();
  if (ElemTy->isIntegerTy())
    return Intrinsic::smax;
  assert(ElemTy->isFloatingPointTy());
  return Intrinsic::maxnum;
}

static Intrinsic::ID getMinForClamp(Type *ElemTy,
                                    Intrinsic::ID ClampIntrinsic) {
  if (ClampIntrinsic == Intrinsic::dx_uclamp)
    return Intrinsic::umin;
  assert(ClampIntrinsic == Intrinsic::dx_clamp);
  if (ElemTy->isVectorTy())
    ElemTy = ElemTy->getScalarType();
  if (ElemTy->isIntegerTy())
    return Intrinsic::smin;
  assert(ElemTy->isFloatingPointTy());
  return Intrinsic::minnum;
}

static bool expandClampIntrinsic(CallInst *Orig, Intrinsic::ID ClampIntrinsic) {
  Value *X = Orig->getOperand(0);
  Value *Min = Orig->getOperand(1);
  Value *Max = Orig->getOperand(2);
  Type *Ty = X->getType();
  IRBuilder<> Builder(Orig->getParent());
  Builder.SetInsertPoint(Orig);
  auto *MaxCall = Builder.CreateIntrinsic(
      Ty, getMaxForClamp(Ty, ClampIntrinsic), {X, Min}, nullptr, "dx.max");
  auto *MinCall =
      Builder.CreateIntrinsic(Ty, getMinForClamp(Ty, ClampIntrinsic),
                              {MaxCall, Max}, nullptr, "dx.min");

  Orig->replaceAllUsesWith(MinCall);
  Orig->eraseFromParent();
  return true;
}

static bool expandIntrinsic(Function &F, CallInst *Orig) {
  switch (F.getIntrinsicID()) {
  case Intrinsic::exp:
    return expandExpIntrinsic(Orig);
  case Intrinsic::dx_any:
    return expandAnyIntrinsic(Orig);
  case Intrinsic::dx_uclamp:
  case Intrinsic::dx_clamp:
    return expandClampIntrinsic(Orig, F.getIntrinsicID());
  case Intrinsic::dx_lerp:
    return expandLerpIntrinsic(Orig);
  case Intrinsic::dx_rcp:
    return expandRcpIntrinsic(Orig);
  }
  return false;
}

static bool expansionIntrinsics(Module &M) {
  for (auto &F : make_early_inc_range(M.functions())) {
    if (!isIntrinsicExpansion(F))
      continue;
    bool IntrinsicExpanded = false;
    for (User *U : make_early_inc_range(F.users())) {
      auto *IntrinsicCall = dyn_cast<CallInst>(U);
      if (!IntrinsicCall)
        continue;
      IntrinsicExpanded = expandIntrinsic(F, IntrinsicCall);
    }
    if (F.user_empty() && IntrinsicExpanded)
      F.eraseFromParent();
  }
  return true;
}

PreservedAnalyses DXILIntrinsicExpansion::run(Module &M,
                                              ModuleAnalysisManager &) {
  if (expansionIntrinsics(M))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}

bool DXILIntrinsicExpansionLegacy::runOnModule(Module &M) {
  return expansionIntrinsics(M);
}

char DXILIntrinsicExpansionLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(DXILIntrinsicExpansionLegacy, DEBUG_TYPE,
                      "DXIL Intrinsic Expansion", false, false)
INITIALIZE_PASS_END(DXILIntrinsicExpansionLegacy, DEBUG_TYPE,
                    "DXIL Intrinsic Expansion", false, false)

ModulePass *llvm::createDXILIntrinsicExpansionLegacyPass() {
  return new DXILIntrinsicExpansionLegacy();
}
