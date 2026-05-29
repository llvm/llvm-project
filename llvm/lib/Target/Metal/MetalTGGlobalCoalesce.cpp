//===- MetalTGGlobalCoalesce.cpp - Merge cvt/dot TG globals ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MetalTGGlobalCoalesce.h"
#include "Metal.h"
#include "MetalAddressSpaces.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "metal-tg-global-coalesce"

// Metal threadgroup address space.

// AIR MMA intrinsic prefix and the well-known TG global name prefixes used by
// the Triton frontend for layout conversion / dot buffers.
static constexpr StringLiteral kMMAPrefix("air.simdgroup_matrix_8x8_");
static constexpr StringLiteral kCvtPrefix("__tg_cvt_");
static constexpr StringLiteral kDotPrefix("__tg_dot_");
static constexpr StringLiteral kDotAbInfix("_ab_");

static bool moduleHasMMA(Module &M) {
  for (Function &F : M)
    if (F.getName().starts_with(kMMAPrefix))
      return true;
  return false;
}

static bool tgGlobalCoalesce(Module &M) {
  if (!moduleHasMMA(M))
    return false;

  SmallVector<GlobalVariable *, 4> CvtGlobals;
  SmallVector<GlobalVariable *, 4> DotAbGlobals;
  for (GlobalVariable &GV : M.globals()) {
    if (GV.getAddressSpace() != metal::AS::Threadgroup)
      continue;
    auto *AT = dyn_cast<ArrayType>(GV.getValueType());
    if (!AT || AT->getNumElements() <= 64)
      continue;
    StringRef Name = GV.getName();
    if (Name.starts_with(kCvtPrefix))
      CvtGlobals.push_back(&GV);
    else if (Name.starts_with(kDotPrefix) && Name.contains(kDotAbInfix))
      DotAbGlobals.push_back(&GV);
  }

  if (CvtGlobals.empty() || DotAbGlobals.empty())
    return false;

  bool Changed = false;
  const DataLayout &DL = M.getDataLayout();

  for (GlobalVariable *Cvt : CvtGlobals) {
    auto *CvtAT = dyn_cast<ArrayType>(Cvt->getValueType());
    if (!CvtAT)
      continue;

    // Prefer matching element type; fall back to same-size element type.
    GlobalVariable *Target = nullptr;
    size_t TargetIdx = 0;
    for (size_t I = 0; I < DotAbGlobals.size(); ++I) {
      auto *DotAT = dyn_cast<ArrayType>(DotAbGlobals[I]->getValueType());
      if (!DotAT)
        continue;
      if (DotAT->getElementType() == CvtAT->getElementType()) {
        Target = DotAbGlobals[I];
        TargetIdx = I;
        break;
      }
    }

    bool NeedsBitcast = false;
    if (!Target) {
      unsigned CvtElemSize = DL.getTypeSizeInBits(CvtAT->getElementType());
      for (size_t I = 0; I < DotAbGlobals.size(); ++I) {
        auto *DotAT = dyn_cast<ArrayType>(DotAbGlobals[I]->getValueType());
        if (!DotAT)
          continue;
        unsigned DotElemSize = DL.getTypeSizeInBits(DotAT->getElementType());
        if (DotElemSize == CvtElemSize) {
          Target = DotAbGlobals[I];
          TargetIdx = I;
          NeedsBitcast = true;
          break;
        }
      }
    }

    if (!Target)
      continue;

    auto *TargetAT = cast<ArrayType>(Target->getValueType());
    Type *TargetElemTy = TargetAT->getElementType();

    // Resize target if cvt is larger.
    if (CvtAT->getNumElements() > TargetAT->getNumElements()) {
      auto *NewAT = ArrayType::get(TargetElemTy, CvtAT->getNumElements());
      auto *NewGV = new GlobalVariable(
          M, NewAT, false, Target->getLinkage(), UndefValue::get(NewAT),
          Target->getName(), Target, GlobalVariable::NotThreadLocal,
          Target->getAddressSpace());
      Target->replaceAllUsesWith(NewGV);
      Target->eraseFromParent();
      Target = NewGV;
      DotAbGlobals[TargetIdx] = NewGV;
    }

    if (NeedsBitcast) {
      SmallVector<GetElementPtrInst *, 16> GEPsToRewrite;
      for (User *U : Cvt->users())
        if (auto *GEP = dyn_cast<GetElementPtrInst>(U))
          GEPsToRewrite.push_back(GEP);

      for (GetElementPtrInst *GEP : GEPsToRewrite) {
        IRBuilder<> B(GEP);
        SmallVector<Value *, 4> Indices;
        for (Value *Idx : GEP->indices())
          Indices.push_back(Idx);

        Value *NewGEP;
        if (GEP->getSourceElementType() == CvtAT)
          NewGEP = B.CreateGEP(Target->getValueType(), Target, Indices,
                               GEP->getName(), GEP->getNoWrapFlags());
        else
          NewGEP = B.CreateGEP(TargetElemTy, Target, Indices, GEP->getName(),
                               GEP->getNoWrapFlags());

        SmallVector<Instruction *, 8> Users;
        for (User *U : GEP->users())
          Users.push_back(cast<Instruction>(U));

        for (Instruction *U : Users) {
          if (auto *SI = dyn_cast<StoreInst>(U)) {
            if (SI->getPointerOperand() == GEP) {
              IRBuilder<> SB(SI);
              Value *Val = SI->getValueOperand();
              Value *Cast =
                  SB.CreateBitCast(Val, TargetElemTy, Val->getName() + "_bc");
              SB.CreateAlignedStore(Cast, NewGEP, SI->getAlign(),
                                    SI->isVolatile());
              SI->eraseFromParent();
            }
          } else if (auto *LI = dyn_cast<LoadInst>(U)) {
            IRBuilder<> LB(LI);
            auto *NewLoad = LB.CreateAlignedLoad(
                TargetElemTy, NewGEP, LI->getAlign(), LI->getName() + "_fl");
            if (LI->isVolatile())
              NewLoad->setVolatile(true);
            Value *Cast =
                LB.CreateBitCast(NewLoad, LI->getType(), LI->getName());
            LI->replaceAllUsesWith(Cast);
            LI->eraseFromParent();
          } else {
            U->replaceUsesOfWith(GEP, NewGEP);
          }
        }

        if (GEP->use_empty())
          GEP->eraseFromParent();
      }
    } else {
      Cvt->replaceAllUsesWith(Target);
    }

    Cvt->eraseFromParent();
    Changed = true;
  }

  return Changed;
}

PreservedAnalyses MetalTGGlobalCoalescePass::run(Module &M,
                                                 ModuleAnalysisManager &AM) {
  return tgGlobalCoalesce(M) ? PreservedAnalyses::none()
                             : PreservedAnalyses::all();
}

bool MetalTGGlobalCoalesceLegacy::runOnModule(Module &M) {
  return tgGlobalCoalesce(M);
}

char MetalTGGlobalCoalesceLegacy::ID = 0;

INITIALIZE_PASS(MetalTGGlobalCoalesceLegacy, DEBUG_TYPE,
                "Metal Threadgroup Global Coalesce", false, false)

ModulePass *llvm::createMetalTGGlobalCoalesceLegacyPass() {
  return new MetalTGGlobalCoalesceLegacy();
}
