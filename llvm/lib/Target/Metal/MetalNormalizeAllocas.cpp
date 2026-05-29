//===- MetalNormalizeAllocas.cpp - Pre-serialization IR cleanup ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MetalNormalizeAllocas.h"
#include "Metal.h"
#include "MetalAddressSpaces.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "metal-normalize-allocas"

// Metal address spaces.

/// Walks users of V and returns true if any load (or store value) is float.
/// Recurses through GEP users.
static bool hasFloatUse(Value *V) {
  for (User *U : V->users()) {
    if (auto *LI = dyn_cast<LoadInst>(U))
      if (LI->getType()->isFloatTy())
        return true;
    if (auto *SI = dyn_cast<StoreInst>(U))
      if (SI->getValueOperand()->getType()->isFloatTy())
        return true;
    if (isa<GetElementPtrInst>(U))
      if (hasFloatUse(U))
        return true;
  }
  return false;
}

static bool normalizeAllocas(Module &M) {
  bool Changed = false;
  Type *I32 = Type::getInt32Ty(M.getContext());

  // Strip 'disjoint' flag from 'or' instructions (Metal v1 bitcode).
  for (Function &F : M)
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *BO = dyn_cast<PossiblyDisjointInst>(&I))
          if (BO->isDisjoint()) {
            BO->setIsDisjoint(false);
            Changed = true;
          }

  // Hoist allocas from non-entry blocks to the entry block.
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    BasicBlock &Entry = F.getEntryBlock();
    Instruction *InsertPt = &*Entry.getFirstInsertionPt();
    for (BasicBlock &BB : F) {
      if (&BB == &Entry)
        continue;
      for (auto It = BB.begin(); It != BB.end();) {
        auto *AI = dyn_cast<AllocaInst>(&*It++);
        if (!AI)
          continue;
        AI->moveBefore(InsertPt->getIterator());
        Changed = true;
      }
    }
  }

  for (Function &F : M) {
    for (BasicBlock &BB : F) {
      for (auto It = BB.begin(); It != BB.end();) {
        Instruction &I = *It++;

        // Normalize alloca i64 -> i32.
        if (auto *AI = dyn_cast<AllocaInst>(&I)) {
          auto *Size = dyn_cast<ConstantInt>(AI->getArraySize());
          if (Size && Size->getType()->isIntegerTy(64)) {
            AI->setOperand(0, ConstantInt::get(I32, Size->getZExtValue()));
            Changed = true;
          }
          continue;
        }

        // Keep no-op ptr->ptr bitcasts; they carry typed-pointer change info.
        if (isa<BitCastInst>(&I))
          continue;

        // Insert ptr->ptr bitcast before store to device/threadgroup memory
        // when the pointer's typed pointee differs from the stored value type.
        if (auto *SI = dyn_cast<StoreInst>(&I)) {
          Value *Ptr = SI->getPointerOperand();
          unsigned AS = Ptr->getType()->getPointerAddressSpace();
          if ((AS == metal::AS::Device || AS == metal::AS::Threadgroup) &&
              !isa<BitCastInst>(Ptr)) {
            Type *ValTy = SI->getValueOperand()->getType();
            bool NeedsBitcast = false;
            if (AS == metal::AS::Device && !ValTy->isFloatTy())
              NeedsBitcast = true;
            if (AS == metal::AS::Threadgroup) {
              if (auto *GEP = dyn_cast<GetElementPtrInst>(Ptr))
                if (GEP->getSourceElementType() != ValTy)
                  NeedsBitcast = true;
            }
            if (NeedsBitcast) {
              auto *BC =
                  CastInst::Create(Instruction::BitCast, Ptr, Ptr->getType(),
                                   "", SI->getIterator());
              SI->setOperand(1, BC);
              Changed = true;
            }
          }
          continue;
        }

        // Insert ptr->ptr bitcast before load from TG memory when GEP source
        // type differs from the loaded value type.
        if (auto *LI = dyn_cast<LoadInst>(&I)) {
          Value *Ptr = LI->getPointerOperand();
          if (Ptr->getType()->getPointerAddressSpace() ==
                  metal::AS::Threadgroup &&
              !isa<BitCastInst>(Ptr)) {
            if (auto *GEP = dyn_cast<GetElementPtrInst>(Ptr))
              if (GEP->getSourceElementType() != LI->getType()) {
                auto *BC =
                    CastInst::Create(Instruction::BitCast, Ptr, Ptr->getType(),
                                     "", LI->getIterator());
                LI->setOperand(0, BC);
                Changed = true;
              }
          }
          continue;
        }

        // GEP source-type fixups.
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
          unsigned AS = GEP->getPointerAddressSpace();

          // Float GEP through bfloat/half TG global pointer: insert bitcast.
          if (AS == metal::AS::Threadgroup &&
              GEP->getSourceElementType()->isFloatTy() &&
              GEP->getNumIndices() == 1 &&
              !isa<BitCastInst>(GEP->getPointerOperand())) {
            Value *Ptr = GEP->getPointerOperand();
            bool NeedsBitcast = false;
            if (auto *PGEP = dyn_cast<GetElementPtrInst>(Ptr)) {
              if (auto *GV =
                      dyn_cast<GlobalVariable>(PGEP->getPointerOperand())) {
                if (auto *AT = dyn_cast<ArrayType>(GV->getValueType()))
                  if (AT->getElementType()->isBFloatTy() ||
                      AT->getElementType()->isHalfTy())
                    NeedsBitcast = true;
              }
            }
            if (NeedsBitcast) {
              auto *BC =
                  CastInst::Create(Instruction::BitCast, Ptr, Ptr->getType(),
                                   "", GEP->getIterator());
              GEP->setOperand(0, BC);
              Changed = true;
            }
          }

          if ((AS == metal::AS::Device || AS == metal::AS::Threadgroup) &&
              GEP->getNumIndices() == 1 &&
              (GEP->getSourceElementType()->isHalfTy() ||
               GEP->getSourceElementType()->isBFloatTy())) {
            if (hasFloatUse(GEP)) {
              IRBuilder<> B(GEP);
              Type *F32 = Type::getFloatTy(M.getContext());
              Value *Idx = GEP->getOperand(1);
              Value *NewIdx = B.CreateAShr(
                  Idx, ConstantInt::get(Idx->getType(), 1), "idx_f");
              auto *NewGEP = B.CreateInBoundsGEP(F32, GEP->getPointerOperand(),
                                                 NewIdx, GEP->getName());
              GEP->replaceAllUsesWith(NewGEP);
              GEP->eraseFromParent();
              Changed = true;
            } else if (GEP->getSourceElementType()->isBFloatTy() &&
                       !isa<BitCastInst>(GEP->getPointerOperand())) {
              auto *BC = CastInst::Create(
                  Instruction::BitCast, GEP->getPointerOperand(),
                  GEP->getPointerOperand()->getType(), "", GEP->getIterator());
              GEP->setOperand(0, BC);
              Changed = true;
            }
          }
        }
      }
    }
  }

  return Changed;
}

PreservedAnalyses MetalNormalizeAllocasPass::run(Module &M,
                                                 ModuleAnalysisManager &AM) {
  return normalizeAllocas(M) ? PreservedAnalyses::none()
                             : PreservedAnalyses::all();
}

bool MetalNormalizeAllocasLegacy::runOnModule(Module &M) {
  return normalizeAllocas(M);
}

char MetalNormalizeAllocasLegacy::ID = 0;

INITIALIZE_PASS(MetalNormalizeAllocasLegacy, DEBUG_TYPE,
                "Metal Normalize Allocas", false, false)

ModulePass *llvm::createMetalNormalizeAllocasLegacyPass() {
  return new MetalNormalizeAllocasLegacy();
}
