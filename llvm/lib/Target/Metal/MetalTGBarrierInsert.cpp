//===- MetalTGBarrierInsert.cpp - Insert TG memory barriers ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MetalTGBarrierInsert.h"
#include "Metal.h"
#include "MetalAddressSpaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "metal-tg-barrier-insert"

// Metal threadgroup address space.

// AIR barrier intrinsic names (new and legacy).
static constexpr StringLiteral kBarrier("air.wg.barrier");
static constexpr StringLiteral kBarrierOld("air.threadgroup.barrier");

static bool isTGStore(const Instruction *I) {
  if (auto *SI = dyn_cast<StoreInst>(I))
    return SI->getPointerAddressSpace() == metal::AS::Threadgroup;
  return false;
}

static bool isTGLoad(const Instruction *I) {
  if (auto *LI = dyn_cast<LoadInst>(I))
    return LI->getPointerAddressSpace() == metal::AS::Threadgroup;
  return false;
}

static bool isBarrierCall(const Instruction *I) {
  if (auto *CI = dyn_cast<CallInst>(I))
    if (const Function *F = CI->getCalledFunction()) {
      StringRef N = F->getName();
      return N == kBarrier || N == kBarrierOld;
    }
  return false;
}

static CallInst *createBarrier(IRBuilder<> &B, Module &M) {
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  FunctionType *FTy =
      FunctionType::get(Type::getVoidTy(Ctx), {I32, I32}, false);
  FunctionCallee FC = M.getOrInsertFunction(kBarrier, FTy);
  return B.CreateCall(FC, {ConstantInt::get(I32, 2), ConstantInt::get(I32, 1)});
}

static bool predecessorEndsWithBarrier(BasicBlock *BB) {
  if (auto *Pred = BB->getSinglePredecessor())
    if (auto *Term = Pred->getTerminator())
      if (Instruction *Prev = Term->getPrevNode())
        if (isBarrierCall(Prev))
          return true;
  return false;
}

static void ensureBarrierBeforeConditionalBranch(BranchInst *BI, Module &M,
                                                 bool &Changed) {
  auto *Parent = BI->getParent();
  if (Instruction *Prev = BI->getPrevNode())
    if (isBarrierCall(Prev))
      return;
  if (predecessorEndsWithBarrier(Parent))
    return;

  Parent->splitBasicBlock(BI, Parent->getName() + ".tgbr");
  IRBuilder<> B(Parent->getTerminator());
  createBarrier(B, M);
  Changed = true;
}

static bool tgBarrierInsert(Module &M) {
  bool Changed = false;

  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    // Collect blocks with TG stores and loads.
    SmallPtrSet<BasicBlock *, 8> TGStoreBlocks, TGLoadBlocks;
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (isTGStore(&I))
          TGStoreBlocks.insert(&BB);
        if (isTGLoad(&I))
          TGLoadBlocks.insert(&BB);
      }
    }
    if (TGStoreBlocks.empty())
      continue;

    // Conditional-branch successor targets (barrier divergence risk).
    SmallPtrSet<BasicBlock *, 8> CondTargets;
    for (BasicBlock &BB : F) {
      auto *BI = dyn_cast<BranchInst>(BB.getTerminator());
      if (BI && BI->isConditional()) {
        CondTargets.insert(BI->getSuccessor(0));
        CondTargets.insert(BI->getSuccessor(1));
      }
    }

    // Strategy 1: barrier before TG stores in non-conditional-target blocks.
    // Within a block, only the first TG store in a "store burst" needs the
    // barrier; subsequent consecutive stores by the same thread cannot race
    // with the prior barrier-protected region. A new TG load resets the
    // burst (the load might consume a value that needs fresh synchronisation
    // before the next store).
    for (BasicBlock &BB : F) {
      if (!TGStoreBlocks.count(&BB) || CondTargets.count(&BB))
        continue;
      bool BarrierActive = false;
      for (auto It = BB.begin(); It != BB.end(); ++It) {
        if (isBarrierCall(&*It)) {
          BarrierActive = true;
          continue;
        }
        if (isTGLoad(&*It)) {
          BarrierActive = false;
          continue;
        }
        if (!isTGStore(&*It))
          continue;
        if (BarrierActive)
          continue;
        IRBuilder<> B(&*It);
        createBarrier(B, M);
        Changed = true;
        BarrierActive = true;
      }
    }

    // Strategy 2: barrier before any conditional branch whose true successor
    // writes TG memory, so all threads participate.
    for (BasicBlock &BB : F) {
      auto *BI = dyn_cast<BranchInst>(BB.getTerminator());
      if (!BI || !BI->isConditional())
        continue;
      BasicBlock *TrueBB = BI->getSuccessor(0);
      if (TGStoreBlocks.count(TrueBB))
        ensureBarrierBeforeConditionalBranch(BI, M, Changed);
    }

    // Strategy 3: WAR hazard - barrier between TG load and TG store.
    for (BasicBlock &BB : F) {
      if (!TGLoadBlocks.count(&BB))
        continue;

      bool SeenUnguardedLoad = false;
      for (auto It = BB.begin(); It != BB.end(); ++It) {
        if (isTGLoad(&*It)) {
          SeenUnguardedLoad = true;
        } else if (isBarrierCall(&*It)) {
          SeenUnguardedLoad = false;
        } else if (isTGStore(&*It) && SeenUnguardedLoad) {
          IRBuilder<> B(&*It);
          createBarrier(B, M);
          Changed = true;
          SeenUnguardedLoad = false;
        }
      }

      if (!SeenUnguardedLoad)
        continue;

      Instruction *Term = BB.getTerminator();
      bool SuccHasTGStore = false;
      for (unsigned I = 0; I < Term->getNumSuccessors(); ++I) {
        BasicBlock *Succ = Term->getSuccessor(I);
        if (TGStoreBlocks.count(Succ)) {
          SuccHasTGStore = true;
          break;
        }
        Instruction *SuccTerm = Succ->getTerminator();
        if (isa<BranchInst>(SuccTerm)) {
          for (unsigned J = 0; J < SuccTerm->getNumSuccessors(); ++J) {
            if (TGStoreBlocks.count(SuccTerm->getSuccessor(J))) {
              bool SuccHasBarrier = false;
              for (Instruction &SI : *Succ)
                if (isBarrierCall(&SI)) {
                  SuccHasBarrier = true;
                  break;
                }
              if (!SuccHasBarrier)
                SuccHasTGStore = true;
            }
          }
        }
      }

      if (SuccHasTGStore) {
        // Barrier immediately before a conditional branch in the same block
        // crashes Metal's GPU JIT. Split off so they live in separate blocks.
        if (auto *CBI = dyn_cast<BranchInst>(Term)) {
          if (CBI->isConditional()) {
            BB.splitBasicBlock(Term, BB.getName() + ".war");
            IRBuilder<> B(BB.getTerminator());
            createBarrier(B, M);
          } else {
            IRBuilder<> B(Term);
            createBarrier(B, M);
          }
        } else {
          IRBuilder<> B(Term);
          createBarrier(B, M);
        }
        Changed = true;
      }
    }
  }

  return Changed;
}

PreservedAnalyses MetalTGBarrierInsertPass::run(Module &M,
                                                ModuleAnalysisManager &AM) {
  return tgBarrierInsert(M) ? PreservedAnalyses::none()
                            : PreservedAnalyses::all();
}

bool MetalTGBarrierInsertLegacy::runOnModule(Module &M) {
  return tgBarrierInsert(M);
}

char MetalTGBarrierInsertLegacy::ID = 0;

INITIALIZE_PASS(MetalTGBarrierInsertLegacy, DEBUG_TYPE,
                "Metal Threadgroup Barrier Insertion", false, false)

ModulePass *llvm::createMetalTGBarrierInsertLegacyPass() {
  return new MetalTGBarrierInsertLegacy();
}
