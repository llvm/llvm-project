//===- DXILForwardHandleAccesses.cpp - Cleanup Handles --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DXILForwardHandleAccesses.h"
#include "DXILShaderFlags.h"
#include "DirectX.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/DXILResource.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsDirectX.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/Local.h"

#define DEBUG_TYPE "dxil-forward-handle-accesses"

using namespace llvm;

static void diagnoseAmbiguousHandle(IntrinsicInst *NewII,
                                    IntrinsicInst *PrevII) {
  Function *F = NewII->getFunction();
  LLVMContext &Context = F->getParent()->getContext();
  Context.diagnose(DiagnosticInfoGeneric(
      Twine("Handle at \"") + NewII->getName() + "\" overwrites handle at \"" +
      PrevII->getName() + "\""));
}

static void diagnoseHandleNotFound(LoadInst *LI) {
  Function *F = LI->getFunction();
  LLVMContext &Context = F->getParent()->getContext();
  Context.diagnose(DiagnosticInfoGeneric(
      LI, Twine("Load of \"") + LI->getPointerOperand()->getName() +
              "\" is not a global resource handle"));
}

static void diagnoseUndominatedLoad(LoadInst *LI, IntrinsicInst *Handle) {
  Function *F = LI->getFunction();
  LLVMContext &Context = F->getParent()->getContext();
  Context.diagnose(DiagnosticInfoGeneric(
      LI, Twine("Load at \"") + LI->getName() +
              "\" is not dominated by handle creation at \"" +
              Handle->getName() + "\""));
}

static void
processHandle(IntrinsicInst *II,
              DenseMap<GlobalVariable *, IntrinsicInst *> &HandleMap) {
  for (User *U : II->users())
    if (auto *SI = dyn_cast<StoreInst>(U))
      if (auto *GV = dyn_cast<GlobalVariable>(SI->getPointerOperand())) {
        auto Entry = HandleMap.try_emplace(GV, II);
        if (Entry.second)
          LLVM_DEBUG(dbgs() << "Added " << GV->getName() << " to handle map\n");
        else
          diagnoseAmbiguousHandle(II, Entry.first->second);
      }
}

static bool forwardHandleAccesses(Function &F, DominatorTree &DT) {
  bool Changed = false;

  DenseMap<GlobalVariable *, IntrinsicInst *> HandleMap;
  SmallVector<LoadInst *> LoadsToProcess;
  DenseMap<AllocaInst *, SmallVector<IntrinsicInst *>> LifeTimeIntrinsicMap;
  for (BasicBlock &BB : F)
    for (Instruction &Inst : BB)
      if (auto *II = dyn_cast<IntrinsicInst>(&Inst)) {
        switch (II->getIntrinsicID()) {
        case Intrinsic::dx_resource_handlefrombinding:
        case Intrinsic::dx_resource_handlefromimplicitbinding:
          processHandle(II, HandleMap);
          break;
        case Intrinsic::lifetime_start:
        case Intrinsic::lifetime_end:
          if (II->arg_size() >= 1) {
            Value *Ptr = II->getArgOperand(0);
            if (auto *Alloca = dyn_cast<AllocaInst>(Ptr))
              LifeTimeIntrinsicMap[Alloca].push_back(II);
          }
          break;
        default:
          continue;
        }
      } else if (auto *LI = dyn_cast<LoadInst>(&Inst))
        if (isa<dxil::AnyResourceExtType>(LI->getType()))
          LoadsToProcess.push_back(LI);

  for (LoadInst *LI : LoadsToProcess) {
    Value *V = LI->getPointerOperand();
    auto *GV = dyn_cast<GlobalVariable>(V);

    // If we didn't find the global, we may need to walk through a level of
    // indirection. This generally happens at -O0.
    if (!GV) {
      if (auto *NestedLI = dyn_cast<LoadInst>(V)) {
        BasicBlock::iterator BBI(NestedLI);
        Value *Loaded = FindAvailableLoadedValue(
            NestedLI, NestedLI->getParent(), BBI, 0, nullptr, nullptr);
        GV = dyn_cast_or_null<GlobalVariable>(Loaded);
      } else if (auto *NestedAlloca = dyn_cast<AllocaInst>(V)) {

        if (auto It = LifeTimeIntrinsicMap.find(NestedAlloca);
            It != LifeTimeIntrinsicMap.end()) {
          llvm::for_each(It->second,
                         [](IntrinsicInst *II) { II->eraseFromParent(); });
          LifeTimeIntrinsicMap.erase(It);
        }

        for (auto *User : NestedAlloca->users()) {
          auto *Store = dyn_cast<StoreInst>(User);
          if (!Store)
            continue;

          Value *StoredVal = Store->getValueOperand();
          if (!StoredVal)
            continue;

          // Try direct global match
          GV = dyn_cast<GlobalVariable>(StoredVal);
          if (GV)
            break;

          // If it's a load, check its source
          if (auto *Load = dyn_cast<LoadInst>(StoredVal)) {
            GV = dyn_cast<GlobalVariable>(Load->getPointerOperand());
            if (GV)
              break;

            // If loading from an unmodified stack copy of the global, reuse the
            // global's value. Note: we are just repeating what we are doing for
            // the load case for the alloca store pattern.
            BasicBlock::iterator BBI(Load);
            Value *Loaded = FindAvailableLoadedValue(Load, Load->getParent(),
                                                     BBI, 0, nullptr, nullptr);
            GV = dyn_cast<GlobalVariable>(Loaded);
            if (GV)
              break;
          }
        }
      }
    }

    auto It = HandleMap.find(GV);
    if (It == HandleMap.end()) {
      diagnoseHandleNotFound(LI);
      continue;
    }
    Changed = true;

    if (!DT.dominates(It->second, LI)) {
      diagnoseUndominatedLoad(LI, It->second);
      continue;
    }

    LLVM_DEBUG(dbgs() << "Replacing uses of " << GV->getName() << " at "
                      << LI->getName() << " with " << It->second->getName()
                      << "\n");
    LI->replaceAllUsesWith(It->second);
    LI->eraseFromParent();
  }

  return Changed;
}

PreservedAnalyses DXILForwardHandleAccesses::run(Function &F,
                                                 FunctionAnalysisManager &AM) {
  PreservedAnalyses PA;

  DominatorTree *DT = &AM.getResult<DominatorTreeAnalysis>(F);
  bool Changed = forwardHandleAccesses(F, *DT);

  if (!Changed)
    return PreservedAnalyses::all();
  return PA;
}

namespace {
class DXILForwardHandleAccessesLegacy : public FunctionPass {
public:
  bool runOnFunction(Function &F) override {
    DominatorTree *DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    return forwardHandleAccesses(F, *DT);
  }
  StringRef getPassName() const override {
    return "DXIL Forward Handle Accesses";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
  }

  DXILForwardHandleAccessesLegacy() : FunctionPass(ID) {}

  static char ID; // Pass identification.
};
char DXILForwardHandleAccessesLegacy::ID = 0;
} // end anonymous namespace

INITIALIZE_PASS_BEGIN(DXILForwardHandleAccessesLegacy, DEBUG_TYPE,
                      "DXIL Forward Handle Accesses", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(DXILForwardHandleAccessesLegacy, DEBUG_TYPE,
                    "DXIL Forward Handle Accesses", false, false)

FunctionPass *llvm::createDXILForwardHandleAccessesLegacyPass() {
  return new DXILForwardHandleAccessesLegacy();
}
