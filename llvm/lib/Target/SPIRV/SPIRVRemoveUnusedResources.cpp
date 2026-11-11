//===- SPIRVRemoveUnusedResources.cpp - Remove unused resources ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file Pass for removing unused resources and their associated
// global variables
//
//===----------------------------------------------------------------------===//

#include "SPIRV.h"
#include "SPIRVSubtarget.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

#define DEBUG_TYPE "spirv-remove-unused-resources"

using namespace llvm;

static bool removeUnusedResources(Module &M) {
  if (!M.getContext()
           .getLibraryOptions()
           .get<SPIRVOptions>()
           .enable_remove_unused_resources)
    return false;

  SmallVector<IntrinsicInst *> Handles;
  SmallVector<IntrinsicInst *> CounterHandles;
  SmallSetVector<GlobalVariable *, 4> DeadGlobals;

  for (Function &F : M) {
    for (Instruction &I : instructions(F)) {
      if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
        switch (II->getIntrinsicID()) {
        default:
          continue;
        case Intrinsic::spv_resource_handlefrombinding:
        case Intrinsic::spv_resource_handlefromimplicitbinding:
          Handles.push_back(II);
          break;
        case Intrinsic::spv_resource_counterhandlefrombinding:
        case Intrinsic::spv_resource_counterhandlefromimplicitbinding:
          CounterHandles.push_back(II);
          break;
        }
      }
    }
  }

  bool Changed = false;

  auto VisitHandle = [&](Instruction *Handle, bool IsCounter) {
    for (User *U : make_early_inc_range(Handle->users())) {
      auto *SI = dyn_cast<StoreInst>(U);
      if (!SI || SI->isVolatile() || SI->getValueOperand() != Handle)
        continue;
      auto *GV = dyn_cast<GlobalVariable>(SI->getPointerOperand());
      if (!GV || !GV->hasLocalLinkage() || !GV->hasOneUser())
        continue;
      SI->eraseFromParent();
      DeadGlobals.insert(GV);
      Changed = true;
    }

    const unsigned ResourceNameOpIndex = 4;
    if (Handle->use_empty()) {
      if (!IsCounter) {
        if (auto *Name = dyn_cast<GlobalVariable>(
                Handle->getOperand(ResourceNameOpIndex)))
          DeadGlobals.insert(Name);
      }
      Handle->eraseFromParent();
      Changed = true;
    }
  };

  // Visit implicit counters first since they may use other handles
  // that are otherwise dead
  for (auto *Counter : CounterHandles)
    VisitHandle(Counter, true);
  for (auto *Handle : Handles)
    VisitHandle(Handle, false);

  for (auto *GV : DeadGlobals) {
    if (GV->user_empty()) {
      GV->eraseFromParent();
      Changed = true;
    }
  }

  return Changed;
}

PreservedAnalyses
SPIRVRemoveUnusedResourcesPass::run(Module &M, ModuleAnalysisManager &AM) {
  return removeUnusedResources(M) ? PreservedAnalyses::none()
                                  : PreservedAnalyses::all();
}

namespace {
class SPIRVRemoveUnusedResourcesLegacy : public ModulePass {
public:
  SPIRVRemoveUnusedResourcesLegacy() : ModulePass(ID) {}

  bool runOnModule(Module &M) override { return removeUnusedResources(M); }

  StringRef getPassName() const override {
    return "SPIRV Remove Unused Resources";
  }

  static char ID;
};
char SPIRVRemoveUnusedResourcesLegacy::ID = 0;
} // namespace

INITIALIZE_PASS(SPIRVRemoveUnusedResourcesLegacy, DEBUG_TYPE,
                "SPIRV Remove Unused Resources", false, false)

ModulePass *llvm::createSPIRVRemoveUnusedResourcesLegacyPass() {
  return new SPIRVRemoveUnusedResourcesLegacy();
}
