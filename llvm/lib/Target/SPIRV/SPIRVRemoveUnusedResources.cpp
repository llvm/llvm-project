//===- SPIRVRemoveUnusedResources.cpp - Remove unused resources ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file Pass for removing unused global variables used to store resource
// handles
//
//===----------------------------------------------------------------------===//

#include "SPIRV.h"
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
#include "llvm/Support/CommandLine.h"

#define DEBUG_TYPE "spirv-remove-unused-resources"

using namespace llvm;

static cl::opt<bool> DisableSPIRVRemoveUnusedResources(
    "disable-spirv-remove-unused-resources",
    cl::desc("Disable spirv-remove-unused-resources pass"), cl::init(false),
    cl::Hidden);

static bool isResourceHandleCreation(Intrinsic::ID ID) {
  return ID == Intrinsic::spv_resource_handlefrombinding ||
         ID == Intrinsic::spv_resource_handlefromimplicitbinding ||
         ID == Intrinsic::spv_resource_counterhandlefrombinding ||
         ID == Intrinsic::spv_resource_counterhandlefromimplicitbinding;
}

// Remove unused resource handle intrinsics, private global variables that only
// provide storage for these handles, and corresponding stores of handle to
// global variable
static bool removeUnusedResources(Module &M) {
  if (DisableSPIRVRemoveUnusedResources)
    return false;

  SmallVector<IntrinsicInst *> Handles;
  SmallSetVector<GlobalVariable *, 4> DeadGlobals;
  for (Function &F : M) {
    for (Instruction &I : instructions(F)) {
      if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
        if (!isResourceHandleCreation(II->getIntrinsicID()))
          continue;
        Handles.push_back(II);

        for (User *U : II->users()) {
          auto *SI = dyn_cast<StoreInst>(U);
          if (!SI || SI->isVolatile() || SI->getValueOperand() != II)
            continue;
          auto *GV = dyn_cast<GlobalVariable>(SI->getPointerOperand());
          if (!GV || !GV->hasLocalLinkage() || !GV->hasOneUser())
            continue;
          assert(*GV->user_begin() == SI);
          DeadGlobals.insert(GV);
        }
      }
    }
  }

  bool Changed = !DeadGlobals.empty();
  for (GlobalVariable *GV : DeadGlobals) {
    cast<StoreInst>(*GV->user_begin())->eraseFromParent();
    GV->eraseFromParent();
  }

  // Use reverse order so, in case of hidden counters that live in a buffer, the
  // counter handle is removed before the corresponding buffer handle. However,
  // our traversal doesn't go in RPOT order, so may encounter insts before their
  // users. Could either do RPOT or repeat below loop until fixed point. Ignore
  // for now
  for (IntrinsicInst *II : reverse(Handles)) {
    if (II->use_empty()) {
      II->eraseFromParent();
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
