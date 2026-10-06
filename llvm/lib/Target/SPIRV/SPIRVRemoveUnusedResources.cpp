//===- SPIRVRemoveUnusedResources.cpp - Remove unused resources ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file Pass for removing unused SPIRV global variables.
//
//===----------------------------------------------------------------------===//

#include "SPIRV.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/IR/Instructions.h"
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

// Remove module-local globals whose only users are non-volatile stores of
// resource handles.
static bool removeUnusedResources(Module &M) {
  if (DisableSPIRVRemoveUnusedResources)
    return false;

  bool Changed = false;
  for (GlobalVariable &GV : make_early_inc_range(M.globals())) {
    if (!GV.hasLocalLinkage() || GV.user_empty())
      continue;

    if (!all_of(GV.users(), [&GV](User *U) {
          auto *SI = dyn_cast<StoreInst>(U);
          if (!SI || SI->getPointerOperand() != &GV || SI->isVolatile())
            return false;

          auto *II = dyn_cast<IntrinsicInst>(SI->getValueOperand());
          return II && isResourceHandleCreation(II->getIntrinsicID());
        }))
      continue;

    for (User *U : make_early_inc_range(GV.users())) {
      cast<Instruction>(U)->eraseFromParent();
    }

    GV.eraseFromParent();
    Changed = true;
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
