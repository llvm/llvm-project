//===- DXILResourceAccess.cpp - Resource access via load/store ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DXILRemoveUnusedResources.h"
#include "DirectX.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsDirectX.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#define DEBUG_TYPE "dxil-remove-unused-resources"

// Hidden option to disable the pass to make it easier to test
// other passes related to DXIL resources using llc.
static llvm::cl::opt<bool> DisableDXILRemoveUnusedResources(
    "disable-dxil-remove-unused-resources",
    llvm::cl::desc("Disable dxil-remove-unused-resources pass"),
    llvm::cl::init(false), llvm::cl::Hidden);

using namespace llvm;

// Removes all calls to intrinsics dx_resource_handlefrom{implicit}binding that
// either are not used, or their only use is in a store instruction, which
// stores the initialized handle into a global variable that does not have
// external linkage and that is not used anywhere else in the module.
static bool removeUnusedResources(Function &F) {
  if (DisableDXILRemoveUnusedResources)
    return false;

  SmallVector<Instruction *> DeadInstr;
  SmallSetVector<GlobalVariable *, 4> DeadGlobals;
  for (BasicBlock &BB : make_early_inc_range(F)) {
    for (Instruction &I : BB) {
      if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
        if (II->getIntrinsicID() != Intrinsic::dx_resource_handlefrombinding &&
            II->getIntrinsicID() !=
                Intrinsic::dx_resource_handlefromimplicitbinding)
          continue;
        if (II->user_empty()) {
          // Initialized handle is not used anywhere.
          DeadInstr.push_back(II);
          continue;
        }
        if (!II->hasOneUser())
          continue;

        // Initialized handle is only used in one store instruction, the store
        // is into global variable, and that global variable is not used
        // anywhere else and does not have external linkage.
        auto *SI = dyn_cast<StoreInst>(*II->user_begin());
        if (!SI)
          continue;
        assert(SI->getValueOperand() == II &&
               "expected value operand to be the resource handle");

        GlobalVariable *GV = dyn_cast<GlobalVariable>(SI->getPointerOperand());
        if (!GV || GV->hasExternalLinkage())
          continue;

        if (GV->hasOneUser()) {
          assert(*GV->user_begin() == SI &&
                 "expected single user to be the store instruction");
          DeadInstr.push_back(SI);
          DeadInstr.push_back(II);
          DeadGlobals.insert(GV);
        }
      }
    }
  }

  if (DeadInstr.empty())
    return false;

  for (auto *Instr : DeadInstr) {
    if (auto *II = dyn_cast<IntrinsicInst>(Instr)) {
      assert(II->getIntrinsicID() == Intrinsic::dx_resource_handlefrombinding ||
             II->getIntrinsicID() ==
                 Intrinsic::dx_resource_handlefromimplicitbinding);
      const unsigned ResourceNameOpIndex = 4;
      GlobalVariable *ResourceName = dyn_cast_or_null<GlobalVariable>(
          II->getArgOperand(ResourceNameOpIndex));
      if (ResourceName)
        DeadGlobals.insert(ResourceName);
    }
    Instr->eraseFromParent();
  }

  for (auto *GV : DeadGlobals)
    if (GV->use_empty())
      GV->eraseFromParent();

  return true;
}

PreservedAnalyses DXILRemoveUnusedResources::run(Function &F,
                                                 FunctionAnalysisManager &AM) {
  removeUnusedResources(F);
  return PreservedAnalyses::all();
}

namespace {
class DXILRemoveUnusedResourcesLegacy : public FunctionPass {
public:
  bool runOnFunction(Function &F) override { return removeUnusedResources(F); }
  StringRef getPassName() const override {
    return "DXIL Remove Unused Resources";
  }
  DXILRemoveUnusedResourcesLegacy() : FunctionPass(ID) {}

  static char ID; // Pass identification.
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
};
char DXILRemoveUnusedResourcesLegacy::ID = 0;
} // end anonymous namespace

INITIALIZE_PASS_BEGIN(DXILRemoveUnusedResourcesLegacy, DEBUG_TYPE,
                      "DXIL Remove Unused Resources", false, false)
INITIALIZE_PASS_DEPENDENCY(DXILResourceTypeWrapperPass)
INITIALIZE_PASS_END(DXILRemoveUnusedResourcesLegacy, DEBUG_TYPE,
                    "DXIL Remove Unused Resources", false, false)

FunctionPass *llvm::createDXILRemoveUnusedResourcesLegacyPass() {
  return new DXILRemoveUnusedResourcesLegacy();
}
