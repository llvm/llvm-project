//===- AMDGPUMarkPromotablePrivate.cpp - mark private promotable -- ==========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass examines private-address-space (addrspace(5)) alloca instructions
/// and marks those of them that can safely allocate their objects in VGPRs, to
/// be then accessed using VGPR indexing.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUMemoryUtils.h"
#include "AMDGPUTargetMachine.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-mark-promotable-private"

namespace {
class AMDGPUMarkPromotablePrivate {
public:
  AMDGPUMarkPromotablePrivate() {}

  bool runOnFunction(Function &F);
};

bool AMDGPUMarkPromotablePrivate::runOnFunction(Function &F) {
  const DataLayout &DL = F.getParent()->getDataLayout();
  MDNode *PrivateInVGPRMD = MDNode::get(F.getContext(), {});
  bool Changed = false;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      DenseSet<Value *> Pointers;
      if (auto *AI = dyn_cast<AllocaInst>(&I);
          AI && AMDGPU::IsPromotableToVGPR(*AI, DL, Pointers)) {
        AI->setMetadata("amdgpu.promotable.to.vgpr", PrivateInVGPRMD);
        // Set the metadata for all the pointers to this alloca
        // to facilitate the promotion to VGPR during Instruction selection.
        for (Value *Ptr : Pointers) {
          if (auto *Inst = dyn_cast<Instruction>(Ptr))
            Inst->setMetadata("amdgpu.promotable.to.vgpr", PrivateInVGPRMD);
        }
        Changed = true;
      }
    }
  }
  return Changed;
}

class AMDGPUMarkPromotablePrivateLegacy : public FunctionPass {
public:
  static char ID;

  AMDGPUMarkPromotablePrivateLegacy() : FunctionPass(ID) {
    initializeAMDGPUMarkPromotablePrivateLegacyPass(
        *PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {}

  bool runOnFunction(Function &F) override {
    return AMDGPUMarkPromotablePrivate().runOnFunction(F);
  }
};

} // namespace

char AMDGPUMarkPromotablePrivateLegacy::ID = 0;

char &llvm::AMDGPUMarkPromotablePrivateLegacyPassID =
    AMDGPUMarkPromotablePrivateLegacy::ID;

INITIALIZE_PASS_BEGIN(AMDGPUMarkPromotablePrivateLegacy, DEBUG_TYPE,
                      "Mark promotable private objects", false, false)
INITIALIZE_PASS_END(AMDGPUMarkPromotablePrivateLegacy, DEBUG_TYPE,
                    "Mark promotable private objects", false, false)

FunctionPass *llvm::createAMDGPUMarkPromotablePrivateLegacyPass() {
  return new AMDGPUMarkPromotablePrivateLegacy();
}

PreservedAnalyses
AMDGPUMarkPromotablePrivatePass::run(Function &F, FunctionAnalysisManager &) {
  return AMDGPUMarkPromotablePrivate().runOnFunction(F)
             ? PreservedAnalyses::none()
             : PreservedAnalyses::all();
}
