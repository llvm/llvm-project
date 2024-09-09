//===- AMDGPUMarkPromotableLaneShared.cpp - mark lane-shared promotable -- ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass looks over all the lane-shared global variables, and mark those
/// that are suitable for shared-vgpr indexing access.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUMemoryUtils.h"
#include "AMDGPUTargetMachine.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-mark-promotable-lane-shared"

namespace {
class AMDGPUMarkPromotableLaneShared {
public:
  AMDGPUMarkPromotableLaneShared() {}

  bool runOnFunction(Function &F);
};

bool AMDGPUMarkPromotableLaneShared::runOnFunction(Function &F) {
  auto M = F.getParent();
  SmallVector<Constant *> LaneSharedGlobals;
  for (auto &GV : M->globals()) {
    if (GV.getAddressSpace() == AMDGPUAS::LANE_SHARED &&
        !GV.hasAttribute("lane-shared-in-vgpr") &&
        !GV.hasAttribute("lane-shared-in-mem"))
      LaneSharedGlobals.push_back(&GV);
  }
  if (LaneSharedGlobals.empty())
    return false;

  bool Changed = false;
  for (auto *GVC : LaneSharedGlobals) {
    GlobalVariable &GV = *cast<GlobalVariable>(GVC);
    if (AMDGPU::IsPromotableToVGPR(GV, M->getDataLayout())) {
      GV.addAttribute("lane-shared-in-vgpr");
      Changed = true;
    } else
      GV.addAttribute("lane-shared-in-mem");
  }
  return Changed;
}

class AMDGPUMarkPromotableLaneSharedLegacy : public FunctionPass {
public:
  static char ID;

  AMDGPUMarkPromotableLaneSharedLegacy() : FunctionPass(ID) {
    initializeAMDGPUMarkPromotableLaneSharedLegacyPass(
        *PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {}

  bool runOnFunction(Function &F) override {
    return AMDGPUMarkPromotableLaneShared().runOnFunction(F);
  }
};

} // namespace

char AMDGPUMarkPromotableLaneSharedLegacy::ID = 0;

char &llvm::AMDGPUMarkPromotableLaneSharedLegacyPassID =
    AMDGPUMarkPromotableLaneSharedLegacy::ID;

INITIALIZE_PASS_BEGIN(AMDGPUMarkPromotableLaneSharedLegacy, DEBUG_TYPE,
                      "Mark promotable lane-shared", false, false)
INITIALIZE_PASS_END(AMDGPUMarkPromotableLaneSharedLegacy, DEBUG_TYPE,
                    "Mark promotable lane-shared", false, false)

FunctionPass *llvm::createAMDGPUMarkPromotableLaneSharedLegacyPass() {
  return new AMDGPUMarkPromotableLaneSharedLegacy();
}

PreservedAnalyses
AMDGPUMarkPromotableLaneSharedPass::run(Function &F,
                                        FunctionAnalysisManager &) {
  return AMDGPUMarkPromotableLaneShared().runOnFunction(F)
             ? PreservedAnalyses::none()
             : PreservedAnalyses::all();
}
