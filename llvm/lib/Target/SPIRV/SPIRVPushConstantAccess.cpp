//===- SPIRVPushConstantAccess.cpp - Translate CBuffer Loads ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass changes the types of all the globals in the PushConstant
// address space into a target extension type, and makes all references
// to this global go though a custom SPIR-V intrinsic.
//
// This allows the backend to properly lower the push constant struct type
// to a fully laid out type, and generate the proper OpAccessChain.
//
//===----------------------------------------------------------------------===//

#include "SPIRVPushConstantAccess.h"
#include "SPIRV.h"
#include "SPIRVSubtarget.h"
#include "SPIRVTargetMachine.h"
#include "SPIRVUtils.h"
#include "llvm/Frontend/HLSL/CBuffer.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ReplaceConstant.h"

#define DEBUG_TYPE "spirv-pushconstant-access"
using namespace llvm;

static bool replacePushConstantAccesses(Module &M, SPIRVGlobalRegistry *GR) {
  SmallVector<GlobalVariable *> PushConstants;
  for (GlobalVariable &GV : M.globals()) {
    if (GV.getAddressSpace() !=
        storageClassToAddressSpace(SPIRV::StorageClass::PushConstant))
      continue;

    GV.removeDeadConstantUsers();
    PushConstants.push_back(&GV);
  }

  for (GlobalVariable *GV : PushConstants) {
    Type *PCType = llvm::TargetExtType::get(
        M.getContext(), "spirv.PushConstant", {GV->getValueType()});
    GlobalVariable *NewGV = new GlobalVariable(
        M, PCType, GV->isConstant(), GV->getLinkage(),
        /* initializer= */ nullptr, GV->getName(),
        /* InsertBefore= */ GV, GV->getThreadLocalMode(), GV->getAddressSpace(),
        GV->isExternallyInitialized());

    SmallVector<User *, 4> Users(GV->user_begin(), GV->user_end());
    for (User *U : Users) {
      Instruction *I = cast<Instruction>(U);
      IRBuilder<> Builder(I);
      Value *GetPointerCall = Builder.CreateIntrinsic(
          NewGV->getType(), Intrinsic::spv_pushconstant_getpointer, {NewGV});
      GR->buildAssignPtr(Builder, GV->getValueType(), GetPointerCall);

      for (unsigned N = 0; N < I->getNumOperands(); ++N) {
        if (I->getOperand(N) == GV)
          I->setOperand(N, GetPointerCall);
      }
    }

    GV->eraseFromParent();
  }

  return true;
}

PreservedAnalyses SPIRVPushConstantAccess::run(Module &M,
                                               ModuleAnalysisManager &AM) {
  const SPIRVSubtarget *ST = TM.getSubtargetImpl();
  SPIRVGlobalRegistry *GR = ST->getSPIRVGlobalRegistry();
  return replacePushConstantAccesses(M, GR) ? PreservedAnalyses::none()
                                            : PreservedAnalyses::all();
}

namespace {
class SPIRVPushConstantAccessLegacy : public ModulePass {
  SPIRVTargetMachine *TM = nullptr;

public:
  bool runOnModule(Module &M) override {
    const SPIRVSubtarget *ST = TM->getSubtargetImpl();
    SPIRVGlobalRegistry *GR = ST->getSPIRVGlobalRegistry();
    return replacePushConstantAccesses(M, GR);
  }
  StringRef getPassName() const override {
    return "SPIRV push constant Access";
  }
  SPIRVPushConstantAccessLegacy(SPIRVTargetMachine *TM)
      : ModulePass(ID), TM(TM) {}

  static char ID; // Pass identification.
};
char SPIRVPushConstantAccessLegacy::ID = 0;
} // end anonymous namespace

INITIALIZE_PASS(SPIRVPushConstantAccessLegacy, DEBUG_TYPE,
                "SPIRV push constant Access", false, false)

ModulePass *
llvm::createSPIRVPushConstantAccessLegacyPass(SPIRVTargetMachine *TM) {
  return new SPIRVPushConstantAccessLegacy(TM);
}
