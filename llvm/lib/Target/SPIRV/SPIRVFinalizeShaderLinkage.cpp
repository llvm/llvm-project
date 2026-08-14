//===- SPIRVFinalizeShaderLinkage.cpp - Finalize shader linkage ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shader-only analogue of DXILFinalizeLinkage: internalizes non-entry,
// non-exported HLSL helper functions and erases the resulting dead ones.
//
//===----------------------------------------------------------------------===//

#include "SPIRVFinalizeShaderLinkage.h"
#include "SPIRV.h"
#include "SPIRVSubtarget.h"
#include "SPIRVTargetMachine.h"
#include "SPIRVUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

#define DEBUG_TYPE "spirv-finalize-shader-linkage"

using namespace llvm;

namespace {

bool finalizeShaderLinkage(const SPIRVTargetMachine &TM, Module &M) {
  //  This pass only applies to shader targets because shaders don't have
  //  linkers.
  if (!TM.getSubtargetImpl()->isShader())
    return false;

  bool Changed = false;

  for (Function &F : M) {
    if (F.isIntrinsic() || F.isDeclaration() || isEntryPoint(F))
      continue;
    if (F.hasExternalLinkage() && !F.hasHiddenVisibility())
      continue;
    if (!F.hasLocalLinkage()) {
      F.setLinkage(GlobalValue::InternalLinkage);
      Changed = true;
    }
  }

  // Erase dead helpers, iterating to a fixpoint for helper-calls-helper chains.
  bool LocalChange = true;
  while (LocalChange) {
    LocalChange = false;
    for (Function &F : make_early_inc_range(M))
      if (F.isDefTriviallyDead()) {
        F.eraseFromParent();
        LocalChange = Changed = true;
      }
  }
  return Changed;
}

class SPIRVFinalizeShaderLinkageLegacy : public ModulePass {
public:
  static char ID;
  SPIRVFinalizeShaderLinkageLegacy(const SPIRVTargetMachine &TM)
      : ModulePass(ID), TM(TM) {}
  StringRef getPassName() const override {
    return "SPIRV Finalize Shader Linkage";
  }
  bool runOnModule(Module &M) override { return finalizeShaderLinkage(TM, M); }

private:
  const SPIRVTargetMachine &TM;
};

} // namespace

PreservedAnalyses SPIRVFinalizeShaderLinkage::run(Module &M,
                                                  ModuleAnalysisManager &AM) {
  return finalizeShaderLinkage(TM, M) ? PreservedAnalyses::none()
                                      : PreservedAnalyses::all();
}

char SPIRVFinalizeShaderLinkageLegacy::ID = 0;

INITIALIZE_PASS(SPIRVFinalizeShaderLinkageLegacy,
                "spirv-finalize-shader-linkage",
                "Finalize SPIR-V shader linkage", false, false)

ModulePass *
llvm::createSPIRVFinalizeShaderLinkagePass(const SPIRVTargetMachine &TM) {
  return new SPIRVFinalizeShaderLinkageLegacy(TM);
}
