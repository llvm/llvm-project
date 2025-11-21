//===- DXILFinalizeLinkage.cpp - Finalize linkage of functions ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DXILFinalizeLinkage.h"
#include "DirectX.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"

#define DEBUG_TYPE "dxil-finalize-linkage"

using namespace llvm;

static bool finalizeLinkage(Module &M) {
  bool MadeChange = false;

  // Convert private globals and external globals with no usage to internal
  // linkage.
  for (GlobalVariable &GV : M.globals()) {
    GV.removeDeadConstantUsers();
    if (GV.hasPrivateLinkage() || (GV.hasExternalLinkage() && GV.use_empty())) {
      GV.setLinkage(GlobalValue::InternalLinkage);
      MadeChange = true;
    }
  }

  SmallVector<Function *> Funcs;

  // Collect non-entry and non-exported functions to set to internal linkage.
  for (Function &EF : M.functions()) {
    if (EF.isIntrinsic())
      continue;
    if (EF.hasExternalLinkage() && EF.hasDefaultVisibility())
      continue;
    if (EF.hasFnAttribute("hlsl.shader"))
      continue;
    Funcs.push_back(&EF);
  }

  for (Function *F : Funcs) {
    if (F->getLinkage() == GlobalValue::ExternalLinkage) {
      F->setLinkage(GlobalValue::InternalLinkage);
      MadeChange = true;
    }
    if (F->isDefTriviallyDead()) {
      M.getFunctionList().erase(F);
      MadeChange = true;
    }
  }

  return MadeChange;
}

PreservedAnalyses DXILFinalizeLinkage::run(Module &M,
                                           ModuleAnalysisManager &AM) {
  if (finalizeLinkage(M))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}

bool DXILFinalizeLinkageLegacy::runOnModule(Module &M) {
  return finalizeLinkage(M);
}

char DXILFinalizeLinkageLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(DXILFinalizeLinkageLegacy, DEBUG_TYPE,
                      "DXIL Finalize Linkage", false, false)
INITIALIZE_PASS_END(DXILFinalizeLinkageLegacy, DEBUG_TYPE,
                    "DXIL Finalize Linkage", false, false)

ModulePass *llvm::createDXILFinalizeLinkageLegacyPass() {
  return new DXILFinalizeLinkageLegacy();
}
