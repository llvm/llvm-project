//===- DXILFinalizeLinkage.cpp - Finalize linkage of functions ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DXILFinalizeLinkage.h"
#include "DirectX.h"
#include "llvm/Analysis/DXILResource.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"

#define DEBUG_TYPE "dxil-finalize-linkage"

using namespace llvm;

static bool finalizeLinkage(Module &M) {
  SmallPtrSet<Function *, 8> Funcs;

  // Collect non-entry and non-exported functions to set to internal linkage.
  for (Function &EF : M.functions()) {
    if (EF.hasFnAttribute("hlsl.shader") || EF.hasFnAttribute("hlsl.export"))
      continue;
    Funcs.insert(&EF);
  }

  for (Function *F : Funcs) {
    if (F->getLinkage() == GlobalValue::ExternalLinkage)
      F->setLinkage(GlobalValue::InternalLinkage);
    if (F->isDefTriviallyDead())
      M.getFunctionList().erase(F);
  }

  return false;
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

void DXILFinalizeLinkageLegacy::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addPreserved<DXILResourceWrapperPass>();
}

char DXILFinalizeLinkageLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(DXILFinalizeLinkageLegacy, DEBUG_TYPE,
                      "DXIL Finalize Linkage", false, false)
INITIALIZE_PASS_END(DXILFinalizeLinkageLegacy, DEBUG_TYPE,
                    "DXIL Finalize Linkage", false, false)

ModulePass *llvm::createDXILFinalizeLinkageLegacyPass() {
  return new DXILFinalizeLinkageLegacy();
}
