//===- HLSLFinalizeLinkage.cpp - Finalize linkage of functions ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/HLSL/HLSLFinalizeLinkage.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Module.h"

#define DEBUG_TYPE "dxil-finalize-linkage"

using namespace llvm;

static bool finalizeLinkage(Module &M) {
  SmallPtrSet<Function *, 8> Funcs;

  for(auto &Var : M.globals()) {
    if (Var.getLinkage() == GlobalValue::ExternalLinkage) {
      Var.setLinkage(GlobalValue::InternalLinkage);
    }
  }

  // Collect non-entry and non-exported functions to set to internal linkage.
  for (Function &EF : M.functions()) {
    if (EF.isIntrinsic())
      continue;
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

PreservedAnalyses HLSLFinalizeLinkage::run(Module &M,
                                           ModuleAnalysisManager &AM) {
  if (finalizeLinkage(M))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
