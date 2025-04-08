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
  SmallVector<Function *> Funcs;

  // Collect non-entry and non-exported functions to set to internal linkage.
  for (Function &EF : M.functions()) {
    if (EF.isIntrinsic())
      continue;
    if (EF.hasFnAttribute("hlsl.shader") || EF.hasFnAttribute("hlsl.export"))
      continue;
    Funcs.push_back(&EF);
  }

  for (Function *F : Funcs) {
    if (F->getLinkage() == GlobalValue::ExternalLinkage)
      F->setLinkage(GlobalValue::InternalLinkage);
    if (F->isDefTriviallyDead())
      M.getFunctionList().erase(F);
  }

  // Do a pass over intrinsics that are no longer used and remove them.
  Funcs.clear();
  for (Function &F : M.functions())
    if (F.isIntrinsic() && F.use_empty())
      Funcs.push_back(&F);
  for (Function *F : Funcs)
    F->eraseFromParent();

  return false;
}

PreservedAnalyses HLSLFinalizeLinkage::run(Module &M,
                                           ModuleAnalysisManager &AM) {
  if (finalizeLinkage(M))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}