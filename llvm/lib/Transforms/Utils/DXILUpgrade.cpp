//===- DXILUpgrade.cpp - Upgrade DXIL metadata to LLVM constructs ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/DXILUpgrade.h"

using namespace llvm;

static bool handleValVerMetadata(Module &M) {
  NamedMDNode *ValVer = M.getNamedMetadata("dx.valver");
  if (!ValVer)
    return false;

  // We don't need the validation version internally, so we drop it.
  ValVer->dropAllReferences();
  ValVer->eraseFromParent();
  return true;
}

PreservedAnalyses DXILUpgradePass::run(Module &M, ModuleAnalysisManager &AM) {
  PreservedAnalyses PA;
  // We never add, remove, or change functions here.
  PA.preserve<FunctionAnalysisManagerModuleProxy>();
  PA.preserveSet<AllAnalysesOn<Function>>();

  bool Changed = false;
  Changed |= handleValVerMetadata(M);

  if (!Changed)
    return PreservedAnalyses::all();
  return PA;
}
