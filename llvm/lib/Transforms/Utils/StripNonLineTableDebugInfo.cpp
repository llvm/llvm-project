//===- StripNonLineTableDebugInfo.cpp -- Strip parts of Debug Info --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DebugInfo.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils.h"
using namespace llvm;

namespace {

/// This pass strips all debug info that is not related line tables.
/// The result will be the same as if the program where compiled with
/// -gline-tables-only.
struct StripNonLineTableDebugInfo : public ModulePass {
  static char ID; // Pass identification, replacement for typeid
  StripNonLineTableDebugInfo() : ModulePass(ID) {
    initializeStripNonLineTableDebugInfoPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  bool runOnModule(Module &M) override {
    return llvm::stripNonLineTableDebugInfo(M);
  }
};
}

char StripNonLineTableDebugInfo::ID = 0;
INITIALIZE_PASS(StripNonLineTableDebugInfo, "strip-nonlinetable-debuginfo",
                "Strip all debug info except linetables", false, false)

ModulePass *llvm::createStripNonLineTableDebugInfoPass() {
  return new StripNonLineTableDebugInfo();
}
