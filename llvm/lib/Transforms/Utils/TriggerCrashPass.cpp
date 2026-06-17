//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM Exceptions
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/TriggerCrashPass.h"
#include <cstdlib>

using namespace llvm;

PreservedAnalyses TriggerCrashModulePass::run(Module &,
                                              ModuleAnalysisManager &) {
  abort();
  return PreservedAnalyses::all();
}

PreservedAnalyses TriggerCrashFunctionPass::run(Function &,
                                                FunctionAnalysisManager &) {
  abort();
  return PreservedAnalyses::all();
}

namespace {
class TriggerCrashFunctionLegacyPass : public FunctionPass {
public:
  static char ID;
  TriggerCrashFunctionLegacyPass() : FunctionPass(ID) {}
  bool runOnFunction(Function &F) override {
    abort();
    return false;
  }
  StringRef getPassName() const override { return "TriggerCrashFunctionPass"; }
};
} // namespace

char TriggerCrashFunctionLegacyPass::ID = 0;

FunctionPass *llvm::createTriggerCrashFunctionPass() {
  return new TriggerCrashFunctionLegacyPass();
}
