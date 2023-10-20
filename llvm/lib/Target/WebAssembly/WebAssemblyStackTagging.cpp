//===- WebAssemblyTargetMachine.cpp - Define TargetMachine for WebAssembly -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "WebAssembly.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/StackSafetyAnalysis.h"
#include "llvm/Transforms/Utils/MemoryTaggingSupport.h"

using namespace llvm;

#define DEBUG_TYPE "wasm-stack-tagging"

namespace {

struct WebAssemblyStackTagging : public FunctionPass {
  static char ID;
  StackSafetyGlobalInfo const *SSI = nullptr;
  AAResults *AA = nullptr;
  WebAssemblyStackTagging() : FunctionPass(ID) {}

  bool runOnFunction(Function &) override;
private:
#if 1
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<StackSafetyGlobalInfoWrapperPass>();
#if 0
    if (MergeInit)
      AU.addRequired<AAResultsWrapperPass>();
#endif
  }
#endif
}; // end of struct Hello

}

bool WebAssemblyStackTagging::runOnFunction(Function & Fn) {
  if (!Fn.hasFnAttribute(Attribute::SanitizeMemTag))
    return false;
  SSI = &getAnalysis<StackSafetyGlobalInfoWrapperPass>().getResult();
#if 0
  for (auto &I : SInfo.AllocasToInstrument) {
    LLVM_DEBUG(dbgs() << I << "\n");
  }
#endif
  return true;
}

char WebAssemblyStackTagging::ID = 0;
INITIALIZE_PASS_BEGIN(WebAssemblyStackTagging, DEBUG_TYPE, "WebAssembly Stack Tagging",
                      false, false)
#if 0
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
#endif
INITIALIZE_PASS_DEPENDENCY(StackSafetyGlobalInfoWrapperPass)
INITIALIZE_PASS_END(WebAssemblyStackTagging, DEBUG_TYPE, "WebAssembly Stack Tagging",
                    false, false)

FunctionPass *llvm::createWebAssemblyStackTaggingPass() {
  return new WebAssemblyStackTagging();
}
