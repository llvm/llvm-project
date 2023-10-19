//===- WebAssemblyTargetMachine.cpp - Define TargetMachine for WebAssembly -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "WebAssembly.h"
#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/StackSafetyAnalysis.h"
#include "llvm/Transforms/Utils/MemoryTaggingSupport.h"

using namespace llvm;

namespace {

struct WebAssemblyStackTaggingPass : public FunctionPass {
  static char ID;
  StackSafetyGlobalInfo const *SSI = nullptr;
  AAResults *AA = nullptr;
  WebAssemblyStackTaggingPass() : FunctionPass(ID) {}

  bool runOnFunction(Function &) override;
private:
#if 0
  bool const UseStackSafety = false;
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

bool WebAssemblyStackTaggingPass::runOnFunction(Function & Fn) {
  if (!Fn.hasFnAttribute(Attribute::SanitizeMemTag))
    return false;
#if 0
  SSI = &getAnalysis<StackSafetyGlobalInfoWrapperPass>().getResult();
  return true;
#endif
  return false;
}

char WebAssemblyStackTaggingPass::ID = 0;
#if 0
INITIALIZE_PASS_BEGIN(WebAssemblyStackTaggingPass, DEBUG_TYPE, "WebAssembly Stack Tagging",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(StackSafetyGlobalInfoWrapperPass)
INITIALIZE_PASS_END(WebAssemblyStackTaggingPass, DEBUG_TYPE, "WebAssembly Stack Tagging",
                    false, false)
#endif

void llvm::initializeWebAssemblyStackTaggingPass(PassRegistry &) {

}

FunctionPass *llvm::createWebAssemblyStackTaggingPass() {
  return new WebAssemblyStackTaggingPass();
}
