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

using namespace llvm;

namespace {

struct WebAssemblyStackTaggingPass : public FunctionPass {
  static char ID;
  WebAssemblyStackTaggingPass() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
#if 0
    errs() << "Hello: ";
    errs().write_escaped(F.getName()) << '\n';
#endif
    return false;
  }
}; // end of struct Hello

}

char WebAssemblyStackTaggingPass::ID = 0;

void llvm::initializeWebAssemblyStackTaggingPass(PassRegistry &)
{

}

FunctionPass *llvm::createWebAssemblyStackTaggingPass() {
  return new WebAssemblyStackTaggingPass();
}
