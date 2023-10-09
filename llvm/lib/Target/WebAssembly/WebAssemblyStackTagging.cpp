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

  bool runOnFunction(Function &) override;
}; // end of struct Hello

}

bool WebAssemblyStackTaggingPass::runOnFunction(Function & Fn) {
  if (!Fn.hasFnAttribute(Attribute::SanitizeMemTag))
    return false;
  return true;
}

char WebAssemblyStackTaggingPass::ID = 0;

void llvm::initializeWebAssemblyStackTaggingPass(PassRegistry &) {

}

FunctionPass *llvm::createWebAssemblyStackTaggingPass() {
  return new WebAssemblyStackTaggingPass();
}
