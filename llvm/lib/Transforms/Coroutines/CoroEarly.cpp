//===- CoroEarly.cpp - Coroutine Early Function Pass ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This pass lowers coroutine intrinsics that hide the details of the exact
// calling convention for coroutine resume and destroy functions and details of
// the structure of the coroutine frame.
//===----------------------------------------------------------------------===//

#include "CoroInternal.h"
#include "llvm/Pass.h"

using namespace llvm;

#define DEBUG_TYPE "coro-early"

//===----------------------------------------------------------------------===//
//                              Top Level Driver
//===----------------------------------------------------------------------===//

namespace {

struct CoroEarly : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  CoroEarly() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override { return false; }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
};

}

char CoroEarly::ID = 0;
INITIALIZE_PASS(CoroEarly, "coro-early", "Lower early coroutine intrinsics",
                false, false)

Pass *llvm::createCoroEarlyPass() { return new CoroEarly(); }
