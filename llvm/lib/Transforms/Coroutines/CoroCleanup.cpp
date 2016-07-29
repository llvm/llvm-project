//===- CoroCleanup.cpp - Coroutine Cleanup Pass ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This pass lowers all remaining coroutine intrinsics.
//===----------------------------------------------------------------------===//

#include "CoroInternal.h"
#include "llvm/Pass.h"

using namespace llvm;

#define DEBUG_TYPE "coro-cleanup"

//===----------------------------------------------------------------------===//
//                              Top Level Driver
//===----------------------------------------------------------------------===//

namespace {

struct CoroCleanup : FunctionPass {
  static char ID; // Pass identification, replacement for typeid

  CoroCleanup() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override { return false; }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
};

}

char CoroCleanup::ID = 0;
INITIALIZE_PASS(CoroCleanup, "coro-cleanup",
                "Lower all coroutine related intrinsics", false, false)

Pass *llvm::createCoroCleanupPass() { return new CoroCleanup(); }
