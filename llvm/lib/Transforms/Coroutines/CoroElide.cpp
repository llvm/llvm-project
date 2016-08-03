//===- CoroElide.cpp - Coroutine Frame Allocation Elision Pass ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This pass replaces dynamic allocation of coroutine frame with alloca and
// replaces calls to llvm.coro.resume and llvm.coro.destroy with direct calls
// to coroutine sub-functions.
//===----------------------------------------------------------------------===//

#include "CoroInternal.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Pass.h"

using namespace llvm;

#define DEBUG_TYPE "coro-elide"

//===----------------------------------------------------------------------===//
//                              Top Level Driver
//===----------------------------------------------------------------------===//

namespace {

struct CoroElide : FunctionPass {
  static char ID;
  CoroElide() : FunctionPass(ID) {}
  bool runOnFunction(Function &F) override { return false; }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
};

}

char CoroElide::ID = 0;
INITIALIZE_PASS_BEGIN(
    CoroElide, "coro-elide",
    "Coroutine frame allocation elision and indirect calls replacement", false,
    false)
INITIALIZE_PASS_END(
    CoroElide, "coro-elide",
    "Coroutine frame allocation elision and indirect calls replacement", false,
    false)

Pass *llvm::createCoroElidePass() { return new CoroElide(); }
