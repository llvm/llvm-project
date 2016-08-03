//===- CoroSplit.cpp - Converts a coroutine into a state machine ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This pass builds the coroutine frame and outlines resume and destroy parts
// of the coroutine into separate functions.
//===----------------------------------------------------------------------===//

#include "CoroInternal.h"
#include "llvm/Analysis/CallGraphSCCPass.h"

using namespace llvm;

#define DEBUG_TYPE "coro-split"

//===----------------------------------------------------------------------===//
//                              Top Level Driver
//===----------------------------------------------------------------------===//

namespace {

struct CoroSplit : public CallGraphSCCPass {
  static char ID; // Pass identification, replacement for typeid
  CoroSplit() : CallGraphSCCPass(ID) {}

  bool runOnSCC(CallGraphSCC &SCC) override { return false; }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    CallGraphSCCPass::getAnalysisUsage(AU);
  }
};

}

char CoroSplit::ID = 0;
INITIALIZE_PASS(
    CoroSplit, "coro-split",
    "Split coroutine into a set of functions driving its state machine", false,
    false)

Pass *llvm::createCoroSplitPass() { return new CoroSplit(); }
