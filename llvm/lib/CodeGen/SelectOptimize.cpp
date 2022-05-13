//===--- SelectOptimize.cpp - Convert select to branches if profitable ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass converts selects to conditional jumps when profitable.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Function.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

using namespace llvm;

namespace {

class SelectOptimize : public FunctionPass {
public:
  static char ID;
  SelectOptimize() : FunctionPass(ID) {
    initializeSelectOptimizePass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {}
};
} // namespace

char SelectOptimize::ID = 0;
INITIALIZE_PASS(SelectOptimize, "select-optimize", "Optimize selects", false,
                false)

FunctionPass *llvm::createSelectOptimizePass() { return new SelectOptimize(); }

bool SelectOptimize::runOnFunction(Function &F) {
  llvm_unreachable("Unimplemented");
}
