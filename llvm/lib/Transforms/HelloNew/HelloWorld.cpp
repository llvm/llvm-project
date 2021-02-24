//===-- HelloWorld.cpp - Example Transformations --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/HelloNew/HelloWorld.h"

using namespace llvm;

PreservedAnalyses HelloWorldPass::run(Function &F,
                                      FunctionAnalysisManager &AM) {
  outs() << "Function: " << F.getName() << "\n";

  for (auto &BB : F.getBasicBlockList()) {
    outs() << "BB: " << BB.getName() << "\n";
    outs() << "BB ValueID: " << BB.getValueID() << "\n";
    outs() << "BB Parent: " << BB.getParent() << "\n";
    outs() << "BB NextNode: " << BB.getNextNode() << "\n";
    for (auto &I : BB) {
      outs() << "I: " << I.getName() << "\n";
      outs() << "I ValueID: " << I.getValueID() << "\n";
      outs() << "I Parent: " << I.getParent() << "\n";
      outs() << "I NextNode: " << I.getNextNode() << "\n";
    }
  }
  outs() << "\n\n";

  // https://llvm.org/docs/ProgrammersManual.html#iterating-over-the-basicblock-in-a-function
  outs() << "Iterate basic block \n\n";

  for (auto it = F.begin(), et = F.end(); it != et; ++it) {
    errs() << "Basic block (name=" << it->getName() << ") has "
             << it->size() << " instructions.\n";
  }

  return PreservedAnalyses::all();
}
