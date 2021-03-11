//===-- HelloWorld.cpp - Example Transformations --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/MyCFG/MyCFG.h"
#include "TopoSorter.cpp"

using namespace llvm;

TopoSorter topo;

PreservedAnalyses MyCFGPass::run(Function &F, FunctionAnalysisManager &AM) {
  outs() << "===============================================\n";
  outs() << "Function: " << F.getName() << "\n";
  outs() << "Instruction count: " << F.getInstructionCount() << "\n";

  for (auto &bb : F) {
    for (auto &ii: bb) {
      outs() << "Instruction: " << ii << "\n";
    }
    auto *ti = bb.getTerminator();
    outs() << "Terminating instruction: " << *ti << "\n";
    for (unsigned I = 0, NSucc = ti->getNumSuccessors(); I < NSucc; ++I) {
      BasicBlock *Succ = ti->getSuccessor(I);
      for (auto &ii: *Succ) {
        outs() << "Instruction in successor: " << ii << "\n";
      }
    }
  }

  outs() << "+++++++++++++++++++++++++++++++++++++++\n";
  topo.runToposort(F);

  return PreservedAnalyses::all();
}
