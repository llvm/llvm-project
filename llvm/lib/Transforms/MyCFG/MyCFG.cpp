//===-- HelloWorld.cpp - Example Transformations --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/MyCFG/MyCFG.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/CFGPrinter.h"
#include "llvm/Analysis/HeatUtils.h"

using namespace llvm;
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

  outs() << "Trying GrapTraits #######################\n";
  auto *BFI = &AM.getResult<BlockFrequencyAnalysis>(F);
  auto *BPI = &AM.getResult<BranchProbabilityAnalysis>(F);

  DOTFuncInfo CFGInfo(&F, BFI, BPI, getMaxFreq(F, BFI));
  GraphHelper<DOTFuncInfo*>::wg(outs(), &CFGInfo);

  return PreservedAnalyses::all();
}

