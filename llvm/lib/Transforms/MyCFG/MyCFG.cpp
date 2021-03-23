//===-- HelloWorld.cpp - Example Transformations --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/MyCFG/MyCFG.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/BreadthFirstIterator.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/CFGPrinter.h"
#include "llvm/Analysis/HeatUtils.h"

using namespace llvm;
PreservedAnalyses MyCFGPass::run(Function &F, FunctionAnalysisManager &AM) {
  outs() << "===============================================\n";
  outs() << "Basic blocks of " << F.getName() << " in bf-order:\n";
  for (bf_iterator<BasicBlock *> I = bf_begin(&F.getEntryBlock()),
                                 IE = bf_end(&F.getEntryBlock());
       I != IE; ++I) {
    outs() << "  " << (*I)->getName() << "\n";
  }

  outs() << "===============================================\n";
  outs() << "Basic blocks of " << F.getName() << " in post-order:\n";
  for (po_iterator<BasicBlock *> I = po_begin(&F.getEntryBlock()),
                                 IE = po_end(&F.getEntryBlock());
       I != IE; ++I) {
    outs() << "  " << (*I)->getName() << "\n";
  }

  // Use LLVM's Strongly Connected Components (SCCs) iterator to produce
  // a reverse topological sort of SCCs.
  outs() << "===============================================\n";
  outs() << "SCCs for " << F.getName() << " in post-order:\n";
  for (scc_iterator<Function *> I = scc_begin(&F), IE = scc_end(&F); I != IE;
       ++I) {
    // Obtain the vector of BBs in this SCC and print it out.
    const std::vector<BasicBlock *> &SCCBBs = *I;
    outs() << "  SCC: ";
    for (std::vector<BasicBlock *>::const_iterator BBI = SCCBBs.begin(),
                                                   BBIE = SCCBBs.end();
         BBI != BBIE; ++BBI) {
      outs() << (*BBI)->getName() << "  ";
    }
    outs() << "\n";
  }

  outs() << "===============================================\n";
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

  outs() << "\n\n";
  outs() << "===============================================\n";
  outs() << "Trying GrapTraits #######################\n";
  auto *BFI = &AM.getResult<BlockFrequencyAnalysis>(F);
  auto *BPI = &AM.getResult<BranchProbabilityAnalysis>(F);

  DOTFuncInfo CFGInfo(&F, BFI, BPI, getMaxFreq(F, BFI));
  WriteGraph(outs(), &CFGInfo);

  outs() << "\n\n";
  outs() << "===============================================\n";
  outs() << "Trying to be customized GrapTraits #######################\n";
  GraphHelper<DOTFuncInfo*>::wg(outs(), &CFGInfo);

  return PreservedAnalyses::all();
}
