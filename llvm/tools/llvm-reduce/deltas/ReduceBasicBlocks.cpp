//===- ReduceArguments.cpp - Specialized Delta Pass -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function which calls the Generic Delta pass in order
// to reduce uninteresting Arguments from defined functions.
//
//===----------------------------------------------------------------------===//

#include "ReduceBasicBlocks.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>

using namespace llvm;

/// Replaces BB Terminator with one that only contains Chunk BBs
static void replaceBranchTerminator(BasicBlock &BB,
                                    std::set<BasicBlock *> BBsToKeep) {
  auto Term = BB.getTerminator();
  std::vector<BasicBlock *> ChunkSucessors;
  for (auto Succ : successors(&BB))
    if (BBsToKeep.count(Succ))
      ChunkSucessors.push_back(Succ);

  // BB only references Chunk BBs
  if (ChunkSucessors.size() == Term->getNumSuccessors())
    return;

  bool IsBranch = isa<BranchInst>(Term) || isa<InvokeInst>(Term);
  Value *Address = nullptr;
  if (auto IndBI = dyn_cast<IndirectBrInst>(Term))
    Address = IndBI->getAddress();

  Term->replaceAllUsesWith(UndefValue::get(Term->getType()));
  Term->eraseFromParent();

  if (ChunkSucessors.empty()) {
    auto *FnRetTy = BB.getParent()->getReturnType();
    ReturnInst::Create(BB.getContext(),
                       FnRetTy->isVoidTy() ? nullptr : UndefValue::get(FnRetTy),
                       &BB);
    return;
  }

  if (IsBranch)
    BranchInst::Create(ChunkSucessors[0], &BB);

  if (Address) {
    auto NewIndBI =
        IndirectBrInst::Create(Address, ChunkSucessors.size(), &BB);
    for (auto Dest : ChunkSucessors)
      NewIndBI->addDestination(Dest);
  }
}

/// Removes uninteresting BBs from switch, if the default case ends up being
/// uninteresting, the switch is replaced with a void return (since it has to be
/// replace with something)
static void removeUninterestingBBsFromSwitch(SwitchInst &SwInst,
                                             std::set<BasicBlock *> BBsToKeep) {
  if (!BBsToKeep.count(SwInst.getDefaultDest())) {
    auto *FnRetTy = SwInst.getParent()->getParent()->getReturnType();
    ReturnInst::Create(SwInst.getContext(),
                       FnRetTy->isVoidTy() ? nullptr : UndefValue::get(FnRetTy),
                       SwInst.getParent());
    SwInst.eraseFromParent();
  } else
    for (int I = 0, E = SwInst.getNumCases(); I != E; ++I) {
      auto Case = SwInst.case_begin() + I;
      if (!BBsToKeep.count(Case->getCaseSuccessor())) {
        SwInst.removeCase(Case);
        --I;
        --E;
      }
    }
}

/// Removes out-of-chunk arguments from functions, and modifies their calls
/// accordingly. It also removes allocations of out-of-chunk arguments.
static void extractBasicBlocksFromModule(std::vector<Chunk> ChunksToKeep,
                                         Module *Program) {
  Oracle O(ChunksToKeep);

  std::set<BasicBlock *> BBsToKeep;

  for (auto &F : *Program)
    for (auto &BB : F)
      if (O.shouldKeep())
        BBsToKeep.insert(&BB);

  std::vector<BasicBlock *> BBsToDelete;
  for (auto &F : *Program)
    for (auto &BB : F) {
      if (!BBsToKeep.count(&BB)) {
        BBsToDelete.push_back(&BB);
        // Remove out-of-chunk BB from successor phi nodes
        for (auto *Succ : successors(&BB))
          Succ->removePredecessor(&BB);
      }
    }

  // Replace terminators that reference out-of-chunk BBs
  for (auto &F : *Program)
    for (auto &BB : F) {
      if (auto *SwInst = dyn_cast<SwitchInst>(BB.getTerminator()))
        removeUninterestingBBsFromSwitch(*SwInst, BBsToKeep);
      else
        replaceBranchTerminator(BB, BBsToKeep);
    }

  // Replace out-of-chunk switch uses
  for (auto &BB : BBsToDelete) {
    // Instructions might be referenced in other BBs
    for (auto &I : *BB)
      I.replaceAllUsesWith(UndefValue::get(I.getType()));
    BB->eraseFromParent();
  }
}

/// Counts the amount of basic blocks and prints their name & respective index
static int countBasicBlocks(Module *Program) {
  // TODO: Silence index with --quiet flag
  outs() << "----------------------------\n";
  int BBCount = 0;
  for (auto &F : *Program)
    for (auto &BB : F) {
      if (BB.hasName())
        outs() << "\t" << ++BBCount << ": " << BB.getName() << "\n";
      else
        outs() << "\t" << ++BBCount << ": Unnamed\n";
    }

  return BBCount;
}

void llvm::reduceBasicBlocksDeltaPass(TestRunner &Test) {
  outs() << "*** Reducing Basic Blocks...\n";
  int BBCount = countBasicBlocks(Test.getProgram());
  runDeltaPass(Test, BBCount, extractBasicBlocksFromModule);
}
