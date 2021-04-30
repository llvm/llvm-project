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
#include "llvm/IR/InstIterator.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/CFGPrinter.h"
#include "llvm/Analysis/HeatUtils.h"

using namespace llvm;

void traverse(Function &F) {
  outs() << "===============================================\n";
  outs() << "Basic blocks of " << F.getName() << " in df_iterator:\n";
  for (auto iterator = df_begin(&F.getEntryBlock()),
           IE = df_end(&F.getEntryBlock());
       iterator != IE; ++iterator) {
    outs() << *iterator << "\n";
    for (auto &instruction : **iterator) {
      outs() << instruction << "\n";
    }
  }
  outs() << "\n\n";

  outs() << "===============================================\n";
  outs() << "Basic blocks of " << F.getName() << " in idf_iterator:\n";
  for (auto iterator = idf_begin(&F.getEntryBlock()),
           IE = idf_end(&F.getEntryBlock());
       iterator != IE; ++iterator) {
    outs() << *iterator << "\n";
    for (auto &instruction : **iterator) {
      outs() << instruction << "\n";
    }
  }
  outs() << "\n\n";

  outs() << "===============================================\n";
  outs() << "Basic blocks of " << F.getName() << " in bf_iterator:\n";
  for (auto iterator = bf_begin(&F.getEntryBlock()),
           IE = bf_end(&F.getEntryBlock());
       iterator != IE; ++iterator) {
    outs() << *iterator << "\n";
    for (auto &instruction : **iterator) {
      outs() << instruction << "\n";
    }
  }
  outs() << "\n\n";

  outs() << "===============================================\n";
  outs() << "Basic blocks of " << F.getName() << " in po_iterator:\n";
  for (auto iterator = po_begin(&F.getEntryBlock()),
           IE = po_end(&F.getEntryBlock());
       iterator != IE; ++iterator) {
    outs() << *iterator << "\n";
    for (auto &instruction : **iterator) {
      outs() << instruction << "\n";
    }
  }
  outs() << "\n\n";

  outs() << "===============================================\n";
  outs() << "Basic blocks of " << F.getName() << " in pred_iterator:\n";
  for (auto iterator = pred_begin(&F.getEntryBlock()), IE = pred_end(&F.getEntryBlock());
       iterator != IE; ++iterator) {
    outs() << *iterator << "\n";
    for (auto &instruction : **iterator) {
      outs() << instruction << "\n";
    }
  }
  outs() << "\n\n";

  outs() << "===============================================\n";
  outs() << "Basic blocks of " << F.getName() << " in succ_iterator:\n";
  for (auto iterator = succ_begin(&F.getEntryBlock()),
           IE = succ_end(&F.getEntryBlock());
       iterator != IE; ++iterator) {
    outs() << *iterator << "\n";
    for (auto &instruction : **iterator) {
      outs() << instruction << "\n";
    }
  }
  outs() << "\n\n";
}

void traverseBB(Function &F, bool nested) {
  bool isThreadStartCheckpoint = true;
  for (auto &bb : F) {
    outs() << "Basic Block '" << &bb << "' Instructions: '" << bb << "'\n";
    bool isThreadEndCheckpoint = false;
    bool isExitPointCheckpoint = false;
    for (auto &i : bb) {
      // if instruction is last in the block and has no more successor,
      // then this will be thread end checkpoint
      if (i.isTerminator()) {
        if (i.getNumSuccessors() == 0) {
          isThreadEndCheckpoint |= true;
        }
      }
      // Check if instruction is calling a function
      if (isa<CallInst>(i)) {
        auto *call = &cast<CallBase>(i);
        outs() << "Traversing nested function " << call->getCalledFunction()->getName() << " Instruction '" << i << "'\n";
        traverseBB(*call->getCalledFunction(), true);
        outs() << "Finished traversing nested function " << call->getCalledFunction()->getName() << "\n";
        isExitPointCheckpoint |= true;
      }
    }
    if (isThreadStartCheckpoint and !nested) {
      outs() << "This basic block is thread start checkpoint\n";
      isThreadStartCheckpoint = false;
    }
    if (isExitPointCheckpoint) {
      outs() << "This basic block is an exit-point checkpoint\n";
    }
    if (isThreadEndCheckpoint and !nested) {
      outs() << "This basic block is thread end checkpoint\n";
    }
    outs() << "\n\n";
  }
}

PreservedAnalyses MyCFGPass::run(Function &F, FunctionAnalysisManager &AM) {
  outs() << "Function '" << F.getName() << "'\n";
  traverseBB(F, false);
  return PreservedAnalyses::all();
}
