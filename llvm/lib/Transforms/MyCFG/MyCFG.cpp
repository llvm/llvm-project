//===-- HelloWorld.cpp - Example Transformations --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/MyCFG/MyCFG.h"
#include "llvm/ADT/BreadthFirstIterator.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/CFGPrinter.h"
#include "llvm/Analysis/HeatUtils.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include <sstream>

using namespace llvm;

void traverseCFG(Function &F) {
  outs() << "===============================================\n";
  outs() << "Basic blocks of " << F.getName() << " in df_iterator:\n";
  for (auto iterator = df_begin(&F.getEntryBlock()),
           IE = df_end(&F.getEntryBlock());
       iterator != IE; ++iterator) {
    outs() << iterator->getName() << "\n";
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
    outs() << iterator->getName() << "\n";
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
    outs() << iterator->getName() << "\n";
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
    outs() << iterator->getName() << "\n";
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
    outs() << iterator->getName() << "\n";
    for (auto &instruction : **iterator) {
      outs() << instruction << "\n";
    }
  }
  outs() << "\n\n";
}

void traverseCheckpoints(Function &F, int nestedLevel) {
  std::string prefix = "";
  for (int i = 0; i < nestedLevel; i++) {
    prefix.append(">>");
  }
  bool isThreadStartCheckpoint = F.getName() == "main";
  for (auto &bb : F) {
    outs() << prefix << "Basic Block '" << &bb << "' Instructions: '" << bb << "'\n";

    bool isThreadEndCheckpoint = false;
    bool isExitPointCheckpoint = false;
    for (auto &i : bb) {
      // if instruction is last in the block and has no more successor,
      // then this will be thread end checkpoint
      if (i.isTerminator()) {
        // Thread end only in the original function (main)
        if (i.getNumSuccessors() == 0 && nestedLevel == 0 && F.getName() == "main") {
          isThreadEndCheckpoint |= true;
        }
      }
      // Check if instruction is calling a function
      if (isa<CallInst>(i)) {
        auto *call = &cast<CallBase>(i);
        // use this hack to check if function is external
        if (call != nullptr && call->getCalledFunction() != nullptr && !call->getCalledFunction()->empty()) {
          outs() << prefix << "Traversing nestedLevel function " << call->getCalledFunction()->getName() << " Instruction '" << i << "'\n";
          traverseCheckpoints(*(call->getCalledFunction()), nestedLevel + 1);
          outs() << prefix << "Finished traversing nestedLevel function " << call->getCalledFunction()->getName() << "\n";
        } else {
          // The function is outside of the translation unit, hence it is an exit point
          if (!isThreadStartCheckpoint && !isThreadEndCheckpoint) {
            isExitPointCheckpoint |= true;
          }
        }
      }
    }

    // Get Identifier from the BB address
    std::ostringstream st;
    st << &bb;

    std::string name = st.str();
    if (isThreadStartCheckpoint) {
      outs() << prefix << "This basic block:'" << name << "' is thread start checkpoint\n";
      name.append(" - Thread Start");
      isThreadStartCheckpoint = false;
    } else if (isThreadEndCheckpoint) {
      name.append(" - Thread End");
      outs() << prefix << "This basic block:'" << name << "' is thread end checkpoint\n";
    } else if (isExitPointCheckpoint) {
      name.append(" - Exit Point");
      outs() << prefix << "This basic block:'" << name << "' is exit-point checkpoint\n";
    }
    bb.setName(name);
    if (!nestedLevel) {
      outs() << "\n\n";
    }
  }
}

void findLoopHeader(Function &F) {
  outs() << "==================================================\n";
  outs() << "Analyzing loop\n";
  DominatorTree* DT = new DominatorTree();
  DT->recalculate(F);
  // generate the LoopInfoBase for the current function
  LoopInfoBase<BasicBlock, Loop>* KLoop = new LoopInfoBase<BasicBlock, Loop>();
  KLoop->releaseMemory();
  KLoop->analyze(*DT);
  for (auto &bb : F) {
    auto loop = KLoop->getLoopFor(&bb);
    if (loop != nullptr) {
      auto loopHeader = loop->getHeader();
      outs() << "Loop header: '" << loopHeader << "' Instructions: '" << *loopHeader << "'\n";

      std::ostringstream st;
      st << &bb;

      std::string name = st.str();
      name.append(" - Virtual Checkpoint");
      loopHeader->setName(name);
    }
  }
}

PreservedAnalyses MyCFGPass::run(Function &F, FunctionAnalysisManager &AM) {
  if (F.getName() == "main") {
    outs() << "==================================================\n";
  }
  outs() << "Function '" << F.getName() << "'\n";
  traverseCheckpoints(F, 0);

  findLoopHeader(F);

  traverseCFG(F);
  return PreservedAnalyses::all();
}
