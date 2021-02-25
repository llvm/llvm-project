//===-- HelloWorld.cpp - Example Transformations --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/MyCFG/MyCFG.h"

using namespace llvm;

PreservedAnalyses MyCFGPass::run(Function &F, FunctionAnalysisManager &AM) {
  outs() << "===============================================\n";
  outs() << "Function: " << F.getName() << "\n";
  outs() << "Function instruction count: " << F.getInstructionCount() << "\n";
  
  for (const auto &bb : F) {
    outs() << "Processing Node " << static_cast<const void*>(&bb) << "\n";

    for (auto it = GraphTraits<const BasicBlock *>::child_begin(&bb), end = GraphTraits<const BasicBlock *>::child_end(&bb); it != end; it++) {
      outs() << static_cast<const void*>(&bb) << " to " << static_cast<const void*>(&it) <<  "\n";
    }
  }

  return PreservedAnalyses::all();
}
