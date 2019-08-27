//===- ReduceFunctions.cpp - Specialized Delta Pass -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function which calls the Generic Delta pass in order
// to reduce functions (and any instruction that calls it) in the provided
// Module.
//
//===----------------------------------------------------------------------===//

#include "ReduceFunctions.h"

/// Removes all the Defined Functions (as well as their calls)
/// that aren't inside any of the desired Chunks.
static void extractFunctionsFromModule(const std::vector<Chunk> &ChunksToKeep,
                                       Module *Program) {
  // Get functions inside desired chunks
  std::set<Function *> FuncsToKeep;
  unsigned I = 0, FunctionCount = 0;
  for (auto &F : *Program)
    if (I < ChunksToKeep.size()) {
      if (ChunksToKeep[I].contains(++FunctionCount))
        FuncsToKeep.insert(&F);
      if (FunctionCount == ChunksToKeep[I].end)
        ++I;
    }

  // Delete out-of-chunk functions, and replace their calls with undef
  std::vector<Function *> FuncsToRemove;
  std::vector<CallInst *> CallsToRemove;
  for (auto &F : *Program)
    if (!FuncsToKeep.count(&F)) {
      for (auto U : F.users())
        if (auto *Call = dyn_cast<CallInst>(U)) {
          Call->replaceAllUsesWith(UndefValue::get(Call->getType()));
          CallsToRemove.push_back(Call);
        }
      F.replaceAllUsesWith(UndefValue::get(F.getType()));
      FuncsToRemove.push_back(&F);
    }

  for (auto *C : CallsToRemove)
    C->eraseFromParent();

  for (auto *F : FuncsToRemove)
    F->eraseFromParent();
}

/// Counts the amount of non-declaration functions and prints their
/// respective name & index
static unsigned countFunctions(Module *Program) {
  // TODO: Silence index with --quiet flag
  errs() << "----------------------------\n";
  errs() << "Function Index Reference:\n";
  unsigned FunctionCount = 0;
  for (auto &F : *Program)
    errs() << "\t" << ++FunctionCount << ": " << F.getName() << "\n";

  errs() << "----------------------------\n";
  return FunctionCount;
}

void llvm::reduceFunctionsDeltaPass(TestRunner &Test) {
  errs() << "*** Reducing Functions...\n";
  unsigned Functions = countFunctions(Test.getProgram());
  runDeltaPass(Test, Functions, extractFunctionsFromModule);
  errs() << "----------------------------\n";
}