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
#include "Delta.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/Instructions.h"
#include <set>

using namespace llvm;

/// Removes all the Defined Functions (as well as their calls)
/// that aren't inside any of the desired Chunks.
static void extractFunctionsFromModule(const std::vector<Chunk> &ChunksToKeep,
                                       Module *Program) {
  Oracle O(ChunksToKeep);

  // Get functions inside desired chunks
  std::set<Function *> FuncsToKeep;
  for (auto &F : *Program)
    if (O.shouldKeep())
      FuncsToKeep.insert(&F);

  // Delete out-of-chunk functions, and replace their calls with undef
  std::vector<Function *> FuncsToRemove;
  SetVector<CallInst *> CallsToRemove;
  for (auto &F : *Program)
    if (!FuncsToKeep.count(&F)) {
      for (auto U : F.users())
        if (auto *Call = dyn_cast<CallInst>(U)) {
          Call->replaceAllUsesWith(UndefValue::get(Call->getType()));
          CallsToRemove.insert(Call);
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
static int countFunctions(Module *Program) {
  // TODO: Silence index with --quiet flag
  errs() << "----------------------------\n";
  errs() << "Function Index Reference:\n";
  int FunctionCount = 0;
  for (auto &F : *Program)
    errs() << "\t" << ++FunctionCount << ": " << F.getName() << "\n";

  errs() << "----------------------------\n";
  return FunctionCount;
}

void llvm::reduceFunctionsDeltaPass(TestRunner &Test) {
  errs() << "*** Reducing Functions...\n";
  int Functions = countFunctions(Test.getProgram());
  runDeltaPass(Test, Functions, extractFunctionsFromModule);
  errs() << "----------------------------\n";
}
