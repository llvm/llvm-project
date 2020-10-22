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
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Instructions.h"
#include <iterator>
#include <vector>

using namespace llvm;

/// Removes all the Defined Functions
/// that aren't inside any of the desired Chunks.
static void extractFunctionsFromModule(const std::vector<Chunk> &ChunksToKeep,
                                       Module *Program) {
  Oracle O(ChunksToKeep);

  // Record all out-of-chunk functions.
  std::vector<std::reference_wrapper<Function>> FuncsToRemove;
  copy_if(Program->functions(), std::back_inserter(FuncsToRemove),
          [&O](Function &F) {
            // Intrinsics don't have function bodies that are useful to
            // reduce. Additionally, intrinsics may have additional operand
            // constraints.
            return !F.isIntrinsic() && !O.shouldKeep();
          });

  // Then, drop body of each of them. We want to batch this and do nothing else
  // here so that minimal number of remaining exteranal uses will remain.
  for (Function &F : FuncsToRemove)
    F.dropAllReferences();

  // And finally, we can actually delete them.
  for (Function &F : FuncsToRemove) {
    // Replace all *still* remaining uses with undef.
    F.replaceAllUsesWith(UndefValue::get(F.getType()));
    // And finally, fully drop it.
    F.eraseFromParent();
  }
}

/// Counts the amount of non-declaration functions and prints their
/// respective name & index
static int countFunctions(Module *Program) {
  // TODO: Silence index with --quiet flag
  errs() << "----------------------------\n";
  errs() << "Function Index Reference:\n";
  int FunctionCount = 0;
  for (auto &F : *Program) {
    if (F.isIntrinsic())
      continue;

    errs() << '\t' << ++FunctionCount << ": " << F.getName() << '\n';
  }

  errs() << "----------------------------\n";
  return FunctionCount;
}

void llvm::reduceFunctionsDeltaPass(TestRunner &Test) {
  errs() << "*** Reducing Functions...\n";
  int Functions = countFunctions(Test.getProgram());
  runDeltaPass(Test, Functions, extractFunctionsFromModule);
  errs() << "----------------------------\n";
}
