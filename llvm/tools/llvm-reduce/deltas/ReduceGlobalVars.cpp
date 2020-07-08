//===- ReduceGlobalVars.cpp - Specialized Delta Pass ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function which calls the Generic Delta pass in order
// to reduce initialized Global Variables in the provided Module.
//
//===----------------------------------------------------------------------===//

#include "ReduceGlobalVars.h"
#include "llvm/IR/Constants.h"
#include <set>

using namespace llvm;

/// Removes all the Initialized GVs that aren't inside the desired Chunks.
static void extractGVsFromModule(std::vector<Chunk> ChunksToKeep,
                                 Module *Program) {
  Oracle O(ChunksToKeep);

  // Get GVs inside desired chunks
  std::set<GlobalVariable *> GVsToKeep;
  for (auto &GV : Program->globals())
    if (GV.hasInitializer() && O.shouldKeep())
      GVsToKeep.insert(&GV);

  // Delete out-of-chunk GVs and their uses
  std::vector<GlobalVariable *> ToRemove;
  std::vector<WeakVH> InstToRemove;
  for (auto &GV : Program->globals())
    if (GV.hasInitializer() && !GVsToKeep.count(&GV)) {
      for (auto U : GV.users())
        if (auto *Inst = dyn_cast<Instruction>(U))
          InstToRemove.push_back(Inst);

      GV.replaceAllUsesWith(UndefValue::get(GV.getType()));
      ToRemove.push_back(&GV);
    }

  // Delete (unique) Instruction uses of unwanted GVs
  for (Value *V : InstToRemove) {
    if (!V)
      continue;
    auto *Inst = cast<Instruction>(V);
    Inst->replaceAllUsesWith(UndefValue::get(Inst->getType()));
    Inst->eraseFromParent();
  }

  for (auto *GV : ToRemove)
    GV->eraseFromParent();
}

/// Counts the amount of initialized GVs and displays their
/// respective name & index
static int countGVs(Module *Program) {
  // TODO: Silence index with --quiet flag
  outs() << "----------------------------\n";
  outs() << "GlobalVariable Index Reference:\n";
  int GVCount = 0;
  for (auto &GV : Program->globals())
    if (GV.hasInitializer())
      outs() << "\t" << ++GVCount << ": " << GV.getName() << "\n";
  outs() << "----------------------------\n";
  return GVCount;
}

void llvm::reduceGlobalsDeltaPass(TestRunner &Test) {
  outs() << "*** Reducing GVs...\n";
  int GVCount = countGVs(Test.getProgram());
  runDeltaPass(Test, GVCount, extractGVsFromModule);
}
