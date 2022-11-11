//===- ReduceGlobalVars.cpp - Specialized Delta Pass ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function which calls the Generic Delta pass in order
// to reduce Global Variables in the provided Module.
//
//===----------------------------------------------------------------------===//

#include "ReduceGlobalVars.h"
#include "Utils.h"
#include "llvm/IR/Constants.h"
#include <set>

using namespace llvm;

/// Removes all the GVs that aren't inside the desired Chunks.
static void extractGVsFromModule(Oracle &O, Module &Program) {
  // Get GVs inside desired chunks
  std::vector<GlobalVariable *> InitGVsToKeep;
  for (auto &GV : Program.globals())
    if (O.shouldKeep())
      InitGVsToKeep.push_back(&GV);

  // We create a vector first, then convert it to a set, so that we don't have
  // to pay the cost of rebalancing the set frequently if the order we insert
  // the elements doesn't match the order they should appear inside the set.
  std::set<GlobalVariable *> GVsToKeep(InitGVsToKeep.begin(),
                                       InitGVsToKeep.end());

  // Delete out-of-chunk GVs and their uses
  std::vector<GlobalVariable *> ToRemove;
  std::vector<WeakVH> InstToRemove;
  for (auto &GV : Program.globals())
    if (!GVsToKeep.count(&GV)) {
      for (auto *U : GV.users())
        if (auto *Inst = dyn_cast<Instruction>(U))
          InstToRemove.push_back(Inst);

      GV.replaceAllUsesWith(getDefaultValue(GV.getType()));
      ToRemove.push_back(&GV);
    }

  // Delete (unique) Instruction uses of unwanted GVs
  for (Value *V : InstToRemove) {
    if (!V)
      continue;
    auto *Inst = cast<Instruction>(V);
    Inst->replaceAllUsesWith(getDefaultValue(Inst->getType()));
    Inst->eraseFromParent();
  }

  for (auto *GV : ToRemove)
    GV->eraseFromParent();
}

void llvm::reduceGlobalsDeltaPass(TestRunner &Test) {
  runDeltaPass(Test, extractGVsFromModule, "Reducing GlobalVariables");
}
