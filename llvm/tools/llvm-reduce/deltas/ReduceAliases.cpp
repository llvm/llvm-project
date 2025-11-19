//===- ReduceAliases.cpp - Specialized Delta Pass -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function which calls the Generic Delta pass in order
// to reduce aliases in the provided Module.
//
//===----------------------------------------------------------------------===//

#include "ReduceAliases.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

/// Removes all aliases aren't inside any of the
/// desired Chunks.
void llvm::reduceAliasesDeltaPass(Oracle &O, ReducerWorkItem &Program) {
  for (auto &GA : make_early_inc_range(Program.getModule().aliases())) {
    if (!O.shouldKeep()) {
      GA.replaceAllUsesWith(GA.getAliasee());
      GA.eraseFromParent();
    }
  }
}

void llvm::reduceIFuncsDeltaPass(Oracle &O, ReducerWorkItem &WorkItem) {
  Module &Mod = WorkItem.getModule();

  std::vector<GlobalIFunc *> IFuncs;
  for (GlobalIFunc &GI : Mod.ifuncs()) {
    if (!O.shouldKeep())
      IFuncs.push_back(&GI);
  }

  if (!IFuncs.empty())
    lowerGlobalIFuncUsersAsGlobalCtor(Mod, IFuncs);
}
