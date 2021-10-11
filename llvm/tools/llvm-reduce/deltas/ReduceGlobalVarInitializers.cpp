//===- ReduceGlobalVars.cpp - Specialized Delta Pass ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function which calls the Generic Delta pass in order
// to reduce initializers of Global Variables in the provided Module.
//
//===----------------------------------------------------------------------===//

#include "ReduceGlobalVarInitializers.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalValue.h"

using namespace llvm;

/// Removes all the Initialized GVs that aren't inside the desired Chunks.
static void extractGVsFromModule(Oracle &O, Module &Program) {
  // Drop initializers of out-of-chunk GVs
  for (auto &GV : Program.globals())
    if (GV.hasInitializer() && !O.shouldKeep()) {
      GV.setInitializer(nullptr);
      GV.setLinkage(GlobalValue::LinkageTypes::ExternalLinkage);
      GV.setComdat(nullptr);
    }
}

/// Counts the amount of initialized GVs and displays their
/// respective name & index
static int countGVs(Module &Program) {
  // TODO: Silence index with --quiet flag
  outs() << "----------------------------\n";
  outs() << "GlobalVariable Index Reference:\n";
  int GVCount = 0;
  for (auto &GV : Program.globals())
    if (GV.hasInitializer())
      outs() << "\t" << ++GVCount << ": " << GV.getName() << "\n";
  outs() << "----------------------------\n";
  return GVCount;
}

void llvm::reduceGlobalsInitializersDeltaPass(TestRunner &Test) {
  outs() << "*** Reducing GVs initializers...\n";
  int GVCount = countGVs(Test.getProgram());
  runDeltaPass(Test, GVCount, extractGVsFromModule);
}
