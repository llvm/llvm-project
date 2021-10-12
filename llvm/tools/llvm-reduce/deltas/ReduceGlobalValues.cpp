//===- ReduceGlobalValues.cpp - Specialized Delta Pass --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function which calls the Generic Delta pass to reduce
// global value attributes/specifiers.
//
//===----------------------------------------------------------------------===//

#include "ReduceGlobalValues.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalValue.h"

using namespace llvm;

static bool isValidDSOLocalReductionGV(GlobalValue &GV) {
  return GV.isDSOLocal() && !GV.isImplicitDSOLocal();
}

/// Sets dso_local to false for all global values.
static void extractGVsFromModule(Oracle &O, Module &Program) {
  // remove dso_local from global values
  for (auto &GV : Program.global_values())
    if (isValidDSOLocalReductionGV(GV) && !O.shouldKeep()) {
      GV.setDSOLocal(false);
    }
}

/// Counts the amount of global values with dso_local and displays their
/// respective name & index
static int countGVs(Module &Program) {
  // TODO: Silence index with --quiet flag
  outs() << "----------------------------\n";
  outs() << "GlobalValue Index Reference:\n";
  int GVCount = 0;
  for (auto &GV : Program.global_values())
    if (isValidDSOLocalReductionGV(GV))
      outs() << "\t" << ++GVCount << ": " << GV.getName() << "\n";
  outs() << "----------------------------\n";
  return GVCount;
}

void llvm::reduceGlobalValuesDeltaPass(TestRunner &Test) {
  outs() << "*** Reducing GlobalValues...\n";
  int GVCount = countGVs(Test.getProgram());
  runDeltaPass(Test, GVCount, extractGVsFromModule);
}
