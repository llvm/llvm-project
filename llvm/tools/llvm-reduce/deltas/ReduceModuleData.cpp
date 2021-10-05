//===- ReduceModuleData.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a reduce pass to reduce various module data.
//
//===----------------------------------------------------------------------===//

#include "ReduceModuleData.h"

using namespace llvm;

static void clearModuleData(Oracle &O, Module &Program) {
  if (!Program.getModuleIdentifier().empty() && !O.shouldKeep())
    Program.setModuleIdentifier("");
  if (!Program.getSourceFileName().empty() && !O.shouldKeep())
    Program.setSourceFileName("");
  if (!Program.getDataLayoutStr().empty() && !O.shouldKeep())
    Program.setDataLayout("");
  if (!Program.getTargetTriple().empty() && !O.shouldKeep())
    Program.setTargetTriple("");
  // TODO: clear line by line rather than all at once
  if (!Program.getModuleInlineAsm().empty() && !O.shouldKeep())
    Program.setModuleInlineAsm("");
}

static int countModuleData(Module &M) {
  int Count = 0;
  if (!M.getModuleIdentifier().empty())
    ++Count;
  if (!M.getSourceFileName().empty())
    ++Count;
  if (!M.getDataLayoutStr().empty())
    ++Count;
  if (!M.getTargetTriple().empty())
    ++Count;
  if (!M.getModuleInlineAsm().empty())
    ++Count;
  return Count;
}

void llvm::reduceModuleDataDeltaPass(TestRunner &Test) {
  outs() << "*** Reducing Module Data...\n";
  int Count = countModuleData(Test.getProgram());
  runDeltaPass(Test, Count, clearModuleData);
}
