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

static void clearModuleData(std::vector<Chunk> ChunksToKeep, Module *Program) {
  Oracle O(ChunksToKeep);

  if (!O.shouldKeep())
    Program->setModuleIdentifier("");
  if (!O.shouldKeep())
    Program->setSourceFileName("");
  if (!O.shouldKeep())
    Program->setDataLayout("");
  if (!O.shouldKeep())
    Program->setTargetTriple("");
  // TODO: clear line by line rather than all at once
  if (!O.shouldKeep())
    Program->setModuleInlineAsm("");
}

void llvm::reduceModuleDataDeltaPass(TestRunner &Test) {
  outs() << "*** Reducing Module Data...\n";
  runDeltaPass(Test, 5, clearModuleData);
}
