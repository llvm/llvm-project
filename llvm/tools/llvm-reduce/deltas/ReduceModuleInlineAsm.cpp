//===- ReduceModuleInlineAsm.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function which calls the Generic Delta pass to reduce
// module inline asm.
//
//===----------------------------------------------------------------------===//

#include "ReduceModuleInlineAsm.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalValue.h"

using namespace llvm;

static void clearModuleInlineAsm(std::vector<Chunk> ChunksToKeep,
                                 Module *Program) {
  Oracle O(ChunksToKeep);

  // TODO: clear line by line rather than all at once
  if (!O.shouldKeep())
    Program->setModuleInlineAsm("");
}

void llvm::reduceModuleInlineAsmDeltaPass(TestRunner &Test) {
  outs() << "*** Reducing Module Inline Asm...\n";
  runDeltaPass(Test, 1, clearModuleInlineAsm);
}
