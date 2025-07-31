//===- SimplifyInstructions.cpp - Specialized Delta Pass ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function which calls the Generic Delta pass in order
// to simplify Instructions in defined functions.
//
//===----------------------------------------------------------------------===//

#include "SimplifyInstructions.h"
#include "llvm/Analysis/InstructionSimplify.h"

using namespace llvm;

/// Calls simplifyInstruction in each instruction in functions, and replaces
/// their values.
void llvm::simplifyInstructionsDeltaPass(Oracle &O, ReducerWorkItem &WorkItem) {
  Module &Program = WorkItem.getModule();
  const DataLayout &DL = Program.getDataLayout();

  for (auto &F : Program) {
    for (auto &BB : F) {
      for (auto &Inst : make_early_inc_range(BB)) {
        SimplifyQuery Q(DL, &Inst);
        if (Value *Simplified = simplifyInstruction(&Inst, Q)) {
          if (O.shouldKeep())
            continue;
          Inst.replaceAllUsesWith(Simplified);
          Inst.eraseFromParent();
        }
      }
    }
  }
}
