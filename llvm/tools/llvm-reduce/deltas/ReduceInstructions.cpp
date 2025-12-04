//===- ReduceInstructions.cpp - Specialized Delta Pass ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function which calls the Generic Delta pass in order
// to reduce uninteresting Instructions from defined functions.
//
//===----------------------------------------------------------------------===//

#include "ReduceInstructions.h"
#include "Utils.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

/// Filter out cases where deleting the instruction will likely cause the
/// user/def of the instruction to fail the verifier.
//
// TODO: Technically the verifier only enforces preallocated token usage and
// there is a none token.
static bool shouldAlwaysKeep(const Instruction &I) {
  return I.isEHPad() || I.getType()->isTokenTy() || I.isSwiftError();
}

/// Removes out-of-chunk arguments from functions, and modifies their calls
/// accordingly. It also removes allocations of out-of-chunk arguments.
void llvm::reduceInstructionsDeltaPass(Oracle &O, ReducerWorkItem &WorkItem) {
  Module &Program = WorkItem.getModule();

  for (auto &F : Program) {
    for (auto &BB : F) {
      // Removing the terminator would make the block invalid. Only iterate over
      // instructions before the terminator.
      for (auto &Inst :
           make_early_inc_range(make_range(BB.begin(), std::prev(BB.end())))) {
        if (!shouldAlwaysKeep(Inst) && !O.shouldKeep()) {
          Inst.replaceAllUsesWith(isa<AllocaInst>(Inst)
                                      ? PoisonValue::get(Inst.getType())
                                      : getDefaultValue(Inst.getType()));
          Inst.eraseFromParent();
        }
      }
    }
  }
}
