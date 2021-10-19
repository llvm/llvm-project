//===- ReduceOperands.cpp - Specialized Delta Pass ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function to reduce operands to undef.
//
//===----------------------------------------------------------------------===//

#include "ReduceOperands.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"

using namespace llvm;

/// Returns if the given operand is undef.
static bool operandIsUndefValue(Use &Op) {
  if (auto *C = dyn_cast<Constant>(Op)) {
    return isa<UndefValue>(C);
  }
  return false;
}

/// Returns if an operand can be reduced to undef.
/// TODO: make this logic check what types are reducible rather than
/// check what types that are not reducible.
static bool canReduceOperand(Use &Op) {
  auto *Ty = Op->getType();
  // Can't reduce labels to undef
  return !Ty->isLabelTy() && !operandIsUndefValue(Op);
}

/// Sets Operands to undef.
static void extractOperandsFromModule(Oracle &O, Module &Program) {
  // Extract Operands from the module.
  for (auto &F : Program.functions()) {
    for (auto &I : instructions(&F)) {
      for (auto &Op : I.operands()) {
        // Filter Operands then set to undef.
        if (canReduceOperand(Op) && !O.shouldKeep()) {
          auto *Ty = Op->getType();
          Op.set(UndefValue::get(Ty));
        }
      }
    }
  }
}

/// Counts the amount of operands in the module that can be reduced.
static int countOperands(Module &Program) {
  int Count = 0;
  for (auto &F : Program.functions()) {
    for (auto &I : instructions(&F)) {
      for (auto &Op : I.operands()) {
        if (canReduceOperand(Op)) {
          Count++;
        }
      }
    }
  }
  return Count;
}

void llvm::reduceOperandsDeltaPass(TestRunner &Test) {
  errs() << "*** Reducing Operands...\n";
  int Count = countOperands(Test.getProgram());
  runDeltaPass(Test, Count, extractOperandsFromModule);
}
