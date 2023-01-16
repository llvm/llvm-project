//===- ReduceInstructionFlags.cpp - Specialized Delta Pass ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Try to remove optimization flags on instructions
//
//===----------------------------------------------------------------------===//

#include "ReduceInstructionFlags.h"
#include "Delta.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"

static void reduceFlagsInModule(Oracle &O, ReducerWorkItem &WorkItem) {
  for (Function &F : WorkItem.getModule()) {
    for (Instruction &I : instructions(F)) {
      if (auto *OBO = dyn_cast<OverflowingBinaryOperator>(&I)) {
        if (OBO->hasNoSignedWrap() && !O.shouldKeep())
          I.setHasNoSignedWrap(false);
        if (OBO->hasNoUnsignedWrap() && !O.shouldKeep())
          I.setHasNoUnsignedWrap(false);
      } else if (auto *PE = dyn_cast<PossiblyExactOperator>(&I)) {
        if (PE->isExact() && !O.shouldKeep())
          I.setIsExact(false);
      } else if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
        if (GEP->isInBounds() && !O.shouldKeep())
          GEP->setIsInBounds(false);
      } else if (auto *FPOp = dyn_cast<FPMathOperator>(&I)) {
        FastMathFlags Flags = FPOp->getFastMathFlags();

        if (Flags.allowReassoc() && !O.shouldKeep())
          Flags.setAllowReassoc(false);

        if (Flags.noNaNs() && !O.shouldKeep())
          Flags.setNoNaNs(false);

        if (Flags.noInfs() && !O.shouldKeep())
          Flags.setNoInfs(false);

        if (Flags.noSignedZeros() && !O.shouldKeep())
          Flags.setNoSignedZeros(false);

        if (Flags.allowReciprocal() && !O.shouldKeep())
          Flags.setAllowReciprocal(false);

        if (Flags.allowContract() && !O.shouldKeep())
          Flags.setAllowContract(false);

        if (Flags.approxFunc() && !O.shouldKeep())
          Flags.setApproxFunc(false);

        I.copyFastMathFlags(Flags);
      }
    }
  }
}

void llvm::reduceInstructionFlagsDeltaPass(TestRunner &Test) {
  runDeltaPass(Test, reduceFlagsInModule, "Reducing Instruction Flags");
}
