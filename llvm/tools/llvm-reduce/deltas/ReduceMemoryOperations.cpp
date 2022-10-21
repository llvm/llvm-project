//===- ReduceOpcodes.cpp - Specialized Delta Pass -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ReduceMemoryOperations.h"
#include "Delta.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"

static void removeVolatileInFunction(Oracle &O, Function &F) {
  LLVMContext &Ctx = F.getContext();
  for (Instruction &I : instructions(F)) {
    if (LoadInst *LI = dyn_cast<LoadInst>(&I)) {
      if (LI->isVolatile() && !O.shouldKeep())
        LI->setVolatile(false);
    } else if (StoreInst *SI = dyn_cast<StoreInst>(&I)) {
      if (SI->isVolatile() && !O.shouldKeep())
        SI->setVolatile(false);
    } else if (AtomicRMWInst *RMW = dyn_cast<AtomicRMWInst>(&I)) {
      if (RMW->isVolatile() && !O.shouldKeep())
        RMW->setVolatile(false);
    } else if (AtomicCmpXchgInst *CmpXChg = dyn_cast<AtomicCmpXchgInst>(&I)) {
      if (CmpXChg->isVolatile() && !O.shouldKeep())
        CmpXChg->setVolatile(false);
    } else if (MemIntrinsic *MemIntrin = dyn_cast<MemIntrinsic>(&I)) {
      if (MemIntrin->isVolatile() && !O.shouldKeep())
        MemIntrin->setVolatile(ConstantInt::getFalse(Ctx));
    }
  }
}

static void removeVolatileInModule(Oracle &O, Module &Mod) {
  for (Function &F : Mod)
    removeVolatileInFunction(O, F);
}

void llvm::reduceVolatileInstructionsDeltaPass(TestRunner &Test) {
  runDeltaPass(Test, removeVolatileInModule, "Reducing Volatile Instructions");
}
