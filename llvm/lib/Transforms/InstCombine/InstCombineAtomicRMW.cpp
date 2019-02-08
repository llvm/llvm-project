//===- InstCombineAtomicRMW.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the visit functions for atomic rmw instructions.
//
//===----------------------------------------------------------------------===//
#include "InstCombineInternal.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

Instruction *InstCombiner::visitAtomicRMWInst(AtomicRMWInst &RMWI) {
  switch (RMWI.getOperation()) {
  default:
    break;
  case AtomicRMWInst::Add:
  case AtomicRMWInst::Sub:
  case AtomicRMWInst::Or:
    // Replace atomicrmw <op> addr, 0 => load atomic addr.

    // Volatile RMWs perform a load and a store, we cannot replace
    // this by just a load.
    if (RMWI.isVolatile())
      break;

    auto *CI = dyn_cast<ConstantInt>(RMWI.getValOperand());
    if (!CI || !CI->isZero())
      break;
    // Check if the required ordering is compatible with an
    // atomic load.
    AtomicOrdering Ordering = RMWI.getOrdering();
    assert(Ordering != AtomicOrdering::NotAtomic &&
           Ordering != AtomicOrdering::Unordered &&
           "AtomicRMWs don't make sense with Unordered or NotAtomic");
    if (Ordering != AtomicOrdering::Acquire &&
        Ordering != AtomicOrdering::Monotonic)
      break;
    LoadInst *Load = new LoadInst(RMWI.getType(), RMWI.getPointerOperand());
    Load->setAtomic(Ordering, RMWI.getSyncScopeID());
    return Load;
  }
  return nullptr;
}
