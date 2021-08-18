//===--------- Definition of the AddressSanitizer class ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares common infrastructure for AddressSanitizer and
// HWAddressSanitizer.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_ADDRESSSANITIZERCOMMON_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_ADDRESSSANITIZERCOMMON_H

#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"

namespace llvm {

class InterestingMemoryOperand {
public:
  Use *PtrUse;
  bool IsWrite;
  Type *OpType;
  uint64_t TypeSize;
  MaybeAlign Alignment;
  // The mask Value, if we're looking at a masked load/store.
  Value *MaybeMask;

  InterestingMemoryOperand(Instruction *I, unsigned OperandNo, bool IsWrite,
                           class Type *OpType, MaybeAlign Alignment,
                           Value *MaybeMask = nullptr)
      : IsWrite(IsWrite), OpType(OpType), Alignment(Alignment),
        MaybeMask(MaybeMask) {
    const DataLayout &DL = I->getModule()->getDataLayout();
    TypeSize = DL.getTypeStoreSizeInBits(OpType);
    PtrUse = &I->getOperandUse(OperandNo);
  }

  Instruction *getInsn() { return cast<Instruction>(PtrUse->getUser()); }

  Value *getPtr() { return PtrUse->get(); }
};

// For an alloca valid between lifetime markers Start and End, call the
// Callback for all possible exits out of the lifetime in the containing
// function, which can return from the instructions in RetVec.
//
// Returns whether End was the only possible exit. If it wasn't, the caller
// should remove End to ensure that work done at the other exits does not
// happen outside of the lifetime.
template <typename F>
bool forAllReachableExits(const DominatorTree &DT, const PostDominatorTree &PDT,
                          const Instruction *Start, Instruction *End,
                          const SmallVectorImpl<Instruction *> &RetVec,
                          F Callback) {
  // We need to ensure that if we tag some object, we certainly untag it
  // before the function exits.
  if (PDT.dominates(End, Start)) {
    Callback(End);
  } else {
    SmallVector<Instruction *, 8> ReachableRetVec;
    unsigned NumCoveredExits = 0;
    for (auto &RI : RetVec) {
      if (!isPotentiallyReachable(Start, RI, nullptr, &DT))
        continue;
      ReachableRetVec.push_back(RI);
      if (DT.dominates(End, RI))
        ++NumCoveredExits;
    }
    // If there's a mix of covered and non-covered exits, just put the untag
    // on exits, so we avoid the redundancy of untagging twice.
    if (NumCoveredExits == ReachableRetVec.size()) {
      Callback(End);
    } else {
      for (auto &RI : ReachableRetVec)
        Callback(RI);
      // We may have inserted untag outside of the lifetime interval.
      // Signal the caller to remove the lifetime end call for this alloca.
      return false;
    }
  }
  return true;
}

} // namespace llvm

#endif
