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
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"

namespace llvm {

class InterestingMemoryOperand {
public:
  Use *PtrUse;
  bool IsWrite;
  uint64_t TypeSize;
  MaybeAlign Alignment;
  // The mask Value, if we're looking at a masked load/store.
  Value *MaybeMask;

  InterestingMemoryOperand(Instruction *I, unsigned OperandNo, bool IsWrite,
                           class Type *OpType, MaybeAlign Alignment,
                           Value *MaybeMask = nullptr)
      : IsWrite(IsWrite), Alignment(Alignment), MaybeMask(MaybeMask) {
    const DataLayout &DL = I->getModule()->getDataLayout();
    TypeSize = DL.getTypeStoreSizeInBits(OpType);
    PtrUse = &I->getOperandUse(OperandNo);
  }

  Instruction *getInsn() { return cast<Instruction>(PtrUse->getUser()); }

  Value *getPtr() { return PtrUse->get(); }
};

// For an alloca valid between lifetime markers Start and Ends, call the
// Callback for all possible exits out of the lifetime in the containing
// function, which can return from the instructions in RetVec.
//
// Returns whether Ends covered all possible exits. If they did not,
// the caller should remove Ends to ensure that work done at the other
// exits does not happen outside of the lifetime.
template <typename F>
bool forAllReachableExits(const DominatorTree &DT, const PostDominatorTree &PDT,
                          const Instruction *Start,
                          const SmallVectorImpl<IntrinsicInst *> &Ends,
                          const SmallVectorImpl<Instruction *> &RetVec,
                          F Callback) {
  if (Ends.size() == 1 && PDT.dominates(Ends[0], Start)) {
    Callback(Ends[0]);
    return true;
  }
  SmallVector<Instruction *, 8> ReachableRetVec;
  unsigned NumCoveredExits = 0;
  for (auto &RI : RetVec) {
    if (!isPotentiallyReachable(Start, RI, nullptr, &DT))
      continue;
    ReachableRetVec.push_back(RI);
    // TODO(fmayer): We don't support diamond shapes, where multiple lifetime
    // ends together dominate the RI, but none of them does by itself.
    // Check how often this happens and decide whether to support this here.
    if (std::any_of(Ends.begin(), Ends.end(),
                    [&](Instruction *End) { return DT.dominates(End, RI); }))
      ++NumCoveredExits;
  }
  // If there's a mix of covered and non-covered exits, just put the untag
  // on exits, so we avoid the redundancy of untagging twice.
  if (NumCoveredExits == ReachableRetVec.size()) {
    for (auto *End : Ends)
      Callback(End);
  } else {
    for (auto &RI : ReachableRetVec)
      Callback(RI);
    // We may have inserted untag outside of the lifetime interval.
    // Signal the caller to remove the lifetime end call for this alloca.
    return false;
  }
  return true;
}

// Get AddressSanitizer parameters.
void getAddressSanitizerParams(const Triple &TargetTriple, int LongSize,
                               bool IsKasan, uint64_t *ShadowBase,
                               int *MappingScale, bool *OrShadowOffset);

} // namespace llvm

#endif
