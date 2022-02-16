//===- MemoryTaggingSupport.h - helpers for memory tagging implementations ===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares common infrastructure for HWAddressSanitizer and
// Aarch64StackTagging.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_UTILS_MEMORYTAGGINGSUPPORT_H
#define LLVM_TRANSFORMS_UTILS_MEMORYTAGGINGSUPPORT_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueHandle.h"

namespace llvm {
namespace memtag {
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
  for (auto *RI : RetVec) {
    if (!isPotentiallyReachable(Start, RI, nullptr, &DT))
      continue;
    ReachableRetVec.push_back(RI);
    // TODO(fmayer): We don't support diamond shapes, where multiple lifetime
    // ends together dominate the RI, but none of them does by itself.
    // Check how often this happens and decide whether to support this here.
    if (llvm::any_of(Ends, [&](auto *End) { return DT.dominates(End, RI); }))
      ++NumCoveredExits;
  }
  // If there's a mix of covered and non-covered exits, just put the untag
  // on exits, so we avoid the redundancy of untagging twice.
  if (NumCoveredExits == ReachableRetVec.size()) {
    for (auto *End : Ends)
      Callback(End);
  } else {
    for (auto *RI : ReachableRetVec)
      Callback(RI);
    // We may have inserted untag outside of the lifetime interval.
    // Signal the caller to remove the lifetime end call for this alloca.
    return false;
  }
  return true;
}

bool isStandardLifetime(const SmallVectorImpl<IntrinsicInst *> &LifetimeStart,
                        const SmallVectorImpl<IntrinsicInst *> &LifetimeEnd,
                        const DominatorTree *DT, size_t MaxLifetimes);

Instruction *getUntagLocationIfFunctionExit(Instruction &Inst);

struct AllocaInfo {
  AllocaInst *AI;
  TrackingVH<Instruction> OldAI; // Track through RAUW to replace debug uses.
  SmallVector<IntrinsicInst *, 2> LifetimeStart;
  SmallVector<IntrinsicInst *, 2> LifetimeEnd;
  SmallVector<DbgVariableIntrinsic *, 2> DbgVariableIntrinsics;
};

struct StackInfo {
  MapVector<AllocaInst *, AllocaInfo> AllocasToInstrument;
  SmallVector<Instruction *, 4> UnrecognizedLifetimes;
  SmallVector<Instruction *, 8> RetVec;
  bool CallsReturnTwice = false;
};

class StackInfoBuilder {
public:
  StackInfoBuilder(std::function<bool(const AllocaInst &)> IsInterestingAlloca)
      : IsInterestingAlloca(IsInterestingAlloca) {}

  void visit(Instruction &Inst);
  StackInfo &get() { return Info; };

private:
  StackInfo Info;
  std::function<bool(const AllocaInst &)> IsInterestingAlloca;
};

uint64_t getAllocaSizeInBytes(const AllocaInst &AI);
bool alignAndPadAlloca(memtag::AllocaInfo &Info, llvm::Align Align);

} // namespace memtag
} // namespace llvm

#endif
