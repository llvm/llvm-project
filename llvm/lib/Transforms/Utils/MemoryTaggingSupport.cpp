//== MemoryTaggingSupport.cpp - helpers for memory tagging implementations ===//
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

#include "llvm/Transforms/Utils/MemoryTaggingSupport.h"

#include "llvm/Analysis/ValueTracking.h"

namespace llvm {
namespace memtag {
namespace {
bool maybeReachableFromEachOther(const SmallVectorImpl<IntrinsicInst *> &Insts,
                                 const DominatorTree *DT, size_t MaxLifetimes) {
  // If we have too many lifetime ends, give up, as the algorithm below is N^2.
  if (Insts.size() > MaxLifetimes)
    return true;
  for (size_t I = 0; I < Insts.size(); ++I) {
    for (size_t J = 0; J < Insts.size(); ++J) {
      if (I == J)
        continue;
      if (isPotentiallyReachable(Insts[I], Insts[J], nullptr, DT))
        return true;
    }
  }
  return false;
}
} // namespace

bool isStandardLifetime(const SmallVectorImpl<IntrinsicInst *> &LifetimeStart,
                        const SmallVectorImpl<IntrinsicInst *> &LifetimeEnd,
                        const DominatorTree *DT, size_t MaxLifetimes) {
  // An alloca that has exactly one start and end in every possible execution.
  // If it has multiple ends, they have to be unreachable from each other, so
  // at most one of them is actually used for each execution of the function.
  return LifetimeStart.size() == 1 &&
         (LifetimeEnd.size() == 1 ||
          (LifetimeEnd.size() > 0 &&
           !maybeReachableFromEachOther(LifetimeEnd, DT, MaxLifetimes)));
}

Instruction *getUntagLocationIfFunctionExit(Instruction &Inst) {
  if (isa<ReturnInst>(Inst)) {
    if (CallInst *CI = Inst.getParent()->getTerminatingMustTailCall())
      return CI;
    return &Inst;
  }
  if (isa<ResumeInst, CleanupReturnInst>(Inst)) {
    return &Inst;
  }
  return nullptr;
}

void StackInfoBuilder::visit(Instruction &Inst) {
  if (CallInst *CI = dyn_cast<CallInst>(&Inst)) {
    if (CI->canReturnTwice()) {
      Info.CallsReturnTwice = true;
    }
  }
  if (AllocaInst *AI = dyn_cast<AllocaInst>(&Inst)) {
    if (IsInterestingAlloca(*AI)) {
      Info.AllocasToInstrument[AI].AI = AI;
      Info.AllocasToInstrument[AI].OldAI = AI;
    }
    return;
  }
  auto *II = dyn_cast<IntrinsicInst>(&Inst);
  if (II && (II->getIntrinsicID() == Intrinsic::lifetime_start ||
             II->getIntrinsicID() == Intrinsic::lifetime_end)) {
    AllocaInst *AI = findAllocaForValue(II->getArgOperand(1));
    if (!AI) {
      Info.UnrecognizedLifetimes.push_back(&Inst);
      return;
    }
    if (!IsInterestingAlloca(*AI))
      return;
    if (II->getIntrinsicID() == Intrinsic::lifetime_start)
      Info.AllocasToInstrument[AI].LifetimeStart.push_back(II);
    else
      Info.AllocasToInstrument[AI].LifetimeEnd.push_back(II);
    return;
  }
  if (auto *DVI = dyn_cast<DbgVariableIntrinsic>(&Inst)) {
    for (Value *V : DVI->location_ops()) {
      if (auto *AI = dyn_cast_or_null<AllocaInst>(V)) {
        if (!IsInterestingAlloca(*AI))
          continue;
        AllocaInfo &AInfo = Info.AllocasToInstrument[AI];
        auto &DVIVec = AInfo.DbgVariableIntrinsics;
        if (DVIVec.empty() || DVIVec.back() != DVI)
          DVIVec.push_back(DVI);
      }
    }
  }
  Instruction *ExitUntag = getUntagLocationIfFunctionExit(Inst);
  if (ExitUntag)
    Info.RetVec.push_back(ExitUntag);
}

} // namespace memtag
} // namespace llvm
