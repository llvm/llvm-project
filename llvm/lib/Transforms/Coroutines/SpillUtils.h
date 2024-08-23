//===- SpillUtils.h - Utilities for han dling for spills ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CoroInternal.h"
#include "SuspendCrossingInfo.h"

#ifndef LLVM_TRANSFORMS_COROUTINES_SPILLINGINFO_H
#define LLVM_TRANSFORMS_COROUTINES_SPILLINGINFO_H

namespace llvm {

namespace coro {

using SpillInfo = SmallMapVector<Value *, SmallVector<Instruction *, 2>, 8>;

struct AllocaInfo {
  AllocaInst *Alloca;
  DenseMap<Instruction *, std::optional<APInt>> Aliases;
  bool MayWriteBeforeCoroBegin;
  AllocaInfo(AllocaInst *Alloca,
             DenseMap<Instruction *, std::optional<APInt>> Aliases,
             bool MayWriteBeforeCoroBegin)
      : Alloca(Alloca), Aliases(std::move(Aliases)),
        MayWriteBeforeCoroBegin(MayWriteBeforeCoroBegin) {}
};

void collectSpillsFromArgs(SpillInfo &Spills, Function &F,
                           const SuspendCrossingInfo &Checker);
void collectSpillsAndAllocasFromInsts(
    SpillInfo &Spills, SmallVector<AllocaInfo, 8> &Allocas,
    SmallVector<Instruction *, 4> &DeadInstructions,
    SmallVector<CoroAllocaAllocInst *, 4> &LocalAllocas, Function &F,
    const SuspendCrossingInfo &Checker, const DominatorTree &DT,
    const coro::Shape &Shape);
void collectSpillsFromDbgInfo(SpillInfo &Spills, Function &F,
                              const SuspendCrossingInfo &Checker);

/// Async and Retcon{Once} conventions assume that all spill uses can be sunk
/// after the coro.begin intrinsic.
void sinkSpillUsesAfterCoroBegin(const DominatorTree &DT,
                                 CoroBeginInst *CoroBegin,
                                 coro::SpillInfo &Spills,
                                 SmallVectorImpl<coro::AllocaInfo> &Allocas);

// Get the insertion point for a spill after a Def.
BasicBlock::iterator getSpillInsertionPt(const coro::Shape &, Value *Def,
                                         const DominatorTree &DT);

} // namespace coro

} // namespace llvm

#endif // LLVM_TRANSFORMS_COROUTINES_SPILLINGINFO_H
