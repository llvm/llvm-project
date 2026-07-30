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

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/StackSafetyAnalysis.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/Alignment.h"

namespace llvm {
class DominatorTree;
class IntrinsicInst;
class PostDominatorTree;
class AllocaInst;
class Instruction;
namespace memtag {
struct AllocaInfo {
  struct BBInfo {
    Intrinsic::ID First = Intrinsic::not_intrinsic;
    Intrinsic::ID Last = Intrinsic::not_intrinsic;
  };
  AllocaInst *AI;
  SmallVector<IntrinsicInst *, 2> LifetimeStart;
  SmallVector<IntrinsicInst *, 2> LifetimeEnd;
  SmallVector<DbgVariableRecord *, 2> DbgVariableRecords;
  MapVector<BasicBlock *, struct BBInfo> BBInfos;
};

// For an alloca valid between lifetime markers Start and Ends, call the
// Callback for all possible exits out of the lifetime in the containing
// function, which can return from the instructions in RetVec.
//
// Returns whether Ends covered all possible exits. If they did not,
// the caller should remove Ends to ensure that work done at the other
// exits does not happen outside of the lifetime.
LLVM_ABI void
forAllReachableExits(const DominatorTree &DT, const PostDominatorTree &PDT,
                     const LoopInfo &LI, const AllocaInfo &AInfo,
                     const SmallVectorImpl<Instruction *> &RetVec,
                     llvm::function_ref<void(Instruction *)> Callback);

LLVM_ABI bool isSupportedLifetime(const AllocaInfo &AInfo,
                                  const DominatorTree *DT, const LoopInfo *LI);

LLVM_ABI Instruction *getUntagLocationIfFunctionExit(Instruction &Inst);

struct StackInfo {
  MapVector<AllocaInst *, AllocaInfo> AllocasToInstrument;
  SmallVector<Instruction *, 8> RetVec;
  bool CallsReturnTwice = false;
};

enum class AllocaInterestingness {
  // Uninteresting because of the nature of the alloca.
  kUninteresting,
  // Uninteresting because proven safe.
  kSafe,
  // Interesting.
  kInteresting
};

class StackInfoBuilder {
public:
  StackInfoBuilder(const StackSafetyGlobalInfo *SSI, const char *DebugType)
      : SSI(SSI), DebugType(DebugType) {}

  LLVM_ABI void visit(OptimizationRemarkEmitter &ORE, Instruction &Inst);
  LLVM_ABI AllocaInterestingness getAllocaInterestingness(const AllocaInst &AI);
  StackInfo &get() { return Info; };

private:
  StackInfo Info;
  const StackSafetyGlobalInfo *SSI;
  const char *DebugType;
};

LLVM_ABI uint64_t getAllocaSizeInBytes(const AllocaInst &AI);
LLVM_ABI void alignAndPadAlloca(memtag::AllocaInfo &Info, llvm::Align Align);

LLVM_ABI Value *readRegister(IRBuilder<> &IRB, StringRef Name);
LLVM_ABI Value *getFP(IRBuilder<> &IRB);
LLVM_ABI Value *getPC(const Triple &TargetTriple, IRBuilder<> &IRB);
LLVM_ABI Value *getAndroidSlotPtr(IRBuilder<> &IRB, int Slot);
LLVM_ABI Value *getDarwinSlotPtr(IRBuilder<> &IRB, int Slot);

LLVM_ABI void annotateDebugRecords(AllocaInfo &Info, unsigned int Tag);
LLVM_ABI Value *incrementThreadLong(IRBuilder<> &IRB, Value *ThreadLong,
                                    unsigned int Inc,
                                    bool IsMemtagDarwin = false);

} // namespace memtag
} // namespace llvm

#endif
