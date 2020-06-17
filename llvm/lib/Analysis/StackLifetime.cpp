//===- StackLifetime.cpp - Alloca Lifetime Analysis -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/StackLifetime.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/User.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include <tuple>

using namespace llvm;

#define DEBUG_TYPE "stack-lifetime"

const StackLifetime::LiveRange &
StackLifetime::getLiveRange(const AllocaInst *AI) const {
  const auto IT = AllocaNumbering.find(AI);
  assert(IT != AllocaNumbering.end());
  return LiveRanges[IT->second];
}

static bool readMarker(const Instruction *I, bool *IsStart) {
  if (!I->isLifetimeStartOrEnd())
    return false;

  auto *II = cast<IntrinsicInst>(I);
  *IsStart = II->getIntrinsicID() == Intrinsic::lifetime_start;
  return true;
}

std::vector<const IntrinsicInst *> StackLifetime::getMarkers() const {
  std::vector<const IntrinsicInst *> Markers;
  for (auto &M : InstructionNumbering)
    if (M.getFirst()->isLifetimeStartOrEnd())
      Markers.push_back(M.getFirst());
  return Markers;
}

void StackLifetime::collectMarkers() {
  InterestingAllocas.resize(NumAllocas);
  DenseMap<const BasicBlock *, SmallDenseMap<const IntrinsicInst *, Marker>>
      BBMarkerSet;

  // Compute the set of start/end markers per basic block.
  for (unsigned AllocaNo = 0; AllocaNo < NumAllocas; ++AllocaNo) {
    const AllocaInst *AI = Allocas[AllocaNo];
    SmallVector<const Instruction *, 8> WorkList;
    WorkList.push_back(AI);
    while (!WorkList.empty()) {
      const Instruction *I = WorkList.pop_back_val();
      for (const User *U : I->users()) {
        if (auto *BI = dyn_cast<BitCastInst>(U)) {
          WorkList.push_back(BI);
          continue;
        }
        auto *UI = dyn_cast<IntrinsicInst>(U);
        if (!UI)
          continue;
        bool IsStart;
        if (!readMarker(UI, &IsStart))
          continue;
        if (IsStart)
          InterestingAllocas.set(AllocaNo);
        BBMarkerSet[UI->getParent()][UI] = {AllocaNo, IsStart};
      }
    }
  }

  // Compute instruction numbering. Only the following instructions are
  // considered:
  // * Basic block entries
  // * Lifetime markers
  // For each basic block, compute
  // * the list of markers in the instruction order
  // * the sets of allocas whose lifetime starts or ends in this BB
  LLVM_DEBUG(dbgs() << "Instructions:\n");
  unsigned InstNo = 0;
  for (const BasicBlock *BB : depth_first(&F)) {
    LLVM_DEBUG(dbgs() << "  " << InstNo << ": BB " << BB->getName() << "\n");
    unsigned BBStart = InstNo++;

    BlockLifetimeInfo &BlockInfo =
        BlockLiveness.try_emplace(BB, NumAllocas).first->getSecond();

    auto &BlockMarkerSet = BBMarkerSet[BB];
    if (BlockMarkerSet.empty()) {
      unsigned BBEnd = InstNo;
      BlockInstRange[BB] = std::make_pair(BBStart, BBEnd);
      continue;
    }

    auto ProcessMarker = [&](const IntrinsicInst *I, const Marker &M) {
      LLVM_DEBUG(dbgs() << "  " << InstNo << ":  "
                        << (M.IsStart ? "start " : "end   ") << M.AllocaNo
                        << ", " << *I << "\n");

      BBMarkers[BB].push_back({InstNo, M});

      InstructionNumbering[I] = InstNo++;

      if (M.IsStart) {
        BlockInfo.End.reset(M.AllocaNo);
        BlockInfo.Begin.set(M.AllocaNo);
      } else {
        BlockInfo.Begin.reset(M.AllocaNo);
        BlockInfo.End.set(M.AllocaNo);
      }
    };

    if (BlockMarkerSet.size() == 1) {
      ProcessMarker(BlockMarkerSet.begin()->getFirst(),
                    BlockMarkerSet.begin()->getSecond());
    } else {
      // Scan the BB to determine the marker order.
      for (const Instruction &I : *BB) {
        const IntrinsicInst *II = dyn_cast<IntrinsicInst>(&I);
        if (!II)
          continue;
        auto It = BlockMarkerSet.find(II);
        if (It == BlockMarkerSet.end())
          continue;
        ProcessMarker(II, It->getSecond());
      }
    }

    unsigned BBEnd = InstNo;
    BlockInstRange[BB] = std::make_pair(BBStart, BBEnd);
  }
  NumInst = InstNo;
}

void StackLifetime::calculateLocalLiveness() {
  bool Changed = true;
  while (Changed) {
    Changed = false;

    for (const BasicBlock *BB : depth_first(&F)) {
      BlockLifetimeInfo &BlockInfo = BlockLiveness.find(BB)->getSecond();

      // Compute LiveIn by unioning together the LiveOut sets of all preds.
      BitVector LocalLiveIn;
      for (auto *PredBB : predecessors(BB)) {
        LivenessMap::const_iterator I = BlockLiveness.find(PredBB);
        // If a predecessor is unreachable, ignore it.
        if (I == BlockLiveness.end())
          continue;
        LocalLiveIn |= I->second.LiveOut;
      }

      // Compute LiveOut by subtracting out lifetimes that end in this
      // block, then adding in lifetimes that begin in this block.  If
      // we have both BEGIN and END markers in the same basic block
      // then we know that the BEGIN marker comes after the END,
      // because we already handle the case where the BEGIN comes
      // before the END when collecting the markers (and building the
      // BEGIN/END vectors).
      BitVector LocalLiveOut = LocalLiveIn;
      LocalLiveOut.reset(BlockInfo.End);
      LocalLiveOut |= BlockInfo.Begin;

      // Update block LiveIn set, noting whether it has changed.
      if (LocalLiveIn.test(BlockInfo.LiveIn)) {
        Changed = true;
        BlockInfo.LiveIn |= LocalLiveIn;
      }

      // Update block LiveOut set, noting whether it has changed.
      if (LocalLiveOut.test(BlockInfo.LiveOut)) {
        Changed = true;
        BlockInfo.LiveOut |= LocalLiveOut;
      }
    }
  } // while changed.
}

void StackLifetime::calculateLiveIntervals() {
  for (auto IT : BlockLiveness) {
    const BasicBlock *BB = IT.getFirst();
    BlockLifetimeInfo &BlockInfo = IT.getSecond();
    unsigned BBStart, BBEnd;
    std::tie(BBStart, BBEnd) = BlockInstRange[BB];

    BitVector Started, Ended;
    Started.resize(NumAllocas);
    Ended.resize(NumAllocas);
    SmallVector<unsigned, 8> Start;
    Start.resize(NumAllocas);

    // LiveIn ranges start at the first instruction.
    for (unsigned AllocaNo = 0; AllocaNo < NumAllocas; ++AllocaNo) {
      if (BlockInfo.LiveIn.test(AllocaNo)) {
        Started.set(AllocaNo);
        Start[AllocaNo] = BBStart;
      }
    }

    for (auto &It : BBMarkers[BB]) {
      unsigned InstNo = It.first;
      bool IsStart = It.second.IsStart;
      unsigned AllocaNo = It.second.AllocaNo;

      if (IsStart) {
        assert(!Started.test(AllocaNo) || Start[AllocaNo] == BBStart);
        if (!Started.test(AllocaNo)) {
          Started.set(AllocaNo);
          Ended.reset(AllocaNo);
          Start[AllocaNo] = InstNo;
        }
      } else {
        assert(!Ended.test(AllocaNo));
        if (Started.test(AllocaNo)) {
          LiveRanges[AllocaNo].addRange(Start[AllocaNo], InstNo);
          Started.reset(AllocaNo);
        }
        Ended.set(AllocaNo);
      }
    }

    for (unsigned AllocaNo = 0; AllocaNo < NumAllocas; ++AllocaNo)
      if (Started.test(AllocaNo))
        LiveRanges[AllocaNo].addRange(Start[AllocaNo], BBEnd);
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void StackLifetime::dumpAllocas() const {
  dbgs() << "Allocas:\n";
  for (unsigned AllocaNo = 0; AllocaNo < NumAllocas; ++AllocaNo)
    dbgs() << "  " << AllocaNo << ": " << *Allocas[AllocaNo] << "\n";
}

LLVM_DUMP_METHOD void StackLifetime::dumpBlockLiveness() const {
  dbgs() << "Block liveness:\n";
  for (auto IT : BlockLiveness) {
    const BasicBlock *BB = IT.getFirst();
    const BlockLifetimeInfo &BlockInfo = BlockLiveness.find(BB)->getSecond();
    auto BlockRange = BlockInstRange.find(BB)->getSecond();
    dbgs() << "  BB [" << BlockRange.first << ", " << BlockRange.second
           << "): begin " << BlockInfo.Begin << ", end " << BlockInfo.End
           << ", livein " << BlockInfo.LiveIn << ", liveout "
           << BlockInfo.LiveOut << "\n";
  }
}

LLVM_DUMP_METHOD void StackLifetime::dumpLiveRanges() const {
  dbgs() << "Alloca liveness:\n";
  for (unsigned AllocaNo = 0; AllocaNo < NumAllocas; ++AllocaNo)
    dbgs() << "  " << AllocaNo << ": " << LiveRanges[AllocaNo] << "\n";
}
#endif

StackLifetime::StackLifetime(const Function &F,
                             ArrayRef<const AllocaInst *> Allocas)
    : F(F), Allocas(Allocas), NumAllocas(Allocas.size()) {
  LLVM_DEBUG(dumpAllocas());

  for (unsigned I = 0; I < NumAllocas; ++I)
    AllocaNumbering[Allocas[I]] = I;

  collectMarkers();
}

void StackLifetime::run() {
  LiveRanges.resize(NumAllocas, LiveRange(NumInst));
  for (unsigned I = 0; I < NumAllocas; ++I)
    if (!InterestingAllocas.test(I))
      LiveRanges[I] = getFullLiveRange();

  calculateLocalLiveness();
  LLVM_DEBUG(dumpBlockLiveness());
  calculateLiveIntervals();
  LLVM_DEBUG(dumpLiveRanges());
}
