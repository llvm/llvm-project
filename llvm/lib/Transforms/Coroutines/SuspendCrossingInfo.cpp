//===- SuspendCrossingInfo.cpp - Utility for suspend crossing values ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// The SuspendCrossingInfo maintains data that allows to answer a question
// whether given two BasicBlocks A and B there is a path from A to B that
// passes through a suspend point. Note, SuspendCrossingInfo is invalidated
// by changes to the CFG including adding/removing BBs due to its use of BB
// ptrs in the BlockToIndexMapping.
//===----------------------------------------------------------------------===//

#include "SuspendCrossingInfo.h"

// The "coro-suspend-crossing" flag is very noisy. There is another debug type,
// "coro-frame", which results in leaner debug spew.
#define DEBUG_TYPE "coro-suspend-crossing"

namespace llvm {
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
static std::string getBasicBlockLabel(const BasicBlock *BB) {
  if (BB->hasName())
    return BB->getName().str();

  std::string S;
  raw_string_ostream OS(S);
  BB->printAsOperand(OS, false);
  return OS.str().substr(1);
}

LLVM_DUMP_METHOD void SuspendCrossingInfo::dump(
    StringRef Label, BitVector const &BV,
    const ReversePostOrderTraversal<Function *> &RPOT) const {
  dbgs() << Label << ":";
  for (const BasicBlock *BB : RPOT) {
    auto BBNo = Mapping.blockToIndex(BB);
    if (BV[BBNo])
      dbgs() << " " << getBasicBlockLabel(BB);
  }
  dbgs() << "\n";
}

LLVM_DUMP_METHOD void SuspendCrossingInfo::dump() const {
  if (Block.empty())
    return;

  BasicBlock *const B = Mapping.indexToBlock(0);
  Function *F = B->getParent();

  ReversePostOrderTraversal<Function *> RPOT(F);
  for (const BasicBlock *BB : RPOT) {
    auto BBNo = Mapping.blockToIndex(BB);
    dbgs() << getBasicBlockLabel(BB) << ":\n";
    dump("   Consumes", Block[BBNo].Consumes, RPOT);
    dump("      Kills", Block[BBNo].Kills, RPOT);
  }
  dbgs() << "\n";
}
#endif

bool SuspendCrossingInfo::hasPathCrossingSuspendPoint(BasicBlock *From,
                                                      BasicBlock *To) const {
  size_t const FromIndex = Mapping.blockToIndex(From);
  size_t const ToIndex = Mapping.blockToIndex(To);
  bool const Result = Block[ToIndex].Kills[FromIndex];
  LLVM_DEBUG(if (Result) dbgs() << From->getName() << " => " << To->getName()
                                << " crosses suspend point\n");
  return Result;
}

bool SuspendCrossingInfo::hasPathOrLoopCrossingSuspendPoint(
    BasicBlock *From, BasicBlock *To) const {
  size_t const FromIndex = Mapping.blockToIndex(From);
  size_t const ToIndex = Mapping.blockToIndex(To);
  bool Result = Block[ToIndex].Kills[FromIndex] ||
                (From == To && Block[ToIndex].KillLoop);
  LLVM_DEBUG(if (Result) dbgs() << From->getName() << " => " << To->getName()
                                << " crosses suspend point (path or loop)\n");
  return Result;
}

template <bool Initialize>
bool SuspendCrossingInfo::computeBlockData(
    const ReversePostOrderTraversal<Function *> &RPOT) {
  bool Changed = false;

  for (const BasicBlock *BB : RPOT) {
    auto BBNo = Mapping.blockToIndex(BB);
    auto &B = Block[BBNo];

    // We don't need to count the predecessors when initialization.
    if constexpr (!Initialize)
      // If all the predecessors of the current Block don't change,
      // the BlockData for the current block must not change too.
      if (all_of(predecessors(B), [this](BasicBlock *BB) {
            return !Block[Mapping.blockToIndex(BB)].Changed;
          })) {
        B.Changed = false;
        continue;
      }

    // Saved Consumes and Kills bitsets so that it is easy to see
    // if anything changed after propagation.
    auto SavedConsumes = B.Consumes;
    auto SavedKills = B.Kills;

    for (BasicBlock *PI : predecessors(B)) {
      auto PrevNo = Mapping.blockToIndex(PI);
      auto &P = Block[PrevNo];

      // Propagate Kills and Consumes from predecessors into B.
      B.Consumes |= P.Consumes;
      B.Kills |= P.Kills;

      // If block P is a suspend block, it should propagate kills into block
      // B for every block P consumes.
      if (P.Suspend)
        B.Kills |= P.Consumes;
    }

    if (B.Suspend) {
      // If block B is a suspend block, it should kill all of the blocks it
      // consumes.
      B.Kills |= B.Consumes;
    } else if (B.End) {
      // If block B is an end block, it should not propagate kills as the
      // blocks following coro.end() are reached during initial invocation
      // of the coroutine while all the data are still available on the
      // stack or in the registers.
      B.Kills.reset();
    } else {
      // This is reached when B block it not Suspend nor coro.end and it
      // need to make sure that it is not in the kill set.
      B.KillLoop |= B.Kills[BBNo];
      B.Kills.reset(BBNo);
    }

    if constexpr (!Initialize) {
      B.Changed = (B.Kills != SavedKills) || (B.Consumes != SavedConsumes);
      Changed |= B.Changed;
    }
  }

  return Changed;
}

SuspendCrossingInfo::SuspendCrossingInfo(
    Function &F, const SmallVectorImpl<AnyCoroSuspendInst *> &CoroSuspends,
    const SmallVectorImpl<AnyCoroEndInst *> &CoroEnds)
    : Mapping(F) {
  const size_t N = Mapping.size();
  Block.resize(N);

  // Initialize every block so that it consumes itself
  for (size_t I = 0; I < N; ++I) {
    auto &B = Block[I];
    B.Consumes.resize(N);
    B.Kills.resize(N);
    B.Consumes.set(I);
    B.Changed = true;
  }

  // Mark all CoroEnd Blocks. We do not propagate Kills beyond coro.ends as
  // the code beyond coro.end is reachable during initial invocation of the
  // coroutine.
  for (auto *CE : CoroEnds) {
    // Verify CoroEnd was normalized
    assert(CE->getParent()->getFirstInsertionPt() == CE->getIterator() &&
           CE->getParent()->size() <= 2 && "CoroEnd must be in its own BB");

    getBlockData(CE->getParent()).End = true;
  }

  // Mark all suspend blocks and indicate that they kill everything they
  // consume. Note, that crossing coro.save also requires a spill, as any code
  // between coro.save and coro.suspend may resume the coroutine and all of the
  // state needs to be saved by that time.
  auto markSuspendBlock = [&](IntrinsicInst *BarrierInst) {
    BasicBlock *SuspendBlock = BarrierInst->getParent();
    auto &B = getBlockData(SuspendBlock);
    B.Suspend = true;
    B.Kills |= B.Consumes;
  };
  for (auto *CSI : CoroSuspends) {
    // Verify CoroSuspend was normalized
    assert(CSI->getParent()->getFirstInsertionPt() == CSI->getIterator() &&
           CSI->getParent()->size() <= 2 &&
           "CoroSuspend must be in its own BB");

    markSuspendBlock(CSI);
    if (auto *Save = CSI->getCoroSave())
      markSuspendBlock(Save);
  }

  // It is considered to be faster to use RPO traversal for forward-edges
  // dataflow analysis.
  ReversePostOrderTraversal<Function *> RPOT(&F);
  computeBlockData</*Initialize=*/true>(RPOT);
  while (computeBlockData</*Initialize*/ false>(RPOT))
    ;

  LLVM_DEBUG(dump());
}

} // namespace llvm
