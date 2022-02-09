//===--- SyncDependenceAnalysis.cpp - Compute Control Divergence Effects --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an algorithm that returns for a divergent branch
// the set of basic blocks whose phi nodes become divergent due to divergent
// control. These are the blocks that are reachable by two disjoint paths from
// the branch or loop exits that have a reaching path that is disjoint from a
// path to the loop latch.
//
// The SyncDependenceAnalysis is used in the DivergenceAnalysis to model
// control-induced divergence in phi nodes.
//
//
// -- Reference --
// The algorithm is presented in Section 5 of 
//
//   An abstract interpretation for SPMD divergence
//       on reducible control flow graphs.
//   Julian Rosemann, Simon Moll and Sebastian Hack
//   POPL '21
//
//
// -- Sync dependence --
// Sync dependence characterizes the control flow aspect of the
// propagation of branch divergence. For example,
//
//   %cond = icmp slt i32 %tid, 10
//   br i1 %cond, label %then, label %else
// then:
//   br label %merge
// else:
//   br label %merge
// merge:
//   %a = phi i32 [ 0, %then ], [ 1, %else ]
//
// Suppose %tid holds the thread ID. Although %a is not data dependent on %tid
// because %tid is not on its use-def chains, %a is sync dependent on %tid
// because the branch "br i1 %cond" depends on %tid and affects which value %a
// is assigned to.
//
//
// -- Reduction to SSA construction --
// There are two disjoint paths from A to X, if a certain variant of SSA
// construction places a phi node in X under the following set-up scheme.
//
// This variant of SSA construction ignores incoming undef values.
// That is paths from the entry without a definition do not result in
// phi nodes.
//
//       entry
//     /      \
//    A        \
//  /   \       Y
// B     C     /
//  \   /  \  /
//    D     E
//     \   /
//       F
//
// Assume that A contains a divergent branch. We are interested
// in the set of all blocks where each block is reachable from A
// via two disjoint paths. This would be the set {D, F} in this
// case.
// To generally reduce this query to SSA construction we introduce
// a virtual variable x and assign to x different values in each
// successor block of A.
//
//           entry
//         /      \
//        A        \
//      /   \       Y
// x = 0   x = 1   /
//      \  /   \  /
//        D     E
//         \   /
//           F
//
// Our flavor of SSA construction for x will construct the following
//
//            entry
//          /      \
//         A        \
//       /   \       Y
// x0 = 0   x1 = 1  /
//       \   /   \ /
//     x2 = phi   E
//         \     /
//         x3 = phi
//
// The blocks D and F contain phi nodes and are thus each reachable
// by two disjoins paths from A.
//
// -- Remarks --
// * In case of loop exits we need to check the disjoint path criterion for loops.
//   To this end, we check whether the definition of x differs between the
//   loop exit and the loop header (_after_ SSA construction).
//
// -- Known Limitations & Future Work --
// * The algorithm requires reducible loops because the implementation
//   implicitly performs a single iteration of the underlying data flow analysis.
//   This was done for pragmatism, simplicity and speed.
//
//   Relevant related work for extending the algorithm to irreducible control:
//     A simple algorithm for global data flow analysis problems.
//     Matthew S. Hecht and Jeffrey D. Ullman.
//     SIAM Journal on Computing, 4(4):519–532, December 1975.
//
// * Another reason for requiring reducible loops is that points of
//   synchronization in irreducible loops aren't 'obvious' - there is no unique
//   header where threads 'should' synchronize when entering or coming back
//   around from the latch.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/SyncDependenceAnalysis.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"

#include <functional>
#include <stack>
#include <unordered_set>

#define DEBUG_TYPE "sync-dependence"

// The SDA algorithm operates on a modified CFG - we modify the edges leaving
// loop headers as follows:
//
// * We remove all edges leaving all loop headers.
// * We add additional edges from the loop headers to their exit blocks.
//
// The modification is virtual, that is whenever we visit a loop header we
// pretend it had different successors.
namespace {
using namespace llvm;

// Custom Post-Order Traveral
//
// We cannot use the vanilla (R)PO computation of LLVM because:
// * We (virtually) modify the CFG.
// * We want a loop-compact block enumeration, that is the numbers assigned to
//   blocks of a loop form an interval
//   
using POCB = std::function<void(const BasicBlock &)>;
using VisitedSet = std::set<const BasicBlock *>;
using BlockStack = std::vector<const BasicBlock *>;

// forward
static void computeLoopPO(const LoopInfo &LI, Loop &Loop, POCB CallBack,
                          VisitedSet &Finalized);

// for a nested region (top-level loop or nested loop)
static void computeStackPO(BlockStack &Stack, const LoopInfo &LI, Loop *Loop,
                           POCB CallBack, VisitedSet &Finalized) {
  const auto *LoopHeader = Loop ? Loop->getHeader() : nullptr;
  while (!Stack.empty()) {
    const auto *NextBB = Stack.back();

    auto *NestedLoop = LI.getLoopFor(NextBB);
    bool IsNestedLoop = NestedLoop != Loop;

    // Treat the loop as a node
    if (IsNestedLoop) {
      SmallVector<BasicBlock *, 3> NestedExits;
      NestedLoop->getUniqueExitBlocks(NestedExits);
      bool PushedNodes = false;
      for (const auto *NestedExitBB : NestedExits) {
        if (NestedExitBB == LoopHeader)
          continue;
        if (Loop && !Loop->contains(NestedExitBB))
          continue;
        if (Finalized.count(NestedExitBB))
          continue;
        PushedNodes = true;
        Stack.push_back(NestedExitBB);
      }
      if (!PushedNodes) {
        // All loop exits finalized -> finish this node
        Stack.pop_back();
        computeLoopPO(LI, *NestedLoop, CallBack, Finalized);
      }
      continue;
    }

    // DAG-style
    bool PushedNodes = false;
    for (const auto *SuccBB : successors(NextBB)) {
      if (SuccBB == LoopHeader)
        continue;
      if (Loop && !Loop->contains(SuccBB))
        continue;
      if (Finalized.count(SuccBB))
        continue;
      PushedNodes = true;
      Stack.push_back(SuccBB);
    }
    if (!PushedNodes) {
      // Never push nodes twice
      Stack.pop_back();
      if (!Finalized.insert(NextBB).second)
        continue;
      CallBack(*NextBB);
    }
  }
}

static void computeTopLevelPO(Function &F, const LoopInfo &LI, POCB CallBack) {
  VisitedSet Finalized;
  BlockStack Stack;
  Stack.reserve(24); // FIXME made-up number
  Stack.push_back(&F.getEntryBlock());
  computeStackPO(Stack, LI, nullptr, CallBack, Finalized);
}

static void computeLoopPO(const LoopInfo &LI, Loop &Loop, POCB CallBack,
                          VisitedSet &Finalized) {
  /// Call CallBack on all loop blocks.
  std::vector<const BasicBlock *> Stack;
  const auto *LoopHeader = Loop.getHeader();

  // Visit the header last
  Finalized.insert(LoopHeader);
  CallBack(*LoopHeader);

  // Initialize with immediate successors
  for (const auto *BB : successors(LoopHeader)) {
    if (!Loop.contains(BB))
      continue;
    if (BB == LoopHeader)
      continue;
    Stack.push_back(BB);
  }

  // Compute PO inside region
  computeStackPO(Stack, LI, &Loop, CallBack, Finalized);
}

} // namespace

namespace llvm {

ControlDivergenceDesc SyncDependenceAnalysis::EmptyDivergenceDesc;

SyncDependenceAnalysis::SyncDependenceAnalysis(const DominatorTree &DT,
                                               const PostDominatorTree &PDT,
                                               const LoopInfo &LI)
    : DT(DT), PDT(PDT), LI(LI) {
  computeTopLevelPO(*DT.getRoot()->getParent(), LI,
                    [&](const BasicBlock &BB) { LoopPO.appendBlock(BB); });
}

SyncDependenceAnalysis::~SyncDependenceAnalysis() = default;

// divergence propagator for reducible CFGs
struct DivergencePropagator {
  const ModifiedPO &LoopPOT;
  const DominatorTree &DT;
  const PostDominatorTree &PDT;
  const LoopInfo &LI;
  const BasicBlock &DivTermBlock;

  // * if BlockLabels[IndexOf(B)] == C then C is the dominating definition at
  //   block B
  // * if BlockLabels[IndexOf(B)] ~ undef then we haven't seen B yet
  // * if BlockLabels[IndexOf(B)] == B then B is a join point of disjoint paths
  // from X or B is an immediate successor of X (initial value).
  using BlockLabelVec = std::vector<const BasicBlock *>;
  BlockLabelVec BlockLabels;
  // divergent join and loop exit descriptor.
  std::unique_ptr<ControlDivergenceDesc> DivDesc;

  DivergencePropagator(const ModifiedPO &LoopPOT, const DominatorTree &DT,
                       const PostDominatorTree &PDT, const LoopInfo &LI,
                       const BasicBlock &DivTermBlock)
      : LoopPOT(LoopPOT), DT(DT), PDT(PDT), LI(LI), DivTermBlock(DivTermBlock),
        BlockLabels(LoopPOT.size(), nullptr),
        DivDesc(new ControlDivergenceDesc) {}

  void printDefs(raw_ostream &Out) {
    Out << "Propagator::BlockLabels {\n";
    for (int BlockIdx = (int)BlockLabels.size() - 1; BlockIdx > 0; --BlockIdx) {
      const auto *Label = BlockLabels[BlockIdx];
      Out << LoopPOT.getBlockAt(BlockIdx)->getName().str() << "(" << BlockIdx
          << ") : ";
      if (!Label) {
        Out << "<null>\n";
      } else {
        Out << Label->getName() << "\n";
      }
    }
    Out << "}\n";
  }

  // Push a definition (\p PushedLabel) to \p SuccBlock and return whether this
  // causes a divergent join.
  bool computeJoin(const BasicBlock &SuccBlock, const BasicBlock &PushedLabel) {
    auto SuccIdx = LoopPOT.getIndexOf(SuccBlock);

    // unset or same reaching label
    const auto *OldLabel = BlockLabels[SuccIdx];
    if (!OldLabel || (OldLabel == &PushedLabel)) {
      BlockLabels[SuccIdx] = &PushedLabel;
      return false;
    }

    // Update the definition
    BlockLabels[SuccIdx] = &SuccBlock;
    return true;
  }

  // visiting a virtual loop exit edge from the loop header --> temporal
  // divergence on join
  bool visitLoopExitEdge(const BasicBlock &ExitBlock,
                         const BasicBlock &DefBlock, bool FromParentLoop) {
    // Pushing from a non-parent loop cannot cause temporal divergence.
    if (!FromParentLoop)
      return visitEdge(ExitBlock, DefBlock);

    if (!computeJoin(ExitBlock, DefBlock))
      return false;

    // Identified a divergent loop exit
    DivDesc->LoopDivBlocks.insert(&ExitBlock);
    LLVM_DEBUG(dbgs() << "\tDivergent loop exit: " << ExitBlock.getName()
                      << "\n");
    return true;
  }

  // process \p SuccBlock with reaching definition \p DefBlock
  bool visitEdge(const BasicBlock &SuccBlock, const BasicBlock &DefBlock) {
    if (!computeJoin(SuccBlock, DefBlock))
      return false;

    // Divergent, disjoint paths join.
    DivDesc->JoinDivBlocks.insert(&SuccBlock);
    LLVM_DEBUG(dbgs() << "\tDivergent join: " << SuccBlock.getName());
    return true;
  }

  std::unique_ptr<ControlDivergenceDesc> computeJoinPoints() {
    assert(DivDesc);

    LLVM_DEBUG(dbgs() << "SDA:computeJoinPoints: " << DivTermBlock.getName()
                      << "\n");

    const auto *DivBlockLoop = LI.getLoopFor(&DivTermBlock);

    // Early stopping criterion
    int FloorIdx = LoopPOT.size() - 1;
    const BasicBlock *FloorLabel = nullptr;

    // bootstrap with branch targets
    int BlockIdx = 0;

    for (const auto *SuccBlock : successors(&DivTermBlock)) {
      auto SuccIdx = LoopPOT.getIndexOf(*SuccBlock);
      BlockLabels[SuccIdx] = SuccBlock;

      // Find the successor with the highest index to start with
      BlockIdx = std::max<int>(BlockIdx, SuccIdx);
      FloorIdx = std::min<int>(FloorIdx, SuccIdx);

      // Identify immediate divergent loop exits
      if (!DivBlockLoop)
        continue;

      const auto *BlockLoop = LI.getLoopFor(SuccBlock);
      if (BlockLoop && DivBlockLoop->contains(BlockLoop))
        continue;
      DivDesc->LoopDivBlocks.insert(SuccBlock);
      LLVM_DEBUG(dbgs() << "\tImmediate divergent loop exit: "
                        << SuccBlock->getName() << "\n");
    }

    // propagate definitions at the immediate successors of the node in RPO
    for (; BlockIdx >= FloorIdx; --BlockIdx) {
      LLVM_DEBUG(dbgs() << "Before next visit:\n"; printDefs(dbgs()));

      // Any label available here
      const auto *Label = BlockLabels[BlockIdx];
      if (!Label)
        continue;

      // Ok. Get the block
      const auto *Block = LoopPOT.getBlockAt(BlockIdx);
      LLVM_DEBUG(dbgs() << "SDA::joins. visiting " << Block->getName() << "\n");

      auto *BlockLoop = LI.getLoopFor(Block);
      bool IsLoopHeader = BlockLoop && BlockLoop->getHeader() == Block;
      bool CausedJoin = false;
      int LoweredFloorIdx = FloorIdx;
      if (IsLoopHeader) {
        // Disconnect from immediate successors and propagate directly to loop
        // exits.
        SmallVector<BasicBlock *, 4> BlockLoopExits;
        BlockLoop->getExitBlocks(BlockLoopExits);

        bool IsParentLoop = BlockLoop->contains(&DivTermBlock);
        for (const auto *BlockLoopExit : BlockLoopExits) {
          CausedJoin |= visitLoopExitEdge(*BlockLoopExit, *Label, IsParentLoop);
          LoweredFloorIdx = std::min<int>(LoweredFloorIdx,
                                          LoopPOT.getIndexOf(*BlockLoopExit));
        }
      } else {
        // Acyclic successor case
        for (const auto *SuccBlock : successors(Block)) {
          CausedJoin |= visitEdge(*SuccBlock, *Label);
          LoweredFloorIdx =
              std::min<int>(LoweredFloorIdx, LoopPOT.getIndexOf(*SuccBlock));
        }
      }

      // Floor update
      if (CausedJoin) {
        // 1. Different labels pushed to successors
        FloorIdx = LoweredFloorIdx;
      } else if (FloorLabel != Label) {
        // 2. No join caused BUT we pushed a label that is different than the
        // last pushed label
        FloorIdx = LoweredFloorIdx;
        FloorLabel = Label;
      }
    }

    LLVM_DEBUG(dbgs() << "SDA::joins. After propagation:\n"; printDefs(dbgs()));

    return std::move(DivDesc);
  }
};

#ifndef NDEBUG
static void printBlockSet(ConstBlockSet &Blocks, raw_ostream &Out) {
  Out << "[";
  ListSeparator LS;
  for (const auto *BB : Blocks)
    Out << LS << BB->getName();
  Out << "]";
}
#endif

const ControlDivergenceDesc &
SyncDependenceAnalysis::getJoinBlocks(const Instruction &Term) {
  // trivial case
  if (Term.getNumSuccessors() <= 1) {
    return EmptyDivergenceDesc;
  }

  // already available in cache?
  auto ItCached = CachedControlDivDescs.find(&Term);
  if (ItCached != CachedControlDivDescs.end())
    return *ItCached->second;

  // compute all join points
  // Special handling of divergent loop exits is not needed for LCSSA
  const auto &TermBlock = *Term.getParent();
  DivergencePropagator Propagator(LoopPO, DT, PDT, LI, TermBlock);
  auto DivDesc = Propagator.computeJoinPoints();

  LLVM_DEBUG(dbgs() << "Result (" << Term.getParent()->getName() << "):\n";
             dbgs() << "JoinDivBlocks: ";
             printBlockSet(DivDesc->JoinDivBlocks, dbgs());
             dbgs() << "\nLoopDivBlocks: ";
             printBlockSet(DivDesc->LoopDivBlocks, dbgs()); dbgs() << "\n";);

  auto ItInserted = CachedControlDivDescs.emplace(&Term, std::move(DivDesc));
  assert(ItInserted.second);
  return *ItInserted.first->second;
}

} // namespace llvm
