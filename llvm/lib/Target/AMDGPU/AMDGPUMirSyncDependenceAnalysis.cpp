//===- MirSyncDependenceAnalysis.cpp - Mir Divergent Branch Dependence
//Calculation
//--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is based on Analysis/MirSyncDependenceAnalysis.cpp, just change
// MachineBasicBlock to MachineBasicBlock.
// This file implements an algorithm that returns for a divergent branch
// the set of basic blocks whose phi nodes become divergent due to divergent
// control. These are the blocks that are reachable by two disjoint paths from
// the branch or loop exits that have a reaching path that is disjoint from a
// path to the loop latch.
//
// The SyncDependenceAnalysis is used in the DivergenceAnalysis to model
// control-induced divergence in phi nodes.
//
// -- Summary --
// The SyncDependenceAnalysis lazily computes sync dependences [3].
// The analysis evaluates the disjoint path criterion [2] by a reduction
// to SSA construction. The SSA construction algorithm is implemented as
// a simple data-flow analysis [1].
//
// [1] "A Simple, Fast Dominance Algorithm", SPI '01, Cooper, Harvey and Kennedy
// [2] "Efficiently Computing Static Single Assignment Form
//     and the Control Dependence Graph", TOPLAS '91,
//           Cytron, Ferrante, Rosen, Wegman and Zadeck
// [3] "Improving Performance of OpenCL on CPUs", CC '12, Karrenberg and Hack
// [4] "Divergence Analysis", TOPLAS '13, Sampaio, Souza, Collange and Pereira
//
// -- Sync dependence --
// Sync dependence [4] characterizes the control flow aspect of the
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
// -- Reduction to SSA construction --
// There are two disjoint paths from A to X, if a certain variant of SSA
// construction places a phi node in X under the following set-up scheme [2].
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
// Assume that A contains a divergent branch. We are interested
// in the set of all blocks where each block is reachable from A
// via two disjoint paths. This would be the set {D, F} in this
// case.
// To generally reduce this query to SSA construction we introduce
// a virtual variable x and assign to x different values in each
// successor block of A.
//           entry
//         /      \
//        A        \
//      /   \       Y
// x = 0   x = 1   /
//      \  /   \  /
//        D     E
//         \   /
//           F
// Our flavor of SSA construction for x will construct the following
//            entry
//          /      \
//         A        \
//       /   \       Y
// x0 = 0   x1 = 1  /
//       \   /   \ /
//      x2=phi    E
//         \     /
//          x3=phi
// The blocks D and F contain phi nodes and are thus each reachable
// by two disjoins paths from A.
//
// -- Remarks --
// In case of loop exits we need to check the disjoint path criterion for loops
// [2]. To this end, we check whether the definition of x differs between the
// loop exit and the loop header (_after_ SSA construction).
//
//===----------------------------------------------------------------------===//
#include "AMDGPUMirSyncDependenceAnalysis.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachinePostDominators.h"

#include <stack>
#include <unordered_set>

#define DEBUG_TYPE "sync-dependence"

namespace llvm {

ConstBlockSet SyncDependenceAnalysis::EmptyBlockSet;

SyncDependenceAnalysis::SyncDependenceAnalysis(
    const MachineDominatorTree &DT, const MachinePostDominatorTree &PDT,
    const MachineLoopInfo &LI,
    // AMDGPU change begin.
    DivergentJoinMapTy &JoinMap
    // AMDGPU change end.
    )
    : FuncRPOT(DT.getRoot()->getParent()), DT(DT), PDT(PDT), LI(LI),
      // AMDGPU change begin.
      DivergentJoinMap(JoinMap)
// AMDGPU change end.
{}

SyncDependenceAnalysis::~SyncDependenceAnalysis() {}

using FunctionRPOT = ReversePostOrderTraversal<const MachineFunction *>;

// divergence propagator for reducible CFGs
struct DivergencePropagator {
  const FunctionRPOT &FuncRPOT;
  const MachineDominatorTree &DT;
  const MachinePostDominatorTree &PDT;
  const MachineLoopInfo &LI;

  // identified join points
  std::unique_ptr<ConstBlockSet> JoinBlocks;

  // reached loop exits (by a path disjoint to a path to the loop header)
  SmallPtrSet<const MachineBasicBlock *, 4> ReachedLoopExits;

  // if DefMap[B] == C then C is the dominating definition at block B
  // if DefMap[B] ~ undef then we haven't seen B yet
  // if DefMap[B] == B then B is a join point of disjoint paths from X or B is
  // an immediate successor of X (initial value).
  using DefiningBlockMap =
      std::map<const MachineBasicBlock *, const MachineBasicBlock *>;
  DefiningBlockMap DefMap;

  // all blocks with pending visits
  std::unordered_set<const MachineBasicBlock *> PendingUpdates;

  DivergencePropagator(const FunctionRPOT &FuncRPOT,
                       const MachineDominatorTree &DT,
                       const MachinePostDominatorTree &PDT,
                       const MachineLoopInfo &LI)
      : FuncRPOT(FuncRPOT), DT(DT), PDT(PDT), LI(LI),
        JoinBlocks(new ConstBlockSet) {}

  // set the definition at @block and mark @block as pending for a visit
  void addPending(const MachineBasicBlock &Block,
                  const MachineBasicBlock &DefBlock) {
    bool WasAdded = DefMap.emplace(&Block, &DefBlock).second;
    if (WasAdded)
      PendingUpdates.insert(&Block);
  }

  void printDefs(raw_ostream &Out) {
    Out << "Propagator::DefMap {\n";
    for (const auto *Block : FuncRPOT) {
      auto It = DefMap.find(Block);
      Out << Block->getName() << " : ";
      if (It == DefMap.end()) {
        Out << "\n";
      } else {
        const auto *DefBlock = It->second;
        Out << (DefBlock ? DefBlock->getName() : "<null>") << "\n";
      }
    }
    Out << "}\n";
  }

  // process @succBlock with reaching definition @defBlock
  // the original divergent branch was in @parentLoop (if any)
  void visitSuccessor(const MachineBasicBlock &SuccBlock,
                      const MachineLoop *ParentLoop,
                      const MachineBasicBlock &DefBlock) {

    // @succBlock is a loop exit
    if (ParentLoop && !ParentLoop->contains(&SuccBlock)) {
      DefMap.emplace(&SuccBlock, &DefBlock);
      ReachedLoopExits.insert(&SuccBlock);
      return;
    }

    // first reaching def?
    auto ItLastDef = DefMap.find(&SuccBlock);
    if (ItLastDef == DefMap.end()) {
      addPending(SuccBlock, DefBlock);
      return;
    }

    // a join of at least two definitions
    if (ItLastDef->second != &DefBlock) {
      // do we know this join already?
      if (!JoinBlocks->insert(&SuccBlock).second)
        return;

      // update the definition
      addPending(SuccBlock, SuccBlock);
    }
  }

  // find all blocks reachable by two disjoint paths from @rootTerm.
  // This method works for both divergent terminators and loops with
  // divergent exits.
  // @rootBlock is either the block containing the branch or the header of the
  // divergent loop.
  // @nodeSuccessors is the set of successors of the node (MachineLoop or
  // Terminator) headed by @rootBlock.
  // @parentLoop is the parent loop of the MachineLoop or the loop that contains
  // the Terminator.
  template <typename SuccessorIterable>
  std::unique_ptr<ConstBlockSet> computeJoinPoints(
      const MachineBasicBlock &RootBlock, SuccessorIterable NodeSuccessors,
      const MachineLoop *ParentLoop, const MachineBasicBlock *PdBoundBlock) {
    assert(JoinBlocks);

    // bootstrap with branch targets
    for (const auto *SuccBlock : NodeSuccessors) {
      DefMap.emplace(SuccBlock, SuccBlock);

      if (ParentLoop && !ParentLoop->contains(SuccBlock)) {
        // immediate loop exit from node.
        ReachedLoopExits.insert(SuccBlock);
        continue;
      } else {
        // regular successor
        PendingUpdates.insert(SuccBlock);
      }
    }

    auto ItBeginRPO = FuncRPOT.begin();

    // skip until term (TODO RPOT won't let us start at @term directly)
    for (; *ItBeginRPO != &RootBlock; ++ItBeginRPO) {
    }

    auto ItEndRPO = FuncRPOT.end();
    assert(ItBeginRPO != ItEndRPO);

    // propagate definitions at the immediate successors of the node in RPO
    auto ItBlockRPO = ItBeginRPO;
    while (++ItBlockRPO != ItEndRPO && *ItBlockRPO != PdBoundBlock) {
      const auto *Block = *ItBlockRPO;

      // skip @block if not pending update
      auto ItPending = PendingUpdates.find(Block);
      if (ItPending == PendingUpdates.end())
        continue;
      PendingUpdates.erase(ItPending);

      // propagate definition at @block to its successors
      auto ItDef = DefMap.find(Block);
      const auto *DefBlock = ItDef->second;
      assert(DefBlock);

      auto *BlockLoop = LI.getLoopFor(Block);
      if (ParentLoop &&
          (ParentLoop != BlockLoop && ParentLoop->contains(BlockLoop))) {
        // if the successor is the header of a nested loop pretend its a
        // single node with the loop's exits as successors
        SmallVector<MachineBasicBlock *, 4> BlockLoopExits;
        BlockLoop->getExitBlocks(BlockLoopExits);
        for (const auto *BlockLoopExit : BlockLoopExits) {
          visitSuccessor(*BlockLoopExit, ParentLoop, *DefBlock);
        }

      } else {
        // the successors are either on the same loop level or loop exits
        for (const auto *SuccBlock : Block->successors()) {
          visitSuccessor(*SuccBlock, ParentLoop, *DefBlock);
        }
      }
    }

    // We need to know the definition at the parent loop header to decide
    // whether the definition at the header is different from the definition at
    // the loop exits, which would indicate a divergent loop exits.
    //
    // A // loop header
    // |
    // B // nested loop header
    // |
    // C -> X (exit from B loop) -..-> (A latch)
    // |
    // D -> back to B (B latch)
    // |
    // proper exit from both loops
    //
    // D post-dominates B as it is the only proper exit from the "A loop".
    // If C has a divergent branch, propagation will therefore stop at D.
    // That implies that B will never receive a definition.
    // But that definition can only be the same as at D (D itself in thise case)
    // because all paths to anywhere have to pass through D.
    //
    const MachineBasicBlock *ParentLoopHeader =
        ParentLoop ? ParentLoop->getHeader() : nullptr;
    if (ParentLoop && ParentLoop->contains(PdBoundBlock)) {
      DefMap[ParentLoopHeader] = DefMap[PdBoundBlock];
    }

    // analyze reached loop exits
    if (!ReachedLoopExits.empty()) {
      assert(ParentLoop);
      const auto *HeaderDefBlock = DefMap[ParentLoopHeader];
      LLVM_DEBUG(printDefs(dbgs()));

      // AMDGPU CHANGE: Allow null HeaderDefBlock
      // Because of the way they walk the blocks (a reverse post order traversal
      // stopping at the immediate post dominator) it is possible that
      // they will reach a loop exit, but not the loop header.
      //
      // We conservatively mark the exit blocks as divergent join points
      // in this case.
      //
      // Problem CFG is below:
      //
      //     +--> A
      //     |   / \
      //     |  B   C
      //     |  | / |
      //     +--L   P
      //
      // In this cfg, C is the RootBlock and P is C's post-dominator.
      // It will only visit L and P and then stop because it hits the
      // post dominator. Most loops do not hit this case because the
      // loop exiting block (C) will branch directly back to the loop
      // header.
      //
      if (HeaderDefBlock) {
        for (const auto *ExitBlock : ReachedLoopExits) {
          auto ItExitDef = DefMap.find(ExitBlock);
          assert((ItExitDef != DefMap.end()) &&
                 "no reaching def at reachable loop exit");
          if (ItExitDef->second != HeaderDefBlock) {
            JoinBlocks->insert(ExitBlock);
          }
        }
      } else {
        for (const auto *ExitBlock : ReachedLoopExits) {
          JoinBlocks->insert(ExitBlock);
        }
      }
    }

    return std::move(JoinBlocks);
  }
};

// AMDGPU change begin.
// For all join blocks caused by divergent RootBlock, the prevs of a join block
// which are in DefMap or the RootBlock are divergent join each other on the
// join block because of divergent RootBlock.
static void
updateJoinMap(const MachineBasicBlock *RootBlock,
              DenseMap<const MachineBasicBlock *,
                       SmallPtrSet<const MachineBasicBlock *, 4>> &JoinMap,
              DivergencePropagator::DefiningBlockMap &DefMap,
              ConstBlockSet &JoinBlocks) {
  for (const MachineBasicBlock *JoinBB : JoinBlocks) {
    // makr divergent join for all pred pair which in DefMap.
    for (auto predIt = JoinBB->pred_begin(); predIt != JoinBB->pred_end();
         predIt++) {
      auto predIt2 = predIt;
      const MachineBasicBlock *pred = *predIt;
      if (DefMap.count(pred) == 0 && pred != RootBlock)
        continue;

      for (predIt2++; predIt2 != JoinBB->pred_end(); predIt2++) {
        const MachineBasicBlock *pred2 = *predIt2;
        if (DefMap.count(pred2) == 0 && pred2 != RootBlock)
          continue;

        JoinMap[pred].insert(pred2);
        JoinMap[pred2].insert(pred);
        LLVM_DEBUG(dbgs() << "joint_bb0: " << pred->getName()
                          << " joint_bb1: " << pred2->getName() << "\n";);
      }
    }
  }
}
// AMDGPU change end.

const ConstBlockSet &
SyncDependenceAnalysis::join_blocks(const MachineLoop &MachineLoop) {
  using LoopExitVec = SmallVector<MachineBasicBlock *, 4>;
  LoopExitVec LoopExits;
  MachineLoop.getExitBlocks(LoopExits);
  if (LoopExits.size() < 1) {
    return EmptyBlockSet;
  }

  // already available in cache?
  auto ItCached = CachedLoopExitJoins.find(&MachineLoop);
  if (ItCached != CachedLoopExitJoins.end()) {
    return *ItCached->second;
  }

  // dont propagte beyond the immediate post dom of the loop
  const auto *PdNode =
      PDT.getNode(const_cast<MachineBasicBlock *>(MachineLoop.getHeader()));
  const auto *IpdNode = PdNode->getIDom();
  const auto *PdBoundBlock = IpdNode ? IpdNode->getBlock() : nullptr;
  while (PdBoundBlock && MachineLoop.contains(PdBoundBlock)) {
    IpdNode = IpdNode->getIDom();
    PdBoundBlock = IpdNode ? IpdNode->getBlock() : nullptr;
  }

  // compute all join points
  DivergencePropagator Propagator{FuncRPOT, DT, PDT, LI};
  auto JoinBlocks = Propagator.computeJoinPoints<const LoopExitVec &>(
      *MachineLoop.getHeader(), LoopExits, MachineLoop.getParentLoop(),
      PdBoundBlock);

  // AMDGPU change begin.
  // Save divergent join pairs.
  updateJoinMap(MachineLoop.getHeader(), DivergentJoinMap, Propagator.DefMap,
                *JoinBlocks.get());
  // AMDGPU change end.

  auto ItInserted =
      CachedLoopExitJoins.emplace(&MachineLoop, std::move(JoinBlocks));
  assert(ItInserted.second);
  return *ItInserted.first->second;
}

const ConstBlockSet &
SyncDependenceAnalysis::join_blocks(const MachineInstr &Term) {
  // trivial case
  if (Term.getParent()->succ_size() < 1) {
    return EmptyBlockSet;
  }

  // already available in cache?
  auto ItCached = CachedBranchJoins.find(&Term);
  if (ItCached != CachedBranchJoins.end())
    return *ItCached->second;

  // dont propagate beyond the immediate post dominator of the branch
  const auto *PdNode =
      PDT.getNode(const_cast<MachineBasicBlock *>(Term.getParent()));
  const auto *IpdNode = PdNode->getIDom();
  const auto *PdBoundBlock = IpdNode ? IpdNode->getBlock() : nullptr;

  // compute all join points
  DivergencePropagator Propagator{FuncRPOT, DT, PDT, LI};
  const auto &TermBlock = *Term.getParent();

  // AMDGPU CHANGE
  // Make sure the post-dominator is outside the loop for the loop header.
  // Otherwise, we may not find all the join blocks in the loop
  // because the search stops too early. Some join points can be reached
  // after the post-dominator!
  //
  // Problem CFG is below:
  //
  //     +--> A
  //     |   / \
  //     |  B   P
  //     |  | / |
  //     +--L   X
  //
  // In this cfg, A is the loop header and P is A's post-dominator.
  // The algorithm to mark join points does an Reverse Post Order walk
  // from A and stops when it reaches the post dominator. It would not
  // mark the phi node in L as divergent even when A had a divergent branch.
  // The fix we made was to make the join point search continue all the way
  // to the loops post dominator (which is X in this example).
  //
  // NOTE: They already made this change for the loop case above, but for
  //       a different bug apparently. See
  //       SyncDependenceAnalysis::join_blocks(MachineLoop&)
  //
  const MachineLoop *MachineLoop = LI.getLoopFor(&TermBlock);
  if (MachineLoop && (MachineLoop->getHeader() == &TermBlock)) {
    while (PdBoundBlock && MachineLoop->contains(PdBoundBlock)) {
      IpdNode = IpdNode->getIDom();
      PdBoundBlock = IpdNode ? IpdNode->getBlock() : nullptr;
    }
  }

  auto JoinBlocks = Propagator.computeJoinPoints(
      TermBlock, Term.getParent()->successors(), MachineLoop, PdBoundBlock);

  // AMDGPU change begin.
  // Save divergent join pairs.
  updateJoinMap(&TermBlock, DivergentJoinMap, Propagator.DefMap,
                *JoinBlocks.get());
  // AMDGPU change end.

  auto ItInserted = CachedBranchJoins.emplace(&Term, std::move(JoinBlocks));
  assert(ItInserted.second);
  return *ItInserted.first->second;
}

} // namespace llvm
