//===- SimpleLoopUnswitch.cpp - Hoist loop-invariant control flow ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/GenericDomTree.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Scalar/SimpleLoopUnswitch.h"
#include <algorithm>
#include <cassert>
#include <iterator>
#include <utility>

#define DEBUG_TYPE "simple-loop-unswitch"

using namespace llvm;

STATISTIC(NumBranches, "Number of branches unswitched");
STATISTIC(NumSwitches, "Number of switches unswitched");
STATISTIC(NumTrivial, "Number of unswitches that are trivial");

static void replaceLoopUsesWithConstant(Loop &L, Value &LIC,
                                        Constant &Replacement) {
  assert(!isa<Constant>(LIC) && "Why are we unswitching on a constant?");

  // Replace uses of LIC in the loop with the given constant.
  for (auto UI = LIC.use_begin(), UE = LIC.use_end(); UI != UE;) {
    // Grab the use and walk past it so we can clobber it in the use list.
    Use *U = &*UI++;
    Instruction *UserI = dyn_cast<Instruction>(U->getUser());
    if (!UserI || !L.contains(UserI))
      continue;

    // Replace this use within the loop body.
    *U = &Replacement;
  }
}

/// Update the dominator tree after removing one exiting predecessor of a loop
/// exit block.
static void updateLoopExitIDom(BasicBlock *LoopExitBB, Loop &L,
                               DominatorTree &DT) {
  assert(pred_begin(LoopExitBB) != pred_end(LoopExitBB) &&
         "Cannot have empty predecessors of the loop exit block if we split "
         "off a block to unswitch!");

  BasicBlock *IDom = *pred_begin(LoopExitBB);
  // Walk all of the other predecessors finding the nearest common dominator
  // until all predecessors are covered or we reach the loop header. The loop
  // header necessarily dominates all loop exit blocks in loop simplified form
  // so we can early-exit the moment we hit that block.
  for (auto PI = std::next(pred_begin(LoopExitBB)), PE = pred_end(LoopExitBB);
       PI != PE && IDom != L.getHeader(); ++PI)
    IDom = DT.findNearestCommonDominator(IDom, *PI);

  DT.changeImmediateDominator(LoopExitBB, IDom);
}

/// Update the dominator tree after unswitching a particular former exit block.
///
/// This handles the full update of the dominator tree after hoisting a block
/// that previously was an exit block (or split off of an exit block) up to be
/// reached from the new immediate dominator of the preheader.
///
/// The common case is simple -- we just move the unswitched block to have an
/// immediate dominator of the old preheader. But in complex cases, there may
/// be other blocks reachable from the unswitched block that are immediately
/// dominated by some node between the unswitched one and the old preheader.
/// All of these also need to be hoisted in the dominator tree. We also want to
/// minimize queries to the dominator tree because each step of this
/// invalidates any DFS numbers that would make queries fast.
static void updateDTAfterUnswitch(BasicBlock *UnswitchedBB, BasicBlock *OldPH,
                                  DominatorTree &DT) {
  DomTreeNode *OldPHNode = DT[OldPH];
  DomTreeNode *UnswitchedNode = DT[UnswitchedBB];
  // If the dominator tree has already been updated for this unswitched node,
  // we're done. This makes it easier to use this routine if there are multiple
  // paths to the same unswitched destination.
  if (UnswitchedNode->getIDom() == OldPHNode)
    return;

  // First collect the domtree nodes that we are hoisting over. These are the
  // set of nodes which may have children that need to be hoisted as well.
  SmallPtrSet<DomTreeNode *, 4> DomChain;
  for (auto *IDom = UnswitchedNode->getIDom(); IDom != OldPHNode;
       IDom = IDom->getIDom())
    DomChain.insert(IDom);

  // The unswitched block ends up immediately dominated by the old preheader --
  // regardless of whether it is the loop exit block or split off of the loop
  // exit block.
  DT.changeImmediateDominator(UnswitchedNode, OldPHNode);

  // For everything that moves up the dominator tree, we need to examine the
  // dominator frontier to see if it additionally should move up the dominator
  // tree. This lambda appends the dominator frontier for a node on the
  // worklist.
  //
  // Note that we don't currently use the IDFCalculator here for two reasons:
  // 1) It computes dominator tree levels for the entire function on each run
  //    of 'compute'. While this isn't terrible, given that we expect to update
  //    relatively small subtrees of the domtree, it isn't necessarily the right
  //    tradeoff.
  // 2) The interface doesn't fit this usage well. It doesn't operate in
  //    append-only, and builds several sets that we don't need.
  //
  // FIXME: Neither of these issues are a big deal and could be addressed with
  // some amount of refactoring of IDFCalculator. That would allow us to share
  // the core logic here (which is solving the same core problem).
  SmallSetVector<BasicBlock *, 4> Worklist;
  SmallVector<DomTreeNode *, 4> DomNodes;
  SmallPtrSet<BasicBlock *, 4> DomSet;
  auto AppendDomFrontier = [&](DomTreeNode *Node) {
    assert(DomNodes.empty() && "Must start with no dominator nodes.");
    assert(DomSet.empty() && "Must start with an empty dominator set.");

    // First flatten this subtree into sequence of nodes by doing a pre-order
    // walk.
    DomNodes.push_back(Node);
    // We intentionally re-evaluate the size as each node can add new children.
    // Because this is a tree walk, this cannot add any duplicates.
    for (int i = 0; i < (int)DomNodes.size(); ++i)
      DomNodes.insert(DomNodes.end(), DomNodes[i]->begin(), DomNodes[i]->end());

    // Now create a set of the basic blocks so we can quickly test for
    // dominated successors. We could in theory use the DFS numbers of the
    // dominator tree for this, but we want this to remain predictably fast
    // even while we mutate the dominator tree in ways that would invalidate
    // the DFS numbering.
    for (DomTreeNode *InnerN : DomNodes)
      DomSet.insert(InnerN->getBlock());

    // Now re-walk the nodes, appending every successor of every node that isn't
    // in the set. Note that we don't append the node itself, even though if it
    // is a successor it does not strictly dominate itself and thus it would be
    // part of the dominance frontier. The reason we don't append it is that
    // the node passed in came *from* the worklist and so it has already been
    // processed.
    for (DomTreeNode *InnerN : DomNodes)
      for (BasicBlock *SuccBB : successors(InnerN->getBlock()))
        if (!DomSet.count(SuccBB))
          Worklist.insert(SuccBB);

    DomNodes.clear();
    DomSet.clear();
  };

  // Append the initial dom frontier nodes.
  AppendDomFrontier(UnswitchedNode);

  // Walk the worklist. We grow the list in the loop and so must recompute size.
  for (int i = 0; i < (int)Worklist.size(); ++i) {
    auto *BB = Worklist[i];

    DomTreeNode *Node = DT[BB];
    assert(!DomChain.count(Node) &&
           "Cannot be dominated by a block you can reach!");

    // If this block had an immediate dominator somewhere in the chain
    // we hoisted over, then its position in the domtree needs to move as it is
    // reachable from a node hoisted over this chain.
    if (!DomChain.count(Node->getIDom()))
      continue;

    DT.changeImmediateDominator(Node, OldPHNode);

    // Now add this node's dominator frontier to the worklist as well.
    AppendDomFrontier(Node);
  }
}

/// Check that all the LCSSA PHI nodes in the loop exit block have trivial
/// incoming values along this edge.
static bool areLoopExitPHIsLoopInvariant(Loop &L, BasicBlock &ExitingBB,
                                         BasicBlock &ExitBB) {
  for (Instruction &I : ExitBB) {
    auto *PN = dyn_cast<PHINode>(&I);
    if (!PN)
      // No more PHIs to check.
      return true;

    // If the incoming value for this edge isn't loop invariant the unswitch
    // won't be trivial.
    if (!L.isLoopInvariant(PN->getIncomingValueForBlock(&ExitingBB)))
      return false;
  }
  llvm_unreachable("Basic blocks should never be empty!");
}

/// Rewrite the PHI nodes in an unswitched loop exit basic block.
///
/// Requires that the loop exit and unswitched basic block are the same, and
/// that the exiting block was a unique predecessor of that block. Rewrites the
/// PHI nodes in that block such that what were LCSSA PHI nodes become trivial
/// PHI nodes from the old preheader that now contains the unswitched
/// terminator.
static void rewritePHINodesForUnswitchedExitBlock(BasicBlock &UnswitchedBB,
                                                  BasicBlock &OldExitingBB,
                                                  BasicBlock &OldPH) {
  for (Instruction &I : UnswitchedBB) {
    auto *PN = dyn_cast<PHINode>(&I);
    if (!PN)
      // No more PHIs to check.
      break;

    // When the loop exit is directly unswitched we just need to update the
    // incoming basic block. We loop to handle weird cases with repeated
    // incoming blocks, but expect to typically only have one operand here.
    for (auto i : seq<int>(0, PN->getNumOperands())) {
      assert(PN->getIncomingBlock(i) == &OldExitingBB &&
             "Found incoming block different from unique predecessor!");
      PN->setIncomingBlock(i, &OldPH);
    }
  }
}

/// Rewrite the PHI nodes in the loop exit basic block and the split off
/// unswitched block.
///
/// Because the exit block remains an exit from the loop, this rewrites the
/// LCSSA PHI nodes in it to remove the unswitched edge and introduces PHI
/// nodes into the unswitched basic block to select between the value in the
/// old preheader and the loop exit.
static void rewritePHINodesForExitAndUnswitchedBlocks(BasicBlock &ExitBB,
                                                      BasicBlock &UnswitchedBB,
                                                      BasicBlock &OldExitingBB,
                                                      BasicBlock &OldPH) {
  assert(&ExitBB != &UnswitchedBB &&
         "Must have different loop exit and unswitched blocks!");
  Instruction *InsertPt = &*UnswitchedBB.begin();
  for (Instruction &I : ExitBB) {
    auto *PN = dyn_cast<PHINode>(&I);
    if (!PN)
      // No more PHIs to check.
      break;

    auto *NewPN = PHINode::Create(PN->getType(), /*NumReservedValues*/ 2,
                                  PN->getName() + ".split", InsertPt);

    // Walk backwards over the old PHI node's inputs to minimize the cost of
    // removing each one. We have to do this weird loop manually so that we
    // create the same number of new incoming edges in the new PHI as we expect
    // each case-based edge to be included in the unswitched switch in some
    // cases.
    // FIXME: This is really, really gross. It would be much cleaner if LLVM
    // allowed us to create a single entry for a predecessor block without
    // having separate entries for each "edge" even though these edges are
    // required to produce identical results.
    for (int i = PN->getNumIncomingValues() - 1; i >= 0; --i) {
      if (PN->getIncomingBlock(i) != &OldExitingBB)
        continue;

      Value *Incoming = PN->removeIncomingValue(i);
      NewPN->addIncoming(Incoming, &OldPH);
    }

    // Now replace the old PHI with the new one and wire the old one in as an
    // input to the new one.
    PN->replaceAllUsesWith(NewPN);
    NewPN->addIncoming(PN, &ExitBB);
  }
}

/// Unswitch a trivial branch if the condition is loop invariant.
///
/// This routine should only be called when loop code leading to the branch has
/// been validated as trivial (no side effects). This routine checks if the
/// condition is invariant and one of the successors is a loop exit. This
/// allows us to unswitch without duplicating the loop, making it trivial.
///
/// If this routine fails to unswitch the branch it returns false.
///
/// If the branch can be unswitched, this routine splits the preheader and
/// hoists the branch above that split. Preserves loop simplified form
/// (splitting the exit block as necessary). It simplifies the branch within
/// the loop to an unconditional branch but doesn't remove it entirely. Further
/// cleanup can be done with some simplify-cfg like pass.
static bool unswitchTrivialBranch(Loop &L, BranchInst &BI, DominatorTree &DT,
                                  LoopInfo &LI) {
  assert(BI.isConditional() && "Can only unswitch a conditional branch!");
  DEBUG(dbgs() << "  Trying to unswitch branch: " << BI << "\n");

  Value *LoopCond = BI.getCondition();

  // Need a trivial loop condition to unswitch.
  if (!L.isLoopInvariant(LoopCond))
    return false;

  // FIXME: We should compute this once at the start and update it!
  SmallVector<BasicBlock *, 16> ExitBlocks;
  L.getExitBlocks(ExitBlocks);
  SmallPtrSet<BasicBlock *, 16> ExitBlockSet(ExitBlocks.begin(),
                                             ExitBlocks.end());

  // Check to see if a successor of the branch is guaranteed to
  // exit through a unique exit block without having any
  // side-effects.  If so, determine the value of Cond that causes
  // it to do this.
  ConstantInt *CondVal = ConstantInt::getTrue(BI.getContext());
  ConstantInt *Replacement = ConstantInt::getFalse(BI.getContext());
  int LoopExitSuccIdx = 0;
  auto *LoopExitBB = BI.getSuccessor(0);
  if (!ExitBlockSet.count(LoopExitBB)) {
    std::swap(CondVal, Replacement);
    LoopExitSuccIdx = 1;
    LoopExitBB = BI.getSuccessor(1);
    if (!ExitBlockSet.count(LoopExitBB))
      return false;
  }
  auto *ContinueBB = BI.getSuccessor(1 - LoopExitSuccIdx);
  assert(L.contains(ContinueBB) &&
         "Cannot have both successors exit and still be in the loop!");

  auto *ParentBB = BI.getParent();
  if (!areLoopExitPHIsLoopInvariant(L, *ParentBB, *LoopExitBB))
    return false;

  DEBUG(dbgs() << "    unswitching trivial branch when: " << CondVal
               << " == " << LoopCond << "\n");

  // Split the preheader, so that we know that there is a safe place to insert
  // the conditional branch. We will change the preheader to have a conditional
  // branch on LoopCond.
  BasicBlock *OldPH = L.getLoopPreheader();
  BasicBlock *NewPH = SplitEdge(OldPH, L.getHeader(), &DT, &LI);

  // Now that we have a place to insert the conditional branch, create a place
  // to branch to: this is the exit block out of the loop that we are
  // unswitching. We need to split this if there are other loop predecessors.
  // Because the loop is in simplified form, *any* other predecessor is enough.
  BasicBlock *UnswitchedBB;
  if (BasicBlock *PredBB = LoopExitBB->getUniquePredecessor()) {
    (void)PredBB;
    assert(PredBB == BI.getParent() &&
           "A branch's parent isn't a predecessor!");
    UnswitchedBB = LoopExitBB;
  } else {
    UnswitchedBB = SplitBlock(LoopExitBB, &LoopExitBB->front(), &DT, &LI);
  }

  // Now splice the branch to gate reaching the new preheader and re-point its
  // successors.
  OldPH->getInstList().splice(std::prev(OldPH->end()),
                              BI.getParent()->getInstList(), BI);
  OldPH->getTerminator()->eraseFromParent();
  BI.setSuccessor(LoopExitSuccIdx, UnswitchedBB);
  BI.setSuccessor(1 - LoopExitSuccIdx, NewPH);

  // Create a new unconditional branch that will continue the loop as a new
  // terminator.
  BranchInst::Create(ContinueBB, ParentBB);

  // Rewrite the relevant PHI nodes.
  if (UnswitchedBB == LoopExitBB)
    rewritePHINodesForUnswitchedExitBlock(*UnswitchedBB, *ParentBB, *OldPH);
  else
    rewritePHINodesForExitAndUnswitchedBlocks(*LoopExitBB, *UnswitchedBB,
                                              *ParentBB, *OldPH);

  // Now we need to update the dominator tree.
  updateDTAfterUnswitch(UnswitchedBB, OldPH, DT);
  // But if we split something off of the loop exit block then we also removed
  // one of the predecessors for the loop exit block and may need to update its
  // idom.
  if (UnswitchedBB != LoopExitBB)
    updateLoopExitIDom(LoopExitBB, L, DT);

  // Since this is an i1 condition we can also trivially replace uses of it
  // within the loop with a constant.
  replaceLoopUsesWithConstant(L, *LoopCond, *Replacement);

  ++NumTrivial;
  ++NumBranches;
  return true;
}

/// Unswitch a trivial switch if the condition is loop invariant.
///
/// This routine should only be called when loop code leading to the switch has
/// been validated as trivial (no side effects). This routine checks if the
/// condition is invariant and that at least one of the successors is a loop
/// exit. This allows us to unswitch without duplicating the loop, making it
/// trivial.
///
/// If this routine fails to unswitch the switch it returns false.
///
/// If the switch can be unswitched, this routine splits the preheader and
/// copies the switch above that split. If the default case is one of the
/// exiting cases, it copies the non-exiting cases and points them at the new
/// preheader. If the default case is not exiting, it copies the exiting cases
/// and points the default at the preheader. It preserves loop simplified form
/// (splitting the exit blocks as necessary). It simplifies the switch within
/// the loop by removing now-dead cases. If the default case is one of those
/// unswitched, it replaces its destination with a new basic block containing
/// only unreachable. Such basic blocks, while technically loop exits, are not
/// considered for unswitching so this is a stable transform and the same
/// switch will not be revisited. If after unswitching there is only a single
/// in-loop successor, the switch is further simplified to an unconditional
/// branch. Still more cleanup can be done with some simplify-cfg like pass.
static bool unswitchTrivialSwitch(Loop &L, SwitchInst &SI, DominatorTree &DT,
                                  LoopInfo &LI) {
  DEBUG(dbgs() << "  Trying to unswitch switch: " << SI << "\n");
  Value *LoopCond = SI.getCondition();

  // If this isn't switching on an invariant condition, we can't unswitch it.
  if (!L.isLoopInvariant(LoopCond))
    return false;

  auto *ParentBB = SI.getParent();

  // FIXME: We should compute this once at the start and update it!
  SmallVector<BasicBlock *, 16> ExitBlocks;
  L.getExitBlocks(ExitBlocks);
  SmallPtrSet<BasicBlock *, 16> ExitBlockSet(ExitBlocks.begin(),
                                             ExitBlocks.end());

  SmallVector<int, 4> ExitCaseIndices;
  for (auto Case : SI.cases()) {
    auto *SuccBB = Case.getCaseSuccessor();
    if (ExitBlockSet.count(SuccBB) &&
        areLoopExitPHIsLoopInvariant(L, *ParentBB, *SuccBB))
      ExitCaseIndices.push_back(Case.getCaseIndex());
  }
  BasicBlock *DefaultExitBB = nullptr;
  if (ExitBlockSet.count(SI.getDefaultDest()) &&
      areLoopExitPHIsLoopInvariant(L, *ParentBB, *SI.getDefaultDest()) &&
      !isa<UnreachableInst>(SI.getDefaultDest()->getTerminator()))
    DefaultExitBB = SI.getDefaultDest();
  else if (ExitCaseIndices.empty())
    return false;

  DEBUG(dbgs() << "    unswitching trivial cases...\n");

  SmallVector<std::pair<ConstantInt *, BasicBlock *>, 4> ExitCases;
  ExitCases.reserve(ExitCaseIndices.size());
  // We walk the case indices backwards so that we remove the last case first
  // and don't disrupt the earlier indices.
  for (unsigned Index : reverse(ExitCaseIndices)) {
    auto CaseI = SI.case_begin() + Index;
    // Save the value of this case.
    ExitCases.push_back({CaseI->getCaseValue(), CaseI->getCaseSuccessor()});
    // Delete the unswitched cases.
    SI.removeCase(CaseI);
  }

  // Check if after this all of the remaining cases point at the same
  // successor.
  BasicBlock *CommonSuccBB = nullptr;
  if (SI.getNumCases() > 0 &&
      std::all_of(std::next(SI.case_begin()), SI.case_end(),
                  [&SI](const SwitchInst::CaseHandle &Case) {
                    return Case.getCaseSuccessor() ==
                           SI.case_begin()->getCaseSuccessor();
                  }))
    CommonSuccBB = SI.case_begin()->getCaseSuccessor();

  if (DefaultExitBB) {
    // We can't remove the default edge so replace it with an edge to either
    // the single common remaining successor (if we have one) or an unreachable
    // block.
    if (CommonSuccBB) {
      SI.setDefaultDest(CommonSuccBB);
    } else {
      BasicBlock *UnreachableBB = BasicBlock::Create(
          ParentBB->getContext(),
          Twine(ParentBB->getName()) + ".unreachable_default",
          ParentBB->getParent());
      new UnreachableInst(ParentBB->getContext(), UnreachableBB);
      SI.setDefaultDest(UnreachableBB);
      DT.addNewBlock(UnreachableBB, ParentBB);
    }
  } else {
    // If we're not unswitching the default, we need it to match any cases to
    // have a common successor or if we have no cases it is the common
    // successor.
    if (SI.getNumCases() == 0)
      CommonSuccBB = SI.getDefaultDest();
    else if (SI.getDefaultDest() != CommonSuccBB)
      CommonSuccBB = nullptr;
  }

  // Split the preheader, so that we know that there is a safe place to insert
  // the switch.
  BasicBlock *OldPH = L.getLoopPreheader();
  BasicBlock *NewPH = SplitEdge(OldPH, L.getHeader(), &DT, &LI);
  OldPH->getTerminator()->eraseFromParent();

  // Now add the unswitched switch.
  auto *NewSI = SwitchInst::Create(LoopCond, NewPH, ExitCases.size(), OldPH);

  // Rewrite the IR for the unswitched basic blocks. This requires two steps.
  // First, we split any exit blocks with remaining in-loop predecessors. Then
  // we update the PHIs in one of two ways depending on if there was a split.
  // We walk in reverse so that we split in the same order as the cases
  // appeared. This is purely for convenience of reading the resulting IR, but
  // it doesn't cost anything really.
  SmallPtrSet<BasicBlock *, 2> UnswitchedExitBBs;
  SmallDenseMap<BasicBlock *, BasicBlock *, 2> SplitExitBBMap;
  // Handle the default exit if necessary.
  // FIXME: It'd be great if we could merge this with the loop below but LLVM's
  // ranges aren't quite powerful enough yet.
  if (DefaultExitBB) {
    if (pred_empty(DefaultExitBB)) {
      UnswitchedExitBBs.insert(DefaultExitBB);
      rewritePHINodesForUnswitchedExitBlock(*DefaultExitBB, *ParentBB, *OldPH);
    } else {
      auto *SplitBB =
          SplitBlock(DefaultExitBB, &DefaultExitBB->front(), &DT, &LI);
      rewritePHINodesForExitAndUnswitchedBlocks(*DefaultExitBB, *SplitBB,
                                                *ParentBB, *OldPH);
      updateLoopExitIDom(DefaultExitBB, L, DT);
      DefaultExitBB = SplitExitBBMap[DefaultExitBB] = SplitBB;
    }
  }
  // Note that we must use a reference in the for loop so that we update the
  // container.
  for (auto &CasePair : reverse(ExitCases)) {
    // Grab a reference to the exit block in the pair so that we can update it.
    BasicBlock *ExitBB = CasePair.second;

    // If this case is the last edge into the exit block, we can simply reuse it
    // as it will no longer be a loop exit. No mapping necessary.
    if (pred_empty(ExitBB)) {
      // Only rewrite once.
      if (UnswitchedExitBBs.insert(ExitBB).second)
        rewritePHINodesForUnswitchedExitBlock(*ExitBB, *ParentBB, *OldPH);
      continue;
    }

    // Otherwise we need to split the exit block so that we retain an exit
    // block from the loop and a target for the unswitched condition.
    BasicBlock *&SplitExitBB = SplitExitBBMap[ExitBB];
    if (!SplitExitBB) {
      // If this is the first time we see this, do the split and remember it.
      SplitExitBB = SplitBlock(ExitBB, &ExitBB->front(), &DT, &LI);
      rewritePHINodesForExitAndUnswitchedBlocks(*ExitBB, *SplitExitBB,
                                                *ParentBB, *OldPH);
      updateLoopExitIDom(ExitBB, L, DT);
    }
    // Update the case pair to point to the split block.
    CasePair.second = SplitExitBB;
  }

  // Now add the unswitched cases. We do this in reverse order as we built them
  // in reverse order.
  for (auto CasePair : reverse(ExitCases)) {
    ConstantInt *CaseVal = CasePair.first;
    BasicBlock *UnswitchedBB = CasePair.second;

    NewSI->addCase(CaseVal, UnswitchedBB);
    updateDTAfterUnswitch(UnswitchedBB, OldPH, DT);
  }

  // If the default was unswitched, re-point it and add explicit cases for
  // entering the loop.
  if (DefaultExitBB) {
    NewSI->setDefaultDest(DefaultExitBB);
    updateDTAfterUnswitch(DefaultExitBB, OldPH, DT);

    // We removed all the exit cases, so we just copy the cases to the
    // unswitched switch.
    for (auto Case : SI.cases())
      NewSI->addCase(Case.getCaseValue(), NewPH);
  }

  // If we ended up with a common successor for every path through the switch
  // after unswitching, rewrite it to an unconditional branch to make it easy
  // to recognize. Otherwise we potentially have to recognize the default case
  // pointing at unreachable and other complexity.
  if (CommonSuccBB) {
    BasicBlock *BB = SI.getParent();
    SI.eraseFromParent();
    BranchInst::Create(CommonSuccBB, BB);
  }

  DT.verifyDomTree();
  ++NumTrivial;
  ++NumSwitches;
  return true;
}

/// This routine scans the loop to find a branch or switch which occurs before
/// any side effects occur. These can potentially be unswitched without
/// duplicating the loop. If a branch or switch is successfully unswitched the
/// scanning continues to see if subsequent branches or switches have become
/// trivial. Once all trivial candidates have been unswitched, this routine
/// returns.
///
/// The return value indicates whether anything was unswitched (and therefore
/// changed).
static bool unswitchAllTrivialConditions(Loop &L, DominatorTree &DT,
                                         LoopInfo &LI) {
  bool Changed = false;

  // If loop header has only one reachable successor we should keep looking for
  // trivial condition candidates in the successor as well. An alternative is
  // to constant fold conditions and merge successors into loop header (then we
  // only need to check header's terminator). The reason for not doing this in
  // LoopUnswitch pass is that it could potentially break LoopPassManager's
  // invariants. Folding dead branches could either eliminate the current loop
  // or make other loops unreachable. LCSSA form might also not be preserved
  // after deleting branches. The following code keeps traversing loop header's
  // successors until it finds the trivial condition candidate (condition that
  // is not a constant). Since unswitching generates branches with constant
  // conditions, this scenario could be very common in practice.
  BasicBlock *CurrentBB = L.getHeader();
  SmallPtrSet<BasicBlock *, 8> Visited;
  Visited.insert(CurrentBB);
  do {
    // Check if there are any side-effecting instructions (e.g. stores, calls,
    // volatile loads) in the part of the loop that the code *would* execute
    // without unswitching.
    if (llvm::any_of(*CurrentBB,
                     [](Instruction &I) { return I.mayHaveSideEffects(); }))
      return Changed;

    TerminatorInst *CurrentTerm = CurrentBB->getTerminator();

    if (auto *SI = dyn_cast<SwitchInst>(CurrentTerm)) {
      // Don't bother trying to unswitch past a switch with a constant
      // condition. This should be removed prior to running this pass by
      // simplify-cfg.
      if (isa<Constant>(SI->getCondition()))
        return Changed;

      if (!unswitchTrivialSwitch(L, *SI, DT, LI))
        // Coludn't unswitch this one so we're done.
        return Changed;

      // Mark that we managed to unswitch something.
      Changed = true;

      // If unswitching turned the terminator into an unconditional branch then
      // we can continue. The unswitching logic specifically works to fold any
      // cases it can into an unconditional branch to make it easier to
      // recognize here.
      auto *BI = dyn_cast<BranchInst>(CurrentBB->getTerminator());
      if (!BI || BI->isConditional())
        return Changed;

      CurrentBB = BI->getSuccessor(0);
      continue;
    }

    auto *BI = dyn_cast<BranchInst>(CurrentTerm);
    if (!BI)
      // We do not understand other terminator instructions.
      return Changed;

    // Don't bother trying to unswitch past an unconditional branch or a branch
    // with a constant value. These should be removed by simplify-cfg prior to
    // running this pass.
    if (!BI->isConditional() || isa<Constant>(BI->getCondition()))
      return Changed;

    // Found a trivial condition candidate: non-foldable conditional branch. If
    // we fail to unswitch this, we can't do anything else that is trivial.
    if (!unswitchTrivialBranch(L, *BI, DT, LI))
      return Changed;

    // Mark that we managed to unswitch something.
    Changed = true;

    // We unswitched the branch. This should always leave us with an
    // unconditional branch that we can follow now.
    BI = cast<BranchInst>(CurrentBB->getTerminator());
    assert(!BI->isConditional() &&
           "Cannot form a conditional branch by unswitching1");
    CurrentBB = BI->getSuccessor(0);

    // When continuing, if we exit the loop or reach a previous visited block,
    // then we can not reach any trivial condition candidates (unfoldable
    // branch instructions or switch instructions) and no unswitch can happen.
  } while (L.contains(CurrentBB) && Visited.insert(CurrentBB).second);

  return Changed;
}

/// Unswitch control flow predicated on loop invariant conditions.
///
/// This first hoists all branches or switches which are trivial (IE, do not
/// require duplicating any part of the loop) out of the loop body. It then
/// looks at other loop invariant control flows and tries to unswitch those as
/// well by cloning the loop if the result is small enough.
static bool unswitchLoop(Loop &L, DominatorTree &DT, LoopInfo &LI,
                         AssumptionCache &AC) {
  assert(L.isLCSSAForm(DT) &&
         "Loops must be in LCSSA form before unswitching.");
  bool Changed = false;

  // Must be in loop simplified form: we need a preheader and dedicated exits.
  if (!L.isLoopSimplifyForm())
    return false;

  // Try trivial unswitch first before loop over other basic blocks in the loop.
  Changed |= unswitchAllTrivialConditions(L, DT, LI);

  // FIXME: Add support for non-trivial unswitching by cloning the loop.

  return Changed;
}

PreservedAnalyses SimpleLoopUnswitchPass::run(Loop &L, LoopAnalysisManager &AM,
                                              LoopStandardAnalysisResults &AR,
                                              LPMUpdater &U) {
  Function &F = *L.getHeader()->getParent();
  (void)F;

  DEBUG(dbgs() << "Unswitching loop in " << F.getName() << ": " << L << "\n");

  if (!unswitchLoop(L, AR.DT, AR.LI, AR.AC))
    return PreservedAnalyses::all();

#ifndef NDEBUG
  // Historically this pass has had issues with the dominator tree so verify it
  // in asserts builds.
  AR.DT.verifyDomTree();
#endif
  return getLoopPassPreservedAnalyses();
}

namespace {

class SimpleLoopUnswitchLegacyPass : public LoopPass {
public:
  static char ID; // Pass ID, replacement for typeid

  explicit SimpleLoopUnswitchLegacyPass() : LoopPass(ID) {
    initializeSimpleLoopUnswitchLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnLoop(Loop *L, LPPassManager &LPM) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    getLoopAnalysisUsage(AU);
  }
};

} // end anonymous namespace

bool SimpleLoopUnswitchLegacyPass::runOnLoop(Loop *L, LPPassManager &LPM) {
  if (skipLoop(L))
    return false;

  Function &F = *L->getHeader()->getParent();

  DEBUG(dbgs() << "Unswitching loop in " << F.getName() << ": " << *L << "\n");

  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  auto &AC = getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);

  bool Changed = unswitchLoop(*L, DT, LI, AC);

#ifndef NDEBUG
  // Historically this pass has had issues with the dominator tree so verify it
  // in asserts builds.
  DT.verifyDomTree();
#endif
  return Changed;
}

char SimpleLoopUnswitchLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(SimpleLoopUnswitchLegacyPass, "simple-loop-unswitch",
                      "Simple unswitch loops", false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(LoopPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(SimpleLoopUnswitchLegacyPass, "simple-loop-unswitch",
                    "Simple unswitch loops", false, false)

Pass *llvm::createSimpleLoopUnswitchLegacyPass() {
  return new SimpleLoopUnswitchLegacyPass();
}
