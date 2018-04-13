//===- SimpleLoopUnswitch.cpp - Hoist loop-invariant control flow ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/SimpleLoopUnswitch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CodeMetrics.h"
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
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/GenericDomTree.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar/SimpleLoopUnswitch.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <algorithm>
#include <cassert>
#include <iterator>
#include <numeric>
#include <utility>

#define DEBUG_TYPE "simple-loop-unswitch"

using namespace llvm;

STATISTIC(NumBranches, "Number of branches unswitched");
STATISTIC(NumSwitches, "Number of switches unswitched");
STATISTIC(NumTrivial, "Number of unswitches that are trivial");

static cl::opt<bool> EnableNonTrivialUnswitch(
    "enable-nontrivial-unswitch", cl::init(false), cl::Hidden,
    cl::desc("Forcibly enables non-trivial loop unswitching rather than "
             "following the configuration passed into the pass."));

static cl::opt<int>
    UnswitchThreshold("unswitch-threshold", cl::init(50), cl::Hidden,
                      cl::desc("The cost threshold for unswitching a loop."));

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

/// Update the IDom for a basic block whose predecessor set has changed.
///
/// This routine is designed to work when the domtree update is relatively
/// localized by leveraging a known common dominator, often a loop header.
///
/// FIXME: Should consider hand-rolling a slightly more efficient non-DFS
/// approach here as we can do that easily by persisting the candidate IDom's
/// dominating set between each predecessor.
///
/// FIXME: Longer term, many uses of this can be replaced by an incremental
/// domtree update strategy that starts from a known dominating block and
/// rebuilds that subtree.
static bool updateIDomWithKnownCommonDominator(BasicBlock *BB,
                                               BasicBlock *KnownDominatingBB,
                                               DominatorTree &DT) {
  assert(pred_begin(BB) != pred_end(BB) &&
         "This routine does not handle unreachable blocks!");

  BasicBlock *OrigIDom = DT[BB]->getIDom()->getBlock();

  BasicBlock *IDom = *pred_begin(BB);
  assert(DT.dominates(KnownDominatingBB, IDom) &&
         "Bad known dominating block!");

  // Walk all of the other predecessors finding the nearest common dominator
  // until all predecessors are covered or we reach the loop header. The loop
  // header necessarily dominates all loop exit blocks in loop simplified form
  // so we can early-exit the moment we hit that block.
  for (auto PI = std::next(pred_begin(BB)), PE = pred_end(BB);
       PI != PE && IDom != KnownDominatingBB; ++PI) {
    assert(DT.dominates(KnownDominatingBB, *PI) &&
           "Bad known dominating block!");
    IDom = DT.findNearestCommonDominator(IDom, *PI);
  }

  if (IDom == OrigIDom)
    return false;

  DT.changeImmediateDominator(BB, IDom);
  return true;
}

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
static void appendDomFrontier(DomTreeNode *Node,
                              SmallSetVector<BasicBlock *, 4> &Worklist,
                              SmallVectorImpl<DomTreeNode *> &DomNodes,
                              SmallPtrSetImpl<BasicBlock *> &DomSet) {
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
  SmallSetVector<BasicBlock *, 4> Worklist;

  // Scratch data structures reused by domfrontier finding.
  SmallVector<DomTreeNode *, 4> DomNodes;
  SmallPtrSet<BasicBlock *, 4> DomSet;

  // Append the initial dom frontier nodes.
  appendDomFrontier(UnswitchedNode, Worklist, DomNodes, DomSet);

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
    appendDomFrontier(Node, Worklist, DomNodes, DomSet);
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
  for (PHINode &PN : UnswitchedBB.phis()) {
    // When the loop exit is directly unswitched we just need to update the
    // incoming basic block. We loop to handle weird cases with repeated
    // incoming blocks, but expect to typically only have one operand here.
    for (auto i : seq<int>(0, PN.getNumOperands())) {
      assert(PN.getIncomingBlock(i) == &OldExitingBB &&
             "Found incoming block different from unique predecessor!");
      PN.setIncomingBlock(i, &OldPH);
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
  for (PHINode &PN : ExitBB.phis()) {
    auto *NewPN = PHINode::Create(PN.getType(), /*NumReservedValues*/ 2,
                                  PN.getName() + ".split", InsertPt);

    // Walk backwards over the old PHI node's inputs to minimize the cost of
    // removing each one. We have to do this weird loop manually so that we
    // create the same number of new incoming edges in the new PHI as we expect
    // each case-based edge to be included in the unswitched switch in some
    // cases.
    // FIXME: This is really, really gross. It would be much cleaner if LLVM
    // allowed us to create a single entry for a predecessor block without
    // having separate entries for each "edge" even though these edges are
    // required to produce identical results.
    for (int i = PN.getNumIncomingValues() - 1; i >= 0; --i) {
      if (PN.getIncomingBlock(i) != &OldExitingBB)
        continue;

      Value *Incoming = PN.removeIncomingValue(i);
      NewPN->addIncoming(Incoming, &OldPH);
    }

    // Now replace the old PHI with the new one and wire the old one in as an
    // input to the new one.
    PN.replaceAllUsesWith(NewPN);
    NewPN->addIncoming(&PN, &ExitBB);
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
    updateIDomWithKnownCommonDominator(LoopExitBB, L.getHeader(), DT);

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
      updateIDomWithKnownCommonDominator(DefaultExitBB, L.getHeader(), DT);
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
      updateIDomWithKnownCommonDominator(ExitBB, L.getHeader(), DT);
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

  assert(DT.verify(DominatorTree::VerificationLevel::Fast));
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

/// Build the cloned blocks for an unswitched copy of the given loop.
///
/// The cloned blocks are inserted before the loop preheader (`LoopPH`) and
/// after the split block (`SplitBB`) that will be used to select between the
/// cloned and original loop.
///
/// This routine handles cloning all of the necessary loop blocks and exit
/// blocks including rewriting their instructions and the relevant PHI nodes.
/// It skips loop and exit blocks that are not necessary based on the provided
/// set. It also correctly creates the unconditional branch in the cloned
/// unswitched parent block to only point at the unswitched successor.
///
/// This does not handle most of the necessary updates to `LoopInfo`. Only exit
/// block splitting is correctly reflected in `LoopInfo`, essentially all of
/// the cloned blocks (and their loops) are left without full `LoopInfo`
/// updates. This also doesn't fully update `DominatorTree`. It adds the cloned
/// blocks to them but doesn't create the cloned `DominatorTree` structure and
/// instead the caller must recompute an accurate DT. It *does* correctly
/// update the `AssumptionCache` provided in `AC`.
static BasicBlock *buildClonedLoopBlocks(
    Loop &L, BasicBlock *LoopPH, BasicBlock *SplitBB,
    ArrayRef<BasicBlock *> ExitBlocks, BasicBlock *ParentBB,
    BasicBlock *UnswitchedSuccBB, BasicBlock *ContinueSuccBB,
    const SmallPtrSetImpl<BasicBlock *> &SkippedLoopAndExitBlocks,
    ValueToValueMapTy &VMap, AssumptionCache &AC, DominatorTree &DT,
    LoopInfo &LI) {
  SmallVector<BasicBlock *, 4> NewBlocks;
  NewBlocks.reserve(L.getNumBlocks() + ExitBlocks.size());

  // We will need to clone a bunch of blocks, wrap up the clone operation in
  // a helper.
  auto CloneBlock = [&](BasicBlock *OldBB) {
    // Clone the basic block and insert it before the new preheader.
    BasicBlock *NewBB = CloneBasicBlock(OldBB, VMap, ".us", OldBB->getParent());
    NewBB->moveBefore(LoopPH);

    // Record this block and the mapping.
    NewBlocks.push_back(NewBB);
    VMap[OldBB] = NewBB;

    // Add the block to the domtree. We'll move it to the correct position
    // below.
    DT.addNewBlock(NewBB, SplitBB);

    return NewBB;
  };

  // First, clone the preheader.
  auto *ClonedPH = CloneBlock(LoopPH);

  // Then clone all the loop blocks, skipping the ones that aren't necessary.
  for (auto *LoopBB : L.blocks())
    if (!SkippedLoopAndExitBlocks.count(LoopBB))
      CloneBlock(LoopBB);

  // Split all the loop exit edges so that when we clone the exit blocks, if
  // any of the exit blocks are *also* a preheader for some other loop, we
  // don't create multiple predecessors entering the loop header.
  for (auto *ExitBB : ExitBlocks) {
    if (SkippedLoopAndExitBlocks.count(ExitBB))
      continue;

    // When we are going to clone an exit, we don't need to clone all the
    // instructions in the exit block and we want to ensure we have an easy
    // place to merge the CFG, so split the exit first. This is always safe to
    // do because there cannot be any non-loop predecessors of a loop exit in
    // loop simplified form.
    auto *MergeBB = SplitBlock(ExitBB, &ExitBB->front(), &DT, &LI);

    // Rearrange the names to make it easier to write test cases by having the
    // exit block carry the suffix rather than the merge block carrying the
    // suffix.
    MergeBB->takeName(ExitBB);
    ExitBB->setName(Twine(MergeBB->getName()) + ".split");

    // Now clone the original exit block.
    auto *ClonedExitBB = CloneBlock(ExitBB);
    assert(ClonedExitBB->getTerminator()->getNumSuccessors() == 1 &&
           "Exit block should have been split to have one successor!");
    assert(ClonedExitBB->getTerminator()->getSuccessor(0) == MergeBB &&
           "Cloned exit block has the wrong successor!");

    // Move the merge block's idom to be the split point as one exit is
    // dominated by one header, and the other by another, so we know the split
    // point dominates both. While the dominator tree isn't fully accurate, we
    // want sub-trees within the original loop to be correctly reflect
    // dominance within that original loop (at least) and that requires moving
    // the merge block out of that subtree.
    // FIXME: This is very brittle as we essentially have a partial contract on
    // the dominator tree. We really need to instead update it and keep it
    // valid or stop relying on it.
    DT.changeImmediateDominator(MergeBB, SplitBB);

    // Remap any cloned instructions and create a merge phi node for them.
    for (auto ZippedInsts : llvm::zip_first(
             llvm::make_range(ExitBB->begin(), std::prev(ExitBB->end())),
             llvm::make_range(ClonedExitBB->begin(),
                              std::prev(ClonedExitBB->end())))) {
      Instruction &I = std::get<0>(ZippedInsts);
      Instruction &ClonedI = std::get<1>(ZippedInsts);

      // The only instructions in the exit block should be PHI nodes and
      // potentially a landing pad.
      assert(
          (isa<PHINode>(I) || isa<LandingPadInst>(I) || isa<CatchPadInst>(I)) &&
          "Bad instruction in exit block!");
      // We should have a value map between the instruction and its clone.
      assert(VMap.lookup(&I) == &ClonedI && "Mismatch in the value map!");

      auto *MergePN =
          PHINode::Create(I.getType(), /*NumReservedValues*/ 2, ".us-phi",
                          &*MergeBB->getFirstInsertionPt());
      I.replaceAllUsesWith(MergePN);
      MergePN->addIncoming(&I, ExitBB);
      MergePN->addIncoming(&ClonedI, ClonedExitBB);
    }
  }

  // Rewrite the instructions in the cloned blocks to refer to the instructions
  // in the cloned blocks. We have to do this as a second pass so that we have
  // everything available. Also, we have inserted new instructions which may
  // include assume intrinsics, so we update the assumption cache while
  // processing this.
  for (auto *ClonedBB : NewBlocks)
    for (Instruction &I : *ClonedBB) {
      RemapInstruction(&I, VMap,
                       RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);
      if (auto *II = dyn_cast<IntrinsicInst>(&I))
        if (II->getIntrinsicID() == Intrinsic::assume)
          AC.registerAssumption(II);
    }

  // Remove the cloned parent as a predecessor of the cloned continue successor
  // if we did in fact clone it.
  auto *ClonedParentBB = cast<BasicBlock>(VMap.lookup(ParentBB));
  if (auto *ClonedContinueSuccBB =
          cast_or_null<BasicBlock>(VMap.lookup(ContinueSuccBB)))
    ClonedContinueSuccBB->removePredecessor(ClonedParentBB,
                                            /*DontDeleteUselessPHIs*/ true);
  // Replace the cloned branch with an unconditional branch to the cloneed
  // unswitched successor.
  auto *ClonedSuccBB = cast<BasicBlock>(VMap.lookup(UnswitchedSuccBB));
  ClonedParentBB->getTerminator()->eraseFromParent();
  BranchInst::Create(ClonedSuccBB, ClonedParentBB);

  // Update any PHI nodes in the cloned successors of the skipped blocks to not
  // have spurious incoming values.
  for (auto *LoopBB : L.blocks())
    if (SkippedLoopAndExitBlocks.count(LoopBB))
      for (auto *SuccBB : successors(LoopBB))
        if (auto *ClonedSuccBB = cast_or_null<BasicBlock>(VMap.lookup(SuccBB)))
          for (PHINode &PN : ClonedSuccBB->phis())
            PN.removeIncomingValue(LoopBB, /*DeletePHIIfEmpty*/ false);

  return ClonedPH;
}

/// Recursively clone the specified loop and all of its children.
///
/// The target parent loop for the clone should be provided, or can be null if
/// the clone is a top-level loop. While cloning, all the blocks are mapped
/// with the provided value map. The entire original loop must be present in
/// the value map. The cloned loop is returned.
static Loop *cloneLoopNest(Loop &OrigRootL, Loop *RootParentL,
                           const ValueToValueMapTy &VMap, LoopInfo &LI) {
  auto AddClonedBlocksToLoop = [&](Loop &OrigL, Loop &ClonedL) {
    assert(ClonedL.getBlocks().empty() && "Must start with an empty loop!");
    ClonedL.reserveBlocks(OrigL.getNumBlocks());
    for (auto *BB : OrigL.blocks()) {
      auto *ClonedBB = cast<BasicBlock>(VMap.lookup(BB));
      ClonedL.addBlockEntry(ClonedBB);
      if (LI.getLoopFor(BB) == &OrigL) {
        assert(!LI.getLoopFor(ClonedBB) &&
               "Should not have an existing loop for this block!");
        LI.changeLoopFor(ClonedBB, &ClonedL);
      }
    }
  };

  // We specially handle the first loop because it may get cloned into
  // a different parent and because we most commonly are cloning leaf loops.
  Loop *ClonedRootL = LI.AllocateLoop();
  if (RootParentL)
    RootParentL->addChildLoop(ClonedRootL);
  else
    LI.addTopLevelLoop(ClonedRootL);
  AddClonedBlocksToLoop(OrigRootL, *ClonedRootL);

  if (OrigRootL.empty())
    return ClonedRootL;

  // If we have a nest, we can quickly clone the entire loop nest using an
  // iterative approach because it is a tree. We keep the cloned parent in the
  // data structure to avoid repeatedly querying through a map to find it.
  SmallVector<std::pair<Loop *, Loop *>, 16> LoopsToClone;
  // Build up the loops to clone in reverse order as we'll clone them from the
  // back.
  for (Loop *ChildL : llvm::reverse(OrigRootL))
    LoopsToClone.push_back({ClonedRootL, ChildL});
  do {
    Loop *ClonedParentL, *L;
    std::tie(ClonedParentL, L) = LoopsToClone.pop_back_val();
    Loop *ClonedL = LI.AllocateLoop();
    ClonedParentL->addChildLoop(ClonedL);
    AddClonedBlocksToLoop(*L, *ClonedL);
    for (Loop *ChildL : llvm::reverse(*L))
      LoopsToClone.push_back({ClonedL, ChildL});
  } while (!LoopsToClone.empty());

  return ClonedRootL;
}

/// Build the cloned loops of an original loop from unswitching.
///
/// Because unswitching simplifies the CFG of the loop, this isn't a trivial
/// operation. We need to re-verify that there even is a loop (as the backedge
/// may not have been cloned), and even if there are remaining backedges the
/// backedge set may be different. However, we know that each child loop is
/// undisturbed, we only need to find where to place each child loop within
/// either any parent loop or within a cloned version of the original loop.
///
/// Because child loops may end up cloned outside of any cloned version of the
/// original loop, multiple cloned sibling loops may be created. All of them
/// are returned so that the newly introduced loop nest roots can be
/// identified.
static Loop *buildClonedLoops(Loop &OrigL, ArrayRef<BasicBlock *> ExitBlocks,
                              const ValueToValueMapTy &VMap, LoopInfo &LI,
                              SmallVectorImpl<Loop *> &NonChildClonedLoops) {
  Loop *ClonedL = nullptr;

  auto *OrigPH = OrigL.getLoopPreheader();
  auto *OrigHeader = OrigL.getHeader();

  auto *ClonedPH = cast<BasicBlock>(VMap.lookup(OrigPH));
  auto *ClonedHeader = cast<BasicBlock>(VMap.lookup(OrigHeader));

  // We need to know the loops of the cloned exit blocks to even compute the
  // accurate parent loop. If we only clone exits to some parent of the
  // original parent, we want to clone into that outer loop. We also keep track
  // of the loops that our cloned exit blocks participate in.
  Loop *ParentL = nullptr;
  SmallVector<BasicBlock *, 4> ClonedExitsInLoops;
  SmallDenseMap<BasicBlock *, Loop *, 16> ExitLoopMap;
  ClonedExitsInLoops.reserve(ExitBlocks.size());
  for (auto *ExitBB : ExitBlocks)
    if (auto *ClonedExitBB = cast_or_null<BasicBlock>(VMap.lookup(ExitBB)))
      if (Loop *ExitL = LI.getLoopFor(ExitBB)) {
        ExitLoopMap[ClonedExitBB] = ExitL;
        ClonedExitsInLoops.push_back(ClonedExitBB);
        if (!ParentL || (ParentL != ExitL && ParentL->contains(ExitL)))
          ParentL = ExitL;
      }
  assert((!ParentL || ParentL == OrigL.getParentLoop() ||
          ParentL->contains(OrigL.getParentLoop())) &&
         "The computed parent loop should always contain (or be) the parent of "
         "the original loop.");

  // We build the set of blocks dominated by the cloned header from the set of
  // cloned blocks out of the original loop. While not all of these will
  // necessarily be in the cloned loop, it is enough to establish that they
  // aren't in unreachable cycles, etc.
  SmallSetVector<BasicBlock *, 16> ClonedLoopBlocks;
  for (auto *BB : OrigL.blocks())
    if (auto *ClonedBB = cast_or_null<BasicBlock>(VMap.lookup(BB)))
      ClonedLoopBlocks.insert(ClonedBB);

  // Rebuild the set of blocks that will end up in the cloned loop. We may have
  // skipped cloning some region of this loop which can in turn skip some of
  // the backedges so we have to rebuild the blocks in the loop based on the
  // backedges that remain after cloning.
  SmallVector<BasicBlock *, 16> Worklist;
  SmallPtrSet<BasicBlock *, 16> BlocksInClonedLoop;
  for (auto *Pred : predecessors(ClonedHeader)) {
    // The only possible non-loop header predecessor is the preheader because
    // we know we cloned the loop in simplified form.
    if (Pred == ClonedPH)
      continue;

    // Because the loop was in simplified form, the only non-loop predecessor
    // should be the preheader.
    assert(ClonedLoopBlocks.count(Pred) && "Found a predecessor of the loop "
                                           "header other than the preheader "
                                           "that is not part of the loop!");

    // Insert this block into the loop set and on the first visit (and if it
    // isn't the header we're currently walking) put it into the worklist to
    // recurse through.
    if (BlocksInClonedLoop.insert(Pred).second && Pred != ClonedHeader)
      Worklist.push_back(Pred);
  }

  // If we had any backedges then there *is* a cloned loop. Put the header into
  // the loop set and then walk the worklist backwards to find all the blocks
  // that remain within the loop after cloning.
  if (!BlocksInClonedLoop.empty()) {
    BlocksInClonedLoop.insert(ClonedHeader);

    while (!Worklist.empty()) {
      BasicBlock *BB = Worklist.pop_back_val();
      assert(BlocksInClonedLoop.count(BB) &&
             "Didn't put block into the loop set!");

      // Insert any predecessors that are in the possible set into the cloned
      // set, and if the insert is successful, add them to the worklist. Note
      // that we filter on the blocks that are definitely reachable via the
      // backedge to the loop header so we may prune out dead code within the
      // cloned loop.
      for (auto *Pred : predecessors(BB))
        if (ClonedLoopBlocks.count(Pred) &&
            BlocksInClonedLoop.insert(Pred).second)
          Worklist.push_back(Pred);
    }

    ClonedL = LI.AllocateLoop();
    if (ParentL) {
      ParentL->addBasicBlockToLoop(ClonedPH, LI);
      ParentL->addChildLoop(ClonedL);
    } else {
      LI.addTopLevelLoop(ClonedL);
    }

    ClonedL->reserveBlocks(BlocksInClonedLoop.size());
    // We don't want to just add the cloned loop blocks based on how we
    // discovered them. The original order of blocks was carefully built in
    // a way that doesn't rely on predecessor ordering. Rather than re-invent
    // that logic, we just re-walk the original blocks (and those of the child
    // loops) and filter them as we add them into the cloned loop.
    for (auto *BB : OrigL.blocks()) {
      auto *ClonedBB = cast_or_null<BasicBlock>(VMap.lookup(BB));
      if (!ClonedBB || !BlocksInClonedLoop.count(ClonedBB))
        continue;

      // Directly add the blocks that are only in this loop.
      if (LI.getLoopFor(BB) == &OrigL) {
        ClonedL->addBasicBlockToLoop(ClonedBB, LI);
        continue;
      }

      // We want to manually add it to this loop and parents.
      // Registering it with LoopInfo will happen when we clone the top
      // loop for this block.
      for (Loop *PL = ClonedL; PL; PL = PL->getParentLoop())
        PL->addBlockEntry(ClonedBB);
    }

    // Now add each child loop whose header remains within the cloned loop. All
    // of the blocks within the loop must satisfy the same constraints as the
    // header so once we pass the header checks we can just clone the entire
    // child loop nest.
    for (Loop *ChildL : OrigL) {
      auto *ClonedChildHeader =
          cast_or_null<BasicBlock>(VMap.lookup(ChildL->getHeader()));
      if (!ClonedChildHeader || !BlocksInClonedLoop.count(ClonedChildHeader))
        continue;

#ifndef NDEBUG
      // We should never have a cloned child loop header but fail to have
      // all of the blocks for that child loop.
      for (auto *ChildLoopBB : ChildL->blocks())
        assert(BlocksInClonedLoop.count(
                   cast<BasicBlock>(VMap.lookup(ChildLoopBB))) &&
               "Child cloned loop has a header within the cloned outer "
               "loop but not all of its blocks!");
#endif

      cloneLoopNest(*ChildL, ClonedL, VMap, LI);
    }
  }

  // Now that we've handled all the components of the original loop that were
  // cloned into a new loop, we still need to handle anything from the original
  // loop that wasn't in a cloned loop.

  // Figure out what blocks are left to place within any loop nest containing
  // the unswitched loop. If we never formed a loop, the cloned PH is one of
  // them.
  SmallPtrSet<BasicBlock *, 16> UnloopedBlockSet;
  if (BlocksInClonedLoop.empty())
    UnloopedBlockSet.insert(ClonedPH);
  for (auto *ClonedBB : ClonedLoopBlocks)
    if (!BlocksInClonedLoop.count(ClonedBB))
      UnloopedBlockSet.insert(ClonedBB);

  // Copy the cloned exits and sort them in ascending loop depth, we'll work
  // backwards across these to process them inside out. The order shouldn't
  // matter as we're just trying to build up the map from inside-out; we use
  // the map in a more stably ordered way below.
  auto OrderedClonedExitsInLoops = ClonedExitsInLoops;
  llvm::sort(OrderedClonedExitsInLoops.begin(),
             OrderedClonedExitsInLoops.end(),
             [&](BasicBlock *LHS, BasicBlock *RHS) {
               return ExitLoopMap.lookup(LHS)->getLoopDepth() <
                      ExitLoopMap.lookup(RHS)->getLoopDepth();
             });

  // Populate the existing ExitLoopMap with everything reachable from each
  // exit, starting from the inner most exit.
  while (!UnloopedBlockSet.empty() && !OrderedClonedExitsInLoops.empty()) {
    assert(Worklist.empty() && "Didn't clear worklist!");

    BasicBlock *ExitBB = OrderedClonedExitsInLoops.pop_back_val();
    Loop *ExitL = ExitLoopMap.lookup(ExitBB);

    // Walk the CFG back until we hit the cloned PH adding everything reachable
    // and in the unlooped set to this exit block's loop.
    Worklist.push_back(ExitBB);
    do {
      BasicBlock *BB = Worklist.pop_back_val();
      // We can stop recursing at the cloned preheader (if we get there).
      if (BB == ClonedPH)
        continue;

      for (BasicBlock *PredBB : predecessors(BB)) {
        // If this pred has already been moved to our set or is part of some
        // (inner) loop, no update needed.
        if (!UnloopedBlockSet.erase(PredBB)) {
          assert(
              (BlocksInClonedLoop.count(PredBB) || ExitLoopMap.count(PredBB)) &&
              "Predecessor not mapped to a loop!");
          continue;
        }

        // We just insert into the loop set here. We'll add these blocks to the
        // exit loop after we build up the set in an order that doesn't rely on
        // predecessor order (which in turn relies on use list order).
        bool Inserted = ExitLoopMap.insert({PredBB, ExitL}).second;
        (void)Inserted;
        assert(Inserted && "Should only visit an unlooped block once!");

        // And recurse through to its predecessors.
        Worklist.push_back(PredBB);
      }
    } while (!Worklist.empty());
  }

  // Now that the ExitLoopMap gives as  mapping for all the non-looping cloned
  // blocks to their outer loops, walk the cloned blocks and the cloned exits
  // in their original order adding them to the correct loop.

  // We need a stable insertion order. We use the order of the original loop
  // order and map into the correct parent loop.
  for (auto *BB : llvm::concat<BasicBlock *const>(
           makeArrayRef(ClonedPH), ClonedLoopBlocks, ClonedExitsInLoops))
    if (Loop *OuterL = ExitLoopMap.lookup(BB))
      OuterL->addBasicBlockToLoop(BB, LI);

#ifndef NDEBUG
  for (auto &BBAndL : ExitLoopMap) {
    auto *BB = BBAndL.first;
    auto *OuterL = BBAndL.second;
    assert(LI.getLoopFor(BB) == OuterL &&
           "Failed to put all blocks into outer loops!");
  }
#endif

  // Now that all the blocks are placed into the correct containing loop in the
  // absence of child loops, find all the potentially cloned child loops and
  // clone them into whatever outer loop we placed their header into.
  for (Loop *ChildL : OrigL) {
    auto *ClonedChildHeader =
        cast_or_null<BasicBlock>(VMap.lookup(ChildL->getHeader()));
    if (!ClonedChildHeader || BlocksInClonedLoop.count(ClonedChildHeader))
      continue;

#ifndef NDEBUG
    for (auto *ChildLoopBB : ChildL->blocks())
      assert(VMap.count(ChildLoopBB) &&
             "Cloned a child loop header but not all of that loops blocks!");
#endif

    NonChildClonedLoops.push_back(cloneLoopNest(
        *ChildL, ExitLoopMap.lookup(ClonedChildHeader), VMap, LI));
  }

  // Return the main cloned loop if any.
  return ClonedL;
}

static void deleteDeadBlocksFromLoop(Loop &L, BasicBlock *DeadSubtreeRoot,
                                     SmallVectorImpl<BasicBlock *> &ExitBlocks,
                                     DominatorTree &DT, LoopInfo &LI) {
  // Walk the dominator tree to build up the set of blocks we will delete here.
  // The order is designed to allow us to always delete bottom-up and avoid any
  // dangling uses.
  SmallSetVector<BasicBlock *, 16> DeadBlocks;
  DeadBlocks.insert(DeadSubtreeRoot);
  for (int i = 0; i < (int)DeadBlocks.size(); ++i)
    for (DomTreeNode *ChildN : *DT[DeadBlocks[i]]) {
      // FIXME: This assert should pass and that means we don't change nearly
      // as much below! Consider rewriting all of this to avoid deleting
      // blocks. They are always cloned before being deleted, and so instead
      // could just be moved.
      // FIXME: This in turn means that we might actually be more able to
      // update the domtree.
      assert((L.contains(ChildN->getBlock()) ||
              llvm::find(ExitBlocks, ChildN->getBlock()) != ExitBlocks.end()) &&
             "Should never reach beyond the loop and exits when deleting!");
      DeadBlocks.insert(ChildN->getBlock());
    }

  // Filter out the dead blocks from the exit blocks list so that it can be
  // used in the caller.
  llvm::erase_if(ExitBlocks,
                 [&](BasicBlock *BB) { return DeadBlocks.count(BB); });

  // Remove these blocks from their successors.
  for (auto *BB : DeadBlocks)
    for (BasicBlock *SuccBB : successors(BB))
      SuccBB->removePredecessor(BB, /*DontDeleteUselessPHIs*/ true);

  // Walk from this loop up through its parents removing all of the dead blocks.
  for (Loop *ParentL = &L; ParentL; ParentL = ParentL->getParentLoop()) {
    for (auto *BB : DeadBlocks)
      ParentL->getBlocksSet().erase(BB);
    llvm::erase_if(ParentL->getBlocksVector(),
                   [&](BasicBlock *BB) { return DeadBlocks.count(BB); });
  }

  // Now delete the dead child loops. This raw delete will clear them
  // recursively.
  llvm::erase_if(L.getSubLoopsVector(), [&](Loop *ChildL) {
    if (!DeadBlocks.count(ChildL->getHeader()))
      return false;

    assert(llvm::all_of(ChildL->blocks(),
                        [&](BasicBlock *ChildBB) {
                          return DeadBlocks.count(ChildBB);
                        }) &&
           "If the child loop header is dead all blocks in the child loop must "
           "be dead as well!");
    LI.destroy(ChildL);
    return true;
  });

  // Remove the mappings for the dead blocks.
  for (auto *BB : DeadBlocks)
    LI.changeLoopFor(BB, nullptr);

  // Drop all the references from these blocks to others to handle cyclic
  // references as we start deleting the blocks themselves.
  for (auto *BB : DeadBlocks)
    BB->dropAllReferences();

  for (auto *BB : llvm::reverse(DeadBlocks)) {
    DT.eraseNode(BB);
    BB->eraseFromParent();
  }
}

/// Recompute the set of blocks in a loop after unswitching.
///
/// This walks from the original headers predecessors to rebuild the loop. We
/// take advantage of the fact that new blocks can't have been added, and so we
/// filter by the original loop's blocks. This also handles potentially
/// unreachable code that we don't want to explore but might be found examining
/// the predecessors of the header.
///
/// If the original loop is no longer a loop, this will return an empty set. If
/// it remains a loop, all the blocks within it will be added to the set
/// (including those blocks in inner loops).
static SmallPtrSet<const BasicBlock *, 16> recomputeLoopBlockSet(Loop &L,
                                                                 LoopInfo &LI) {
  SmallPtrSet<const BasicBlock *, 16> LoopBlockSet;

  auto *PH = L.getLoopPreheader();
  auto *Header = L.getHeader();

  // A worklist to use while walking backwards from the header.
  SmallVector<BasicBlock *, 16> Worklist;

  // First walk the predecessors of the header to find the backedges. This will
  // form the basis of our walk.
  for (auto *Pred : predecessors(Header)) {
    // Skip the preheader.
    if (Pred == PH)
      continue;

    // Because the loop was in simplified form, the only non-loop predecessor
    // is the preheader.
    assert(L.contains(Pred) && "Found a predecessor of the loop header other "
                               "than the preheader that is not part of the "
                               "loop!");

    // Insert this block into the loop set and on the first visit and, if it
    // isn't the header we're currently walking, put it into the worklist to
    // recurse through.
    if (LoopBlockSet.insert(Pred).second && Pred != Header)
      Worklist.push_back(Pred);
  }

  // If no backedges were found, we're done.
  if (LoopBlockSet.empty())
    return LoopBlockSet;

  // Add the loop header to the set.
  LoopBlockSet.insert(Header);

  // We found backedges, recurse through them to identify the loop blocks.
  while (!Worklist.empty()) {
    BasicBlock *BB = Worklist.pop_back_val();
    assert(LoopBlockSet.count(BB) && "Didn't put block into the loop set!");

    // Because we know the inner loop structure remains valid we can use the
    // loop structure to jump immediately across the entire nested loop.
    // Further, because it is in loop simplified form, we can directly jump
    // to its preheader afterward.
    if (Loop *InnerL = LI.getLoopFor(BB))
      if (InnerL != &L) {
        assert(L.contains(InnerL) &&
               "Should not reach a loop *outside* this loop!");
        // The preheader is the only possible predecessor of the loop so
        // insert it into the set and check whether it was already handled.
        auto *InnerPH = InnerL->getLoopPreheader();
        assert(L.contains(InnerPH) && "Cannot contain an inner loop block "
                                      "but not contain the inner loop "
                                      "preheader!");
        if (!LoopBlockSet.insert(InnerPH).second)
          // The only way to reach the preheader is through the loop body
          // itself so if it has been visited the loop is already handled.
          continue;

        // Insert all of the blocks (other than those already present) into
        // the loop set. The only block we expect to already be in the set is
        // the one we used to find this loop as we immediately handle the
        // others the first time we encounter the loop.
        for (auto *InnerBB : InnerL->blocks()) {
          if (InnerBB == BB) {
            assert(LoopBlockSet.count(InnerBB) &&
                   "Block should already be in the set!");
            continue;
          }

          bool Inserted = LoopBlockSet.insert(InnerBB).second;
          (void)Inserted;
          assert(Inserted && "Should only insert an inner loop once!");
        }

        // Add the preheader to the worklist so we will continue past the
        // loop body.
        Worklist.push_back(InnerPH);
        continue;
      }

    // Insert any predecessors that were in the original loop into the new
    // set, and if the insert is successful, add them to the worklist.
    for (auto *Pred : predecessors(BB))
      if (L.contains(Pred) && LoopBlockSet.insert(Pred).second)
        Worklist.push_back(Pred);
  }

  // We've found all the blocks participating in the loop, return our completed
  // set.
  return LoopBlockSet;
}

/// Rebuild a loop after unswitching removes some subset of blocks and edges.
///
/// The removal may have removed some child loops entirely but cannot have
/// disturbed any remaining child loops. However, they may need to be hoisted
/// to the parent loop (or to be top-level loops). The original loop may be
/// completely removed.
///
/// The sibling loops resulting from this update are returned. If the original
/// loop remains a valid loop, it will be the first entry in this list with all
/// of the newly sibling loops following it.
///
/// Returns true if the loop remains a loop after unswitching, and false if it
/// is no longer a loop after unswitching (and should not continue to be
/// referenced).
static bool rebuildLoopAfterUnswitch(Loop &L, ArrayRef<BasicBlock *> ExitBlocks,
                                     LoopInfo &LI,
                                     SmallVectorImpl<Loop *> &HoistedLoops) {
  auto *PH = L.getLoopPreheader();

  // Compute the actual parent loop from the exit blocks. Because we may have
  // pruned some exits the loop may be different from the original parent.
  Loop *ParentL = nullptr;
  SmallVector<Loop *, 4> ExitLoops;
  SmallVector<BasicBlock *, 4> ExitsInLoops;
  ExitsInLoops.reserve(ExitBlocks.size());
  for (auto *ExitBB : ExitBlocks)
    if (Loop *ExitL = LI.getLoopFor(ExitBB)) {
      ExitLoops.push_back(ExitL);
      ExitsInLoops.push_back(ExitBB);
      if (!ParentL || (ParentL != ExitL && ParentL->contains(ExitL)))
        ParentL = ExitL;
    }

  // Recompute the blocks participating in this loop. This may be empty if it
  // is no longer a loop.
  auto LoopBlockSet = recomputeLoopBlockSet(L, LI);

  // If we still have a loop, we need to re-set the loop's parent as the exit
  // block set changing may have moved it within the loop nest. Note that this
  // can only happen when this loop has a parent as it can only hoist the loop
  // *up* the nest.
  if (!LoopBlockSet.empty() && L.getParentLoop() != ParentL) {
    // Remove this loop's (original) blocks from all of the intervening loops.
    for (Loop *IL = L.getParentLoop(); IL != ParentL;
         IL = IL->getParentLoop()) {
      IL->getBlocksSet().erase(PH);
      for (auto *BB : L.blocks())
        IL->getBlocksSet().erase(BB);
      llvm::erase_if(IL->getBlocksVector(), [&](BasicBlock *BB) {
        return BB == PH || L.contains(BB);
      });
    }

    LI.changeLoopFor(PH, ParentL);
    L.getParentLoop()->removeChildLoop(&L);
    if (ParentL)
      ParentL->addChildLoop(&L);
    else
      LI.addTopLevelLoop(&L);
  }

  // Now we update all the blocks which are no longer within the loop.
  auto &Blocks = L.getBlocksVector();
  auto BlocksSplitI =
      LoopBlockSet.empty()
          ? Blocks.begin()
          : std::stable_partition(
                Blocks.begin(), Blocks.end(),
                [&](BasicBlock *BB) { return LoopBlockSet.count(BB); });

  // Before we erase the list of unlooped blocks, build a set of them.
  SmallPtrSet<BasicBlock *, 16> UnloopedBlocks(BlocksSplitI, Blocks.end());
  if (LoopBlockSet.empty())
    UnloopedBlocks.insert(PH);

  // Now erase these blocks from the loop.
  for (auto *BB : make_range(BlocksSplitI, Blocks.end()))
    L.getBlocksSet().erase(BB);
  Blocks.erase(BlocksSplitI, Blocks.end());

  // Sort the exits in ascending loop depth, we'll work backwards across these
  // to process them inside out.
  std::stable_sort(ExitsInLoops.begin(), ExitsInLoops.end(),
                   [&](BasicBlock *LHS, BasicBlock *RHS) {
                     return LI.getLoopDepth(LHS) < LI.getLoopDepth(RHS);
                   });

  // We'll build up a set for each exit loop.
  SmallPtrSet<BasicBlock *, 16> NewExitLoopBlocks;
  Loop *PrevExitL = L.getParentLoop(); // The deepest possible exit loop.

  auto RemoveUnloopedBlocksFromLoop =
      [](Loop &L, SmallPtrSetImpl<BasicBlock *> &UnloopedBlocks) {
        for (auto *BB : UnloopedBlocks)
          L.getBlocksSet().erase(BB);
        llvm::erase_if(L.getBlocksVector(), [&](BasicBlock *BB) {
          return UnloopedBlocks.count(BB);
        });
      };

  SmallVector<BasicBlock *, 16> Worklist;
  while (!UnloopedBlocks.empty() && !ExitsInLoops.empty()) {
    assert(Worklist.empty() && "Didn't clear worklist!");
    assert(NewExitLoopBlocks.empty() && "Didn't clear loop set!");

    // Grab the next exit block, in decreasing loop depth order.
    BasicBlock *ExitBB = ExitsInLoops.pop_back_val();
    Loop &ExitL = *LI.getLoopFor(ExitBB);
    assert(ExitL.contains(&L) && "Exit loop must contain the inner loop!");

    // Erase all of the unlooped blocks from the loops between the previous
    // exit loop and this exit loop. This works because the ExitInLoops list is
    // sorted in increasing order of loop depth and thus we visit loops in
    // decreasing order of loop depth.
    for (; PrevExitL != &ExitL; PrevExitL = PrevExitL->getParentLoop())
      RemoveUnloopedBlocksFromLoop(*PrevExitL, UnloopedBlocks);

    // Walk the CFG back until we hit the cloned PH adding everything reachable
    // and in the unlooped set to this exit block's loop.
    Worklist.push_back(ExitBB);
    do {
      BasicBlock *BB = Worklist.pop_back_val();
      // We can stop recursing at the cloned preheader (if we get there).
      if (BB == PH)
        continue;

      for (BasicBlock *PredBB : predecessors(BB)) {
        // If this pred has already been moved to our set or is part of some
        // (inner) loop, no update needed.
        if (!UnloopedBlocks.erase(PredBB)) {
          assert((NewExitLoopBlocks.count(PredBB) ||
                  ExitL.contains(LI.getLoopFor(PredBB))) &&
                 "Predecessor not in a nested loop (or already visited)!");
          continue;
        }

        // We just insert into the loop set here. We'll add these blocks to the
        // exit loop after we build up the set in a deterministic order rather
        // than the predecessor-influenced visit order.
        bool Inserted = NewExitLoopBlocks.insert(PredBB).second;
        (void)Inserted;
        assert(Inserted && "Should only visit an unlooped block once!");

        // And recurse through to its predecessors.
        Worklist.push_back(PredBB);
      }
    } while (!Worklist.empty());

    // If blocks in this exit loop were directly part of the original loop (as
    // opposed to a child loop) update the map to point to this exit loop. This
    // just updates a map and so the fact that the order is unstable is fine.
    for (auto *BB : NewExitLoopBlocks)
      if (Loop *BBL = LI.getLoopFor(BB))
        if (BBL == &L || !L.contains(BBL))
          LI.changeLoopFor(BB, &ExitL);

    // We will remove the remaining unlooped blocks from this loop in the next
    // iteration or below.
    NewExitLoopBlocks.clear();
  }

  // Any remaining unlooped blocks are no longer part of any loop unless they
  // are part of some child loop.
  for (; PrevExitL; PrevExitL = PrevExitL->getParentLoop())
    RemoveUnloopedBlocksFromLoop(*PrevExitL, UnloopedBlocks);
  for (auto *BB : UnloopedBlocks)
    if (Loop *BBL = LI.getLoopFor(BB))
      if (BBL == &L || !L.contains(BBL))
        LI.changeLoopFor(BB, nullptr);

  // Sink all the child loops whose headers are no longer in the loop set to
  // the parent (or to be top level loops). We reach into the loop and directly
  // update its subloop vector to make this batch update efficient.
  auto &SubLoops = L.getSubLoopsVector();
  auto SubLoopsSplitI =
      LoopBlockSet.empty()
          ? SubLoops.begin()
          : std::stable_partition(
                SubLoops.begin(), SubLoops.end(), [&](Loop *SubL) {
                  return LoopBlockSet.count(SubL->getHeader());
                });
  for (auto *HoistedL : make_range(SubLoopsSplitI, SubLoops.end())) {
    HoistedLoops.push_back(HoistedL);
    HoistedL->setParentLoop(nullptr);

    // To compute the new parent of this hoisted loop we look at where we
    // placed the preheader above. We can't lookup the header itself because we
    // retained the mapping from the header to the hoisted loop. But the
    // preheader and header should have the exact same new parent computed
    // based on the set of exit blocks from the original loop as the preheader
    // is a predecessor of the header and so reached in the reverse walk. And
    // because the loops were all in simplified form the preheader of the
    // hoisted loop can't be part of some *other* loop.
    if (auto *NewParentL = LI.getLoopFor(HoistedL->getLoopPreheader()))
      NewParentL->addChildLoop(HoistedL);
    else
      LI.addTopLevelLoop(HoistedL);
  }
  SubLoops.erase(SubLoopsSplitI, SubLoops.end());

  // Actually delete the loop if nothing remained within it.
  if (Blocks.empty()) {
    assert(SubLoops.empty() &&
           "Failed to remove all subloops from the original loop!");
    if (Loop *ParentL = L.getParentLoop())
      ParentL->removeChildLoop(llvm::find(*ParentL, &L));
    else
      LI.removeLoop(llvm::find(LI, &L));
    LI.destroy(&L);
    return false;
  }

  return true;
}

/// Helper to visit a dominator subtree, invoking a callable on each node.
///
/// Returning false at any point will stop walking past that node of the tree.
template <typename CallableT>
void visitDomSubTree(DominatorTree &DT, BasicBlock *BB, CallableT Callable) {
  SmallVector<DomTreeNode *, 4> DomWorklist;
  DomWorklist.push_back(DT[BB]);
#ifndef NDEBUG
  SmallPtrSet<DomTreeNode *, 4> Visited;
  Visited.insert(DT[BB]);
#endif
  do {
    DomTreeNode *N = DomWorklist.pop_back_val();

    // Visit this node.
    if (!Callable(N->getBlock()))
      continue;

    // Accumulate the child nodes.
    for (DomTreeNode *ChildN : *N) {
      assert(Visited.insert(ChildN).second &&
             "Cannot visit a node twice when walking a tree!");
      DomWorklist.push_back(ChildN);
    }
  } while (!DomWorklist.empty());
}

/// Take an invariant branch that has been determined to be safe and worthwhile
/// to unswitch despite being non-trivial to do so and perform the unswitch.
///
/// This directly updates the CFG to hoist the predicate out of the loop, and
/// clone the necessary parts of the loop to maintain behavior.
///
/// It also updates both dominator tree and loopinfo based on the unswitching.
///
/// Once unswitching has been performed it runs the provided callback to report
/// the new loops and no-longer valid loops to the caller.
static bool unswitchInvariantBranch(
    Loop &L, BranchInst &BI, DominatorTree &DT, LoopInfo &LI,
    AssumptionCache &AC,
    function_ref<void(bool, ArrayRef<Loop *>)> NonTrivialUnswitchCB) {
  assert(BI.isConditional() && "Can only unswitch a conditional branch!");
  assert(L.isLoopInvariant(BI.getCondition()) &&
         "Can only unswitch an invariant branch condition!");

  // Constant and BBs tracking the cloned and continuing successor.
  const int ClonedSucc = 0;
  auto *ParentBB = BI.getParent();
  auto *UnswitchedSuccBB = BI.getSuccessor(ClonedSucc);
  auto *ContinueSuccBB = BI.getSuccessor(1 - ClonedSucc);

  assert(UnswitchedSuccBB != ContinueSuccBB &&
         "Should not unswitch a branch that always goes to the same place!");

  // The branch should be in this exact loop. Any inner loop's invariant branch
  // should be handled by unswitching that inner loop. The caller of this
  // routine should filter out any candidates that remain (but were skipped for
  // whatever reason).
  assert(LI.getLoopFor(ParentBB) == &L && "Branch in an inner loop!");

  SmallVector<BasicBlock *, 4> ExitBlocks;
  L.getUniqueExitBlocks(ExitBlocks);

  // We cannot unswitch if exit blocks contain a cleanuppad instruction as we
  // don't know how to split those exit blocks.
  // FIXME: We should teach SplitBlock to handle this and remove this
  // restriction.
  for (auto *ExitBB : ExitBlocks)
    if (isa<CleanupPadInst>(ExitBB->getFirstNonPHI()))
      return false;

  SmallPtrSet<BasicBlock *, 4> ExitBlockSet(ExitBlocks.begin(),
                                            ExitBlocks.end());

  // Compute the parent loop now before we start hacking on things.
  Loop *ParentL = L.getParentLoop();

  // Compute the outer-most loop containing one of our exit blocks. This is the
  // furthest up our loopnest which can be mutated, which we will use below to
  // update things.
  Loop *OuterExitL = &L;
  for (auto *ExitBB : ExitBlocks) {
    Loop *NewOuterExitL = LI.getLoopFor(ExitBB);
    if (!NewOuterExitL) {
      // We exited the entire nest with this block, so we're done.
      OuterExitL = nullptr;
      break;
    }
    if (NewOuterExitL != OuterExitL && NewOuterExitL->contains(OuterExitL))
      OuterExitL = NewOuterExitL;
  }

  // If the edge we *aren't* cloning in the unswitch (the continuing edge)
  // dominates its target, we can skip cloning the dominated region of the loop
  // and its exits. We compute this as a set of nodes to be skipped.
  SmallPtrSet<BasicBlock *, 4> SkippedLoopAndExitBlocks;
  if (ContinueSuccBB->getUniquePredecessor() ||
      llvm::all_of(predecessors(ContinueSuccBB), [&](BasicBlock *PredBB) {
        return PredBB == ParentBB || DT.dominates(ContinueSuccBB, PredBB);
      })) {
    visitDomSubTree(DT, ContinueSuccBB, [&](BasicBlock *BB) {
      SkippedLoopAndExitBlocks.insert(BB);
      return true;
    });
  }
  // Similarly, if the edge we *are* cloning in the unswitch (the unswitched
  // edge) dominates its target, we will end up with dead nodes in the original
  // loop and its exits that will need to be deleted. Here, we just retain that
  // the property holds and will compute the deleted set later.
  bool DeleteUnswitchedSucc =
      UnswitchedSuccBB->getUniquePredecessor() ||
      llvm::all_of(predecessors(UnswitchedSuccBB), [&](BasicBlock *PredBB) {
        return PredBB == ParentBB || DT.dominates(UnswitchedSuccBB, PredBB);
      });

  // Split the preheader, so that we know that there is a safe place to insert
  // the conditional branch. We will change the preheader to have a conditional
  // branch on LoopCond. The original preheader will become the split point
  // between the unswitched versions, and we will have a new preheader for the
  // original loop.
  BasicBlock *SplitBB = L.getLoopPreheader();
  BasicBlock *LoopPH = SplitEdge(SplitBB, L.getHeader(), &DT, &LI);

  // Keep a mapping for the cloned values.
  ValueToValueMapTy VMap;

  // Build the cloned blocks from the loop.
  auto *ClonedPH = buildClonedLoopBlocks(
      L, LoopPH, SplitBB, ExitBlocks, ParentBB, UnswitchedSuccBB,
      ContinueSuccBB, SkippedLoopAndExitBlocks, VMap, AC, DT, LI);

  // Build the cloned loop structure itself. This may be substantially
  // different from the original structure due to the simplified CFG. This also
  // handles inserting all the cloned blocks into the correct loops.
  SmallVector<Loop *, 4> NonChildClonedLoops;
  Loop *ClonedL =
      buildClonedLoops(L, ExitBlocks, VMap, LI, NonChildClonedLoops);

  // Remove the parent as a predecessor of the unswitched successor.
  UnswitchedSuccBB->removePredecessor(ParentBB, /*DontDeleteUselessPHIs*/ true);

  // Now splice the branch from the original loop and use it to select between
  // the two loops.
  SplitBB->getTerminator()->eraseFromParent();
  SplitBB->getInstList().splice(SplitBB->end(), ParentBB->getInstList(), BI);
  BI.setSuccessor(ClonedSucc, ClonedPH);
  BI.setSuccessor(1 - ClonedSucc, LoopPH);

  // Create a new unconditional branch to the continuing block (as opposed to
  // the one cloned).
  BranchInst::Create(ContinueSuccBB, ParentBB);

  // Delete anything that was made dead in the original loop due to
  // unswitching.
  if (DeleteUnswitchedSucc)
    deleteDeadBlocksFromLoop(L, UnswitchedSuccBB, ExitBlocks, DT, LI);

  SmallVector<Loop *, 4> HoistedLoops;
  bool IsStillLoop = rebuildLoopAfterUnswitch(L, ExitBlocks, LI, HoistedLoops);

  // This will have completely invalidated the dominator tree. We can't easily
  // bound how much is invalid because in some cases we will refine the
  // predecessor set of exit blocks of the loop which can move large unrelated
  // regions of code into a new subtree.
  //
  // FIXME: Eventually, we should use an incremental update utility that
  // leverages the existing information in the dominator tree (and potentially
  // the nature of the change) to more efficiently update things.
  DT.recalculate(*SplitBB->getParent());

  // We can change which blocks are exit blocks of all the cloned sibling
  // loops, the current loop, and any parent loops which shared exit blocks
  // with the current loop. As a consequence, we need to re-form LCSSA for
  // them. But we shouldn't need to re-form LCSSA for any child loops.
  // FIXME: This could be made more efficient by tracking which exit blocks are
  // new, and focusing on them, but that isn't likely to be necessary.
  //
  // In order to reasonably rebuild LCSSA we need to walk inside-out across the
  // loop nest and update every loop that could have had its exits changed. We
  // also need to cover any intervening loops. We add all of these loops to
  // a list and sort them by loop depth to achieve this without updating
  // unnecessary loops.
  auto UpdateLCSSA = [&](Loop &UpdateL) {
#ifndef NDEBUG
    for (Loop *ChildL : UpdateL)
      assert(ChildL->isRecursivelyLCSSAForm(DT, LI) &&
             "Perturbed a child loop's LCSSA form!");
#endif
    formLCSSA(UpdateL, DT, &LI, nullptr);
  };

  // For non-child cloned loops and hoisted loops, we just need to update LCSSA
  // and we can do it in any order as they don't nest relative to each other.
  for (Loop *UpdatedL : llvm::concat<Loop *>(NonChildClonedLoops, HoistedLoops))
    UpdateLCSSA(*UpdatedL);

  // If the original loop had exit blocks, walk up through the outer most loop
  // of those exit blocks to update LCSSA and form updated dedicated exits.
  if (OuterExitL != &L) {
    SmallVector<Loop *, 4> OuterLoops;
    // We start with the cloned loop and the current loop if they are loops and
    // move toward OuterExitL. Also, if either the cloned loop or the current
    // loop have become top level loops we need to walk all the way out.
    if (ClonedL) {
      OuterLoops.push_back(ClonedL);
      if (!ClonedL->getParentLoop())
        OuterExitL = nullptr;
    }
    if (IsStillLoop) {
      OuterLoops.push_back(&L);
      if (!L.getParentLoop())
        OuterExitL = nullptr;
    }
    // Grab all of the enclosing loops now.
    for (Loop *OuterL = ParentL; OuterL != OuterExitL;
         OuterL = OuterL->getParentLoop())
      OuterLoops.push_back(OuterL);

    // Finally, update our list of outer loops. This is nicely ordered to work
    // inside-out.
    for (Loop *OuterL : OuterLoops) {
      // First build LCSSA for this loop so that we can preserve it when
      // forming dedicated exits. We don't want to perturb some other loop's
      // LCSSA while doing that CFG edit.
      UpdateLCSSA(*OuterL);

      // For loops reached by this loop's original exit blocks we may
      // introduced new, non-dedicated exits. At least try to re-form dedicated
      // exits for these loops. This may fail if they couldn't have dedicated
      // exits to start with.
      formDedicatedExitBlocks(OuterL, &DT, &LI, /*PreserveLCSSA*/ true);
    }
  }

#ifndef NDEBUG
  // Verify the entire loop structure to catch any incorrect updates before we
  // progress in the pass pipeline.
  LI.verify(DT);
#endif

  // Now that we've unswitched something, make callbacks to report the changes.
  // For that we need to merge together the updated loops and the cloned loops
  // and check whether the original loop survived.
  SmallVector<Loop *, 4> SibLoops;
  for (Loop *UpdatedL : llvm::concat<Loop *>(NonChildClonedLoops, HoistedLoops))
    if (UpdatedL->getParentLoop() == ParentL)
      SibLoops.push_back(UpdatedL);
  NonTrivialUnswitchCB(IsStillLoop, SibLoops);

  ++NumBranches;
  return true;
}

/// Recursively compute the cost of a dominator subtree based on the per-block
/// cost map provided.
///
/// The recursive computation is memozied into the provided DT-indexed cost map
/// to allow querying it for most nodes in the domtree without it becoming
/// quadratic.
static int
computeDomSubtreeCost(DomTreeNode &N,
                      const SmallDenseMap<BasicBlock *, int, 4> &BBCostMap,
                      SmallDenseMap<DomTreeNode *, int, 4> &DTCostMap) {
  // Don't accumulate cost (or recurse through) blocks not in our block cost
  // map and thus not part of the duplication cost being considered.
  auto BBCostIt = BBCostMap.find(N.getBlock());
  if (BBCostIt == BBCostMap.end())
    return 0;

  // Lookup this node to see if we already computed its cost.
  auto DTCostIt = DTCostMap.find(&N);
  if (DTCostIt != DTCostMap.end())
    return DTCostIt->second;

  // If not, we have to compute it. We can't use insert above and update
  // because computing the cost may insert more things into the map.
  int Cost = std::accumulate(
      N.begin(), N.end(), BBCostIt->second, [&](int Sum, DomTreeNode *ChildN) {
        return Sum + computeDomSubtreeCost(*ChildN, BBCostMap, DTCostMap);
      });
  bool Inserted = DTCostMap.insert({&N, Cost}).second;
  (void)Inserted;
  assert(Inserted && "Should not insert a node while visiting children!");
  return Cost;
}

/// Unswitch control flow predicated on loop invariant conditions.
///
/// This first hoists all branches or switches which are trivial (IE, do not
/// require duplicating any part of the loop) out of the loop body. It then
/// looks at other loop invariant control flows and tries to unswitch those as
/// well by cloning the loop if the result is small enough.
static bool
unswitchLoop(Loop &L, DominatorTree &DT, LoopInfo &LI, AssumptionCache &AC,
             TargetTransformInfo &TTI, bool NonTrivial,
             function_ref<void(bool, ArrayRef<Loop *>)> NonTrivialUnswitchCB) {
  assert(L.isRecursivelyLCSSAForm(DT, LI) &&
         "Loops must be in LCSSA form before unswitching.");
  bool Changed = false;

  // Must be in loop simplified form: we need a preheader and dedicated exits.
  if (!L.isLoopSimplifyForm())
    return false;

  // Try trivial unswitch first before loop over other basic blocks in the loop.
  Changed |= unswitchAllTrivialConditions(L, DT, LI);

  // If we're not doing non-trivial unswitching, we're done. We both accept
  // a parameter but also check a local flag that can be used for testing
  // a debugging.
  if (!NonTrivial && !EnableNonTrivialUnswitch)
    return Changed;

  // Collect all remaining invariant branch conditions within this loop (as
  // opposed to an inner loop which would be handled when visiting that inner
  // loop).
  SmallVector<TerminatorInst *, 4> UnswitchCandidates;
  for (auto *BB : L.blocks())
    if (LI.getLoopFor(BB) == &L)
      if (auto *BI = dyn_cast<BranchInst>(BB->getTerminator()))
        if (BI->isConditional() && L.isLoopInvariant(BI->getCondition()) &&
            BI->getSuccessor(0) != BI->getSuccessor(1))
          UnswitchCandidates.push_back(BI);

  // If we didn't find any candidates, we're done.
  if (UnswitchCandidates.empty())
    return Changed;

  DEBUG(dbgs() << "Considering " << UnswitchCandidates.size()
               << " non-trivial loop invariant conditions for unswitching.\n");

  // Given that unswitching these terminators will require duplicating parts of
  // the loop, so we need to be able to model that cost. Compute the ephemeral
  // values and set up a data structure to hold per-BB costs. We cache each
  // block's cost so that we don't recompute this when considering different
  // subsets of the loop for duplication during unswitching.
  SmallPtrSet<const Value *, 4> EphValues;
  CodeMetrics::collectEphemeralValues(&L, &AC, EphValues);
  SmallDenseMap<BasicBlock *, int, 4> BBCostMap;

  // Compute the cost of each block, as well as the total loop cost. Also, bail
  // out if we see instructions which are incompatible with loop unswitching
  // (convergent, noduplicate, or cross-basic-block tokens).
  // FIXME: We might be able to safely handle some of these in non-duplicated
  // regions.
  int LoopCost = 0;
  for (auto *BB : L.blocks()) {
    int Cost = 0;
    for (auto &I : *BB) {
      if (EphValues.count(&I))
        continue;

      if (I.getType()->isTokenTy() && I.isUsedOutsideOfBlock(BB))
        return Changed;
      if (auto CS = CallSite(&I))
        if (CS.isConvergent() || CS.cannotDuplicate())
          return Changed;

      Cost += TTI.getUserCost(&I);
    }
    assert(Cost >= 0 && "Must not have negative costs!");
    LoopCost += Cost;
    assert(LoopCost >= 0 && "Must not have negative loop costs!");
    BBCostMap[BB] = Cost;
  }
  DEBUG(dbgs() << "  Total loop cost: " << LoopCost << "\n");

  // Now we find the best candidate by searching for the one with the following
  // properties in order:
  //
  // 1) An unswitching cost below the threshold
  // 2) The smallest number of duplicated unswitch candidates (to avoid
  //    creating redundant subsequent unswitching)
  // 3) The smallest cost after unswitching.
  //
  // We prioritize reducing fanout of unswitch candidates provided the cost
  // remains below the threshold because this has a multiplicative effect.
  //
  // This requires memoizing each dominator subtree to avoid redundant work.
  //
  // FIXME: Need to actually do the number of candidates part above.
  SmallDenseMap<DomTreeNode *, int, 4> DTCostMap;
  // Given a terminator which might be unswitched, computes the non-duplicated
  // cost for that terminator.
  auto ComputeUnswitchedCost = [&](TerminatorInst *TI) {
    BasicBlock &BB = *TI->getParent();
    SmallPtrSet<BasicBlock *, 4> Visited;

    int Cost = LoopCost;
    for (BasicBlock *SuccBB : successors(&BB)) {
      // Don't count successors more than once.
      if (!Visited.insert(SuccBB).second)
        continue;

      // This successor's domtree will not need to be duplicated after
      // unswitching if the edge to the successor dominates it (and thus the
      // entire tree). This essentially means there is no other path into this
      // subtree and so it will end up live in only one clone of the loop.
      if (SuccBB->getUniquePredecessor() ||
          llvm::all_of(predecessors(SuccBB), [&](BasicBlock *PredBB) {
            return PredBB == &BB || DT.dominates(SuccBB, PredBB);
          })) {
        Cost -= computeDomSubtreeCost(*DT[SuccBB], BBCostMap, DTCostMap);
        assert(Cost >= 0 &&
               "Non-duplicated cost should never exceed total loop cost!");
      }
    }

    // Now scale the cost by the number of unique successors minus one. We
    // subtract one because there is already at least one copy of the entire
    // loop. This is computing the new cost of unswitching a condition.
    assert(Visited.size() > 1 &&
           "Cannot unswitch a condition without multiple distinct successors!");
    return Cost * (Visited.size() - 1);
  };
  TerminatorInst *BestUnswitchTI = nullptr;
  int BestUnswitchCost;
  for (TerminatorInst *CandidateTI : UnswitchCandidates) {
    int CandidateCost = ComputeUnswitchedCost(CandidateTI);
    DEBUG(dbgs() << "  Computed cost of " << CandidateCost
                 << " for unswitch candidate: " << *CandidateTI << "\n");
    if (!BestUnswitchTI || CandidateCost < BestUnswitchCost) {
      BestUnswitchTI = CandidateTI;
      BestUnswitchCost = CandidateCost;
    }
  }

  if (BestUnswitchCost < UnswitchThreshold) {
    DEBUG(dbgs() << "  Trying to unswitch non-trivial (cost = "
                 << BestUnswitchCost << ") branch: " << *BestUnswitchTI
                 << "\n");
    Changed |= unswitchInvariantBranch(L, cast<BranchInst>(*BestUnswitchTI), DT,
                                       LI, AC, NonTrivialUnswitchCB);
  } else {
    DEBUG(dbgs() << "Cannot unswitch, lowest cost found: " << BestUnswitchCost
                 << "\n");
  }

  return Changed;
}

PreservedAnalyses SimpleLoopUnswitchPass::run(Loop &L, LoopAnalysisManager &AM,
                                              LoopStandardAnalysisResults &AR,
                                              LPMUpdater &U) {
  Function &F = *L.getHeader()->getParent();
  (void)F;

  DEBUG(dbgs() << "Unswitching loop in " << F.getName() << ": " << L << "\n");

  // Save the current loop name in a variable so that we can report it even
  // after it has been deleted.
  std::string LoopName = L.getName();

  auto NonTrivialUnswitchCB = [&L, &U, &LoopName](bool CurrentLoopValid,
                                                  ArrayRef<Loop *> NewLoops) {
    // If we did a non-trivial unswitch, we have added new (cloned) loops.
    U.addSiblingLoops(NewLoops);

    // If the current loop remains valid, we should revisit it to catch any
    // other unswitch opportunities. Otherwise, we need to mark it as deleted.
    if (CurrentLoopValid)
      U.revisitCurrentLoop();
    else
      U.markLoopAsDeleted(L, LoopName);
  };

  if (!unswitchLoop(L, AR.DT, AR.LI, AR.AC, AR.TTI, NonTrivial,
                    NonTrivialUnswitchCB))
    return PreservedAnalyses::all();

  // Historically this pass has had issues with the dominator tree so verify it
  // in asserts builds.
  assert(AR.DT.verify(DominatorTree::VerificationLevel::Fast));
  return getLoopPassPreservedAnalyses();
}

namespace {

class SimpleLoopUnswitchLegacyPass : public LoopPass {
  bool NonTrivial;

public:
  static char ID; // Pass ID, replacement for typeid

  explicit SimpleLoopUnswitchLegacyPass(bool NonTrivial = false)
      : LoopPass(ID), NonTrivial(NonTrivial) {
    initializeSimpleLoopUnswitchLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnLoop(Loop *L, LPPassManager &LPM) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
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
  auto &TTI = getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);

  auto NonTrivialUnswitchCB = [&L, &LPM](bool CurrentLoopValid,
                                         ArrayRef<Loop *> NewLoops) {
    // If we did a non-trivial unswitch, we have added new (cloned) loops.
    for (auto *NewL : NewLoops)
      LPM.addLoop(*NewL);

    // If the current loop remains valid, re-add it to the queue. This is
    // a little wasteful as we'll finish processing the current loop as well,
    // but it is the best we can do in the old PM.
    if (CurrentLoopValid)
      LPM.addLoop(*L);
    else
      LPM.markLoopAsDeleted(*L);
  };

  bool Changed =
      unswitchLoop(*L, DT, LI, AC, TTI, NonTrivial, NonTrivialUnswitchCB);

  // If anything was unswitched, also clear any cached information about this
  // loop.
  LPM.deleteSimpleAnalysisLoop(L);

  // Historically this pass has had issues with the dominator tree so verify it
  // in asserts builds.
  assert(DT.verify(DominatorTree::VerificationLevel::Fast));

  return Changed;
}

char SimpleLoopUnswitchLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(SimpleLoopUnswitchLegacyPass, "simple-loop-unswitch",
                      "Simple unswitch loops", false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(SimpleLoopUnswitchLegacyPass, "simple-loop-unswitch",
                    "Simple unswitch loops", false, false)

Pass *llvm::createSimpleLoopUnswitchLegacyPass(bool NonTrivial) {
  return new SimpleLoopUnswitchLegacyPass(NonTrivial);
}
