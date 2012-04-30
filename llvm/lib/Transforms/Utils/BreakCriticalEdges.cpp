//===- BreakCriticalEdges.cpp - Critical Edge Elimination Pass ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// BreakCriticalEdges pass - Break all of the critical edges in the CFG by
// inserting a dummy basic block.  This pass may be "required" by passes that
// cannot deal with critical edges.  For this usage, the structure type is
// forward declared.  This pass obviously invalidates the CFG, but can update
// dominator trees.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "break-crit-edges"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ProfileInfo.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Type.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumBroken, "Number of blocks inserted");

namespace {
  struct BreakCriticalEdges : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    BreakCriticalEdges() : FunctionPass(ID) {
      initializeBreakCriticalEdgesPass(*PassRegistry::getPassRegistry());
    }

    virtual bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addPreserved<DominatorTree>();
      AU.addPreserved<LoopInfo>();
      AU.addPreserved<ProfileInfo>();

      // No loop canonicalization guarantees are broken by this pass.
      AU.addPreservedID(LoopSimplifyID);
    }
  };
}

char BreakCriticalEdges::ID = 0;
INITIALIZE_PASS(BreakCriticalEdges, "break-crit-edges",
                "Break critical edges in CFG", false, false)

// Publicly exposed interface to pass...
char &llvm::BreakCriticalEdgesID = BreakCriticalEdges::ID;
FunctionPass *llvm::createBreakCriticalEdgesPass() {
  return new BreakCriticalEdges();
}

// runOnFunction - Loop over all of the edges in the CFG, breaking critical
// edges as they are found.
//
bool BreakCriticalEdges::runOnFunction(Function &F) {
  bool Changed = false;
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I) {
    TerminatorInst *TI = I->getTerminator();
    if (TI->getNumSuccessors() > 1 && !isa<IndirectBrInst>(TI))
      for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i)
        if (SplitCriticalEdge(TI, i, this)) {
          ++NumBroken;
          Changed = true;
        }
  }

  return Changed;
}

//===----------------------------------------------------------------------===//
//    Implementation of the external critical edge manipulation functions
//===----------------------------------------------------------------------===//

// isCriticalEdge - Return true if the specified edge is a critical edge.
// Critical edges are edges from a block with multiple successors to a block
// with multiple predecessors.
//
bool llvm::isCriticalEdge(const TerminatorInst *TI, unsigned SuccNum,
                          bool AllowIdenticalEdges) {
  assert(SuccNum < TI->getNumSuccessors() && "Illegal edge specification!");
  if (TI->getNumSuccessors() == 1) return false;

  const BasicBlock *Dest = TI->getSuccessor(SuccNum);
  const_pred_iterator I = pred_begin(Dest), E = pred_end(Dest);

  // If there is more than one predecessor, this is a critical edge...
  assert(I != E && "No preds, but we have an edge to the block?");
  const BasicBlock *FirstPred = *I;
  ++I;        // Skip one edge due to the incoming arc from TI.
  if (!AllowIdenticalEdges)
    return I != E;

  // If AllowIdenticalEdges is true, then we allow this edge to be considered
  // non-critical iff all preds come from TI's block.
  while (I != E) {
    const BasicBlock *P = *I;
    if (P != FirstPred)
      return true;
    // Note: leave this as is until no one ever compiles with either gcc 4.0.1
    // or Xcode 2. This seems to work around the pred_iterator assert in PR 2207
    E = pred_end(P);
    ++I;
  }
  return false;
}

/// createPHIsForSplitLoopExit - When a loop exit edge is split, LCSSA form
/// may require new PHIs in the new exit block. This function inserts the
/// new PHIs, as needed. Preds is a list of preds inside the loop, SplitBB
/// is the new loop exit block, and DestBB is the old loop exit, now the
/// successor of SplitBB.
static void createPHIsForSplitLoopExit(SmallVectorImpl<BasicBlock *> &Preds,
                                       BasicBlock *SplitBB,
                                       BasicBlock *DestBB) {
  // SplitBB shouldn't have anything non-trivial in it yet.
  assert((SplitBB->getFirstNonPHI() == SplitBB->getTerminator() ||
          SplitBB->isLandingPad()) && "SplitBB has non-PHI nodes!");

  // For each PHI in the destination block.
  for (BasicBlock::iterator I = DestBB->begin();
       PHINode *PN = dyn_cast<PHINode>(I); ++I) {
    unsigned Idx = PN->getBasicBlockIndex(SplitBB);
    Value *V = PN->getIncomingValue(Idx);

    // If the input is a PHI which already satisfies LCSSA, don't create
    // a new one.
    if (const PHINode *VP = dyn_cast<PHINode>(V))
      if (VP->getParent() == SplitBB)
        continue;

    // Otherwise a new PHI is needed. Create one and populate it.
    PHINode *NewPN =
      PHINode::Create(PN->getType(), Preds.size(), "split",
                      SplitBB->isLandingPad() ?
                      SplitBB->begin() : SplitBB->getTerminator());
    for (unsigned i = 0, e = Preds.size(); i != e; ++i)
      NewPN->addIncoming(V, Preds[i]);

    // Update the original PHI.
    PN->setIncomingValue(Idx, NewPN);
  }
}

/// SplitCriticalEdge - If this edge is a critical edge, insert a new node to
/// split the critical edge.  This will update DominatorTree information if it
/// is available, thus calling this pass will not invalidate either of them.
/// This returns the new block if the edge was split, null otherwise.
///
/// If MergeIdenticalEdges is true (not the default), *all* edges from TI to the
/// specified successor will be merged into the same critical edge block.
/// This is most commonly interesting with switch instructions, which may
/// have many edges to any one destination.  This ensures that all edges to that
/// dest go to one block instead of each going to a different block, but isn't
/// the standard definition of a "critical edge".
///
/// It is invalid to call this function on a critical edge that starts at an
/// IndirectBrInst.  Splitting these edges will almost always create an invalid
/// program because the address of the new block won't be the one that is jumped
/// to.
///
BasicBlock *llvm::SplitCriticalEdge(TerminatorInst *TI, unsigned SuccNum,
                                    Pass *P, bool MergeIdenticalEdges,
                                    bool DontDeleteUselessPhis,
                                    bool SplitLandingPads) {
  if (!isCriticalEdge(TI, SuccNum, MergeIdenticalEdges)) return 0;

  assert(!isa<IndirectBrInst>(TI) &&
         "Cannot split critical edge from IndirectBrInst");

  BasicBlock *TIBB = TI->getParent();
  BasicBlock *DestBB = TI->getSuccessor(SuccNum);

  // Splitting the critical edge to a landing pad block is non-trivial. Don't do
  // it in this generic function.
  if (DestBB->isLandingPad()) return 0;

  // Create a new basic block, linking it into the CFG.
  BasicBlock *NewBB = BasicBlock::Create(TI->getContext(),
                      TIBB->getName() + "." + DestBB->getName() + "_crit_edge");
  // Create our unconditional branch.
  BranchInst *NewBI = BranchInst::Create(DestBB, NewBB);
  NewBI->setDebugLoc(TI->getDebugLoc());

  // Branch to the new block, breaking the edge.
  TI->setSuccessor(SuccNum, NewBB);

  // Insert the block into the function... right after the block TI lives in.
  Function &F = *TIBB->getParent();
  Function::iterator FBBI = TIBB;
  F.getBasicBlockList().insert(++FBBI, NewBB);

  // If there are any PHI nodes in DestBB, we need to update them so that they
  // merge incoming values from NewBB instead of from TIBB.
  {
    unsigned BBIdx = 0;
    for (BasicBlock::iterator I = DestBB->begin(); isa<PHINode>(I); ++I) {
      // We no longer enter through TIBB, now we come in through NewBB.
      // Revector exactly one entry in the PHI node that used to come from
      // TIBB to come from NewBB.
      PHINode *PN = cast<PHINode>(I);

      // Reuse the previous value of BBIdx if it lines up.  In cases where we
      // have multiple phi nodes with *lots* of predecessors, this is a speed
      // win because we don't have to scan the PHI looking for TIBB.  This
      // happens because the BB list of PHI nodes are usually in the same
      // order.
      if (PN->getIncomingBlock(BBIdx) != TIBB)
        BBIdx = PN->getBasicBlockIndex(TIBB);
      PN->setIncomingBlock(BBIdx, NewBB);
    }
  }

  // If there are any other edges from TIBB to DestBB, update those to go
  // through the split block, making those edges non-critical as well (and
  // reducing the number of phi entries in the DestBB if relevant).
  if (MergeIdenticalEdges) {
    for (unsigned i = SuccNum+1, e = TI->getNumSuccessors(); i != e; ++i) {
      if (TI->getSuccessor(i) != DestBB) continue;

      // Remove an entry for TIBB from DestBB phi nodes.
      DestBB->removePredecessor(TIBB, DontDeleteUselessPhis);

      // We found another edge to DestBB, go to NewBB instead.
      TI->setSuccessor(i, NewBB);
    }
  }



  // If we don't have a pass object, we can't update anything...
  if (P == 0) return NewBB;

  DominatorTree *DT = P->getAnalysisIfAvailable<DominatorTree>();
  LoopInfo *LI = P->getAnalysisIfAvailable<LoopInfo>();
  ProfileInfo *PI = P->getAnalysisIfAvailable<ProfileInfo>();

  // If we have nothing to update, just return.
  if (DT == 0 && LI == 0 && PI == 0)
    return NewBB;

  // Now update analysis information.  Since the only predecessor of NewBB is
  // the TIBB, TIBB clearly dominates NewBB.  TIBB usually doesn't dominate
  // anything, as there are other successors of DestBB.  However, if all other
  // predecessors of DestBB are already dominated by DestBB (e.g. DestBB is a
  // loop header) then NewBB dominates DestBB.
  SmallVector<BasicBlock*, 8> OtherPreds;

  // If there is a PHI in the block, loop over predecessors with it, which is
  // faster than iterating pred_begin/end.
  if (PHINode *PN = dyn_cast<PHINode>(DestBB->begin())) {
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
      if (PN->getIncomingBlock(i) != NewBB)
        OtherPreds.push_back(PN->getIncomingBlock(i));
  } else {
    for (pred_iterator I = pred_begin(DestBB), E = pred_end(DestBB);
         I != E; ++I) {
      BasicBlock *P = *I;
      if (P != NewBB)
        OtherPreds.push_back(P);
    }
  }

  bool NewBBDominatesDestBB = true;

  // Should we update DominatorTree information?
  if (DT) {
    DomTreeNode *TINode = DT->getNode(TIBB);

    // The new block is not the immediate dominator for any other nodes, but
    // TINode is the immediate dominator for the new node.
    //
    if (TINode) {       // Don't break unreachable code!
      DomTreeNode *NewBBNode = DT->addNewBlock(NewBB, TIBB);
      DomTreeNode *DestBBNode = 0;

      // If NewBBDominatesDestBB hasn't been computed yet, do so with DT.
      if (!OtherPreds.empty()) {
        DestBBNode = DT->getNode(DestBB);
        while (!OtherPreds.empty() && NewBBDominatesDestBB) {
          if (DomTreeNode *OPNode = DT->getNode(OtherPreds.back()))
            NewBBDominatesDestBB = DT->dominates(DestBBNode, OPNode);
          OtherPreds.pop_back();
        }
        OtherPreds.clear();
      }

      // If NewBBDominatesDestBB, then NewBB dominates DestBB, otherwise it
      // doesn't dominate anything.
      if (NewBBDominatesDestBB) {
        if (!DestBBNode) DestBBNode = DT->getNode(DestBB);
        DT->changeImmediateDominator(DestBBNode, NewBBNode);
      }
    }
  }

  // Update LoopInfo if it is around.
  if (LI) {
    if (Loop *TIL = LI->getLoopFor(TIBB)) {
      // If one or the other blocks were not in a loop, the new block is not
      // either, and thus LI doesn't need to be updated.
      if (Loop *DestLoop = LI->getLoopFor(DestBB)) {
        if (TIL == DestLoop) {
          // Both in the same loop, the NewBB joins loop.
          DestLoop->addBasicBlockToLoop(NewBB, LI->getBase());
        } else if (TIL->contains(DestLoop)) {
          // Edge from an outer loop to an inner loop.  Add to the outer loop.
          TIL->addBasicBlockToLoop(NewBB, LI->getBase());
        } else if (DestLoop->contains(TIL)) {
          // Edge from an inner loop to an outer loop.  Add to the outer loop.
          DestLoop->addBasicBlockToLoop(NewBB, LI->getBase());
        } else {
          // Edge from two loops with no containment relation.  Because these
          // are natural loops, we know that the destination block must be the
          // header of its loop (adding a branch into a loop elsewhere would
          // create an irreducible loop).
          assert(DestLoop->getHeader() == DestBB &&
                 "Should not create irreducible loops!");
          if (Loop *P = DestLoop->getParentLoop())
            P->addBasicBlockToLoop(NewBB, LI->getBase());
        }
      }
      // If TIBB is in a loop and DestBB is outside of that loop, split the
      // other exit blocks of the loop that also have predecessors outside
      // the loop, to maintain a LoopSimplify guarantee.
      if (!TIL->contains(DestBB) &&
          P->mustPreserveAnalysisID(LoopSimplifyID)) {
        assert(!TIL->contains(NewBB) &&
               "Split point for loop exit is contained in loop!");

        // Update LCSSA form in the newly created exit block.
        if (P->mustPreserveAnalysisID(LCSSAID)) {
          SmallVector<BasicBlock *, 1> OrigPred;
          OrigPred.push_back(TIBB);
          createPHIsForSplitLoopExit(OrigPred, NewBB, DestBB);
        }

        // For each unique exit block...
        // FIXME: This code is functionally equivalent to the corresponding
        // loop in LoopSimplify.
        SmallVector<BasicBlock *, 4> ExitBlocks;
        TIL->getExitBlocks(ExitBlocks);
        for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i) {
          // Collect all the preds that are inside the loop, and note
          // whether there are any preds outside the loop.
          SmallVector<BasicBlock *, 4> Preds;
          bool HasPredOutsideOfLoop = false;
          BasicBlock *Exit = ExitBlocks[i];
          for (pred_iterator I = pred_begin(Exit), E = pred_end(Exit);
               I != E; ++I) {
            BasicBlock *P = *I;
            if (TIL->contains(P)) {
              if (isa<IndirectBrInst>(P->getTerminator())) {
                Preds.clear();
                break;
              }
              Preds.push_back(P);
            } else {
              HasPredOutsideOfLoop = true;
            }
          }
          // If there are any preds not in the loop, we'll need to split
          // the edges. The Preds.empty() check is needed because a block
          // may appear multiple times in the list. We can't use
          // getUniqueExitBlocks above because that depends on LoopSimplify
          // form, which we're in the process of restoring!
          if (!Preds.empty() && HasPredOutsideOfLoop) {
            if (!Exit->isLandingPad()) {
              BasicBlock *NewExitBB =
                SplitBlockPredecessors(Exit, Preds, "split", P);
              if (P->mustPreserveAnalysisID(LCSSAID))
                createPHIsForSplitLoopExit(Preds, NewExitBB, Exit);
            } else if (SplitLandingPads) {
              SmallVector<BasicBlock*, 8> NewBBs;
              SplitLandingPadPredecessors(Exit, Preds,
                                          ".split1", ".split2",
                                          P, NewBBs);
              if (P->mustPreserveAnalysisID(LCSSAID))
                createPHIsForSplitLoopExit(Preds, NewBBs[0], Exit);
            }
          }
        }
      }
      // LCSSA form was updated above for the case where LoopSimplify is
      // available, which means that all predecessors of loop exit blocks
      // are within the loop. Without LoopSimplify form, it would be
      // necessary to insert a new phi.
      assert((!P->mustPreserveAnalysisID(LCSSAID) ||
              P->mustPreserveAnalysisID(LoopSimplifyID)) &&
             "SplitCriticalEdge doesn't know how to update LCCSA form "
             "without LoopSimplify!");
    }
  }

  // Update ProfileInfo if it is around.
  if (PI)
    PI->splitEdge(TIBB, DestBB, NewBB, MergeIdenticalEdges);

  return NewBB;
}
