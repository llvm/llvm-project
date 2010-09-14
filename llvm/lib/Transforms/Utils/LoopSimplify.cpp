//===- LoopSimplify.cpp - Loop Canonicalization Pass ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass performs several transformations to transform natural loops into a
// simpler form, which makes subsequent analyses and transformations simpler and
// more effective.
//
// Loop pre-header insertion guarantees that there is a single, non-critical
// entry edge from outside of the loop to the loop header.  This simplifies a
// number of analyses and transformations, such as LICM.
//
// Loop exit-block insertion guarantees that all exit blocks from the loop
// (blocks which are outside of the loop that have predecessors inside of the
// loop) only have predecessors from inside of the loop (and are thus dominated
// by the loop header).  This simplifies transformations such as store-sinking
// that are built into LICM.
//
// This pass also guarantees that loops will have exactly one backedge.
//
// Indirectbr instructions introduce several complications. If the loop
// contains or is entered by an indirectbr instruction, it may not be possible
// to transform the loop and make these guarantees. Client code should check
// that these conditions are true before relying on them.
//
// Note that the simplifycfg pass will clean up blocks which are split out but
// end up being unnecessary, so usage of this pass should not pessimize
// generated code.
//
// This pass obviously modifies the CFG, but updates loop information and
// dominator information.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "loopsimplify"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Function.h"
#include "llvm/LLVMContext.h"
#include "llvm/Type.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/DepthFirstIterator.h"
using namespace llvm;

STATISTIC(NumInserted, "Number of pre-header or exit blocks inserted");
STATISTIC(NumNested  , "Number of nested loops split out");

namespace {
  struct LoopSimplify : public LoopPass {
    static char ID; // Pass identification, replacement for typeid
    LoopSimplify() : LoopPass(ID) {}

    // AA - If we have an alias analysis object to update, this is it, otherwise
    // this is null.
    AliasAnalysis *AA;
    LoopInfo *LI;
    DominatorTree *DT;
    ScalarEvolution *SE;
    Loop *L;
    virtual bool runOnLoop(Loop *L, LPPassManager &LPM);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      // We need loop information to identify the loops...
      AU.addRequired<DominatorTree>();
      AU.addPreserved<DominatorTree>();

      AU.addRequired<LoopInfo>();
      AU.addPreserved<LoopInfo>();

      AU.addPreserved<AliasAnalysis>();
      AU.addPreserved<ScalarEvolution>();
      AU.addPreservedID(BreakCriticalEdgesID);  // No critical edges added.
      AU.addPreserved<DominanceFrontier>();
      AU.addPreservedID(LCSSAID);
    }

    /// verifyAnalysis() - Verify LoopSimplifyForm's guarantees.
    void verifyAnalysis() const;

  private:
    bool ProcessLoop(Loop *L, LPPassManager &LPM);
    BasicBlock *RewriteLoopExitBlock(Loop *L, BasicBlock *Exit);
    BasicBlock *InsertPreheaderForLoop(Loop *L);
    Loop *SeparateNestedLoop(Loop *L, LPPassManager &LPM);
    BasicBlock *InsertUniqueBackedgeBlock(Loop *L, BasicBlock *Preheader);
    void PlaceSplitBlockCarefully(BasicBlock *NewBB,
                                  SmallVectorImpl<BasicBlock*> &SplitPreds,
                                  Loop *L);
  };
}

char LoopSimplify::ID = 0;
INITIALIZE_PASS(LoopSimplify, "loopsimplify",
                "Canonicalize natural loops", true, false);

// Publically exposed interface to pass...
char &llvm::LoopSimplifyID = LoopSimplify::ID;
Pass *llvm::createLoopSimplifyPass() { return new LoopSimplify(); }

/// runOnLoop - Run down all loops in the CFG (recursively, but we could do
/// it in any convenient order) inserting preheaders...
///
bool LoopSimplify::runOnLoop(Loop *l, LPPassManager &LPM) {
  L = l;
  bool Changed = false;
  LI = &getAnalysis<LoopInfo>();
  AA = getAnalysisIfAvailable<AliasAnalysis>();
  DT = &getAnalysis<DominatorTree>();
  SE = getAnalysisIfAvailable<ScalarEvolution>();

  Changed |= ProcessLoop(L, LPM);

  return Changed;
}

/// ProcessLoop - Walk the loop structure in depth first order, ensuring that
/// all loops have preheaders.
///
bool LoopSimplify::ProcessLoop(Loop *L, LPPassManager &LPM) {
  bool Changed = false;
ReprocessLoop:

  // Check to see that no blocks (other than the header) in this loop have
  // predecessors that are not in the loop.  This is not valid for natural
  // loops, but can occur if the blocks are unreachable.  Since they are
  // unreachable we can just shamelessly delete those CFG edges!
  for (Loop::block_iterator BB = L->block_begin(), E = L->block_end();
       BB != E; ++BB) {
    if (*BB == L->getHeader()) continue;

    SmallPtrSet<BasicBlock*, 4> BadPreds;
    for (pred_iterator PI = pred_begin(*BB),
         PE = pred_end(*BB); PI != PE; ++PI) {
      BasicBlock *P = *PI;
      if (!L->contains(P))
        BadPreds.insert(P);
    }

    // Delete each unique out-of-loop (and thus dead) predecessor.
    for (SmallPtrSet<BasicBlock*, 4>::iterator I = BadPreds.begin(),
         E = BadPreds.end(); I != E; ++I) {

      DEBUG(dbgs() << "LoopSimplify: Deleting edge from dead predecessor ";
            WriteAsOperand(dbgs(), *I, false);
            dbgs() << "\n");

      // Inform each successor of each dead pred.
      for (succ_iterator SI = succ_begin(*I), SE = succ_end(*I); SI != SE; ++SI)
        (*SI)->removePredecessor(*I);
      // Zap the dead pred's terminator and replace it with unreachable.
      TerminatorInst *TI = (*I)->getTerminator();
       TI->replaceAllUsesWith(UndefValue::get(TI->getType()));
      (*I)->getTerminator()->eraseFromParent();
      new UnreachableInst((*I)->getContext(), *I);
      Changed = true;
    }
  }

  // If there are exiting blocks with branches on undef, resolve the undef in
  // the direction which will exit the loop. This will help simplify loop
  // trip count computations.
  SmallVector<BasicBlock*, 8> ExitingBlocks;
  L->getExitingBlocks(ExitingBlocks);
  for (SmallVectorImpl<BasicBlock *>::iterator I = ExitingBlocks.begin(),
       E = ExitingBlocks.end(); I != E; ++I)
    if (BranchInst *BI = dyn_cast<BranchInst>((*I)->getTerminator()))
      if (BI->isConditional()) {
        if (UndefValue *Cond = dyn_cast<UndefValue>(BI->getCondition())) {

          DEBUG(dbgs() << "LoopSimplify: Resolving \"br i1 undef\" to exit in ";
                WriteAsOperand(dbgs(), *I, false);
                dbgs() << "\n");

          BI->setCondition(ConstantInt::get(Cond->getType(),
                                            !L->contains(BI->getSuccessor(0))));
          Changed = true;
        }
      }

  // Does the loop already have a preheader?  If so, don't insert one.
  BasicBlock *Preheader = L->getLoopPreheader();
  if (!Preheader) {
    Preheader = InsertPreheaderForLoop(L);
    if (Preheader) {
      ++NumInserted;
      Changed = true;
    }
  }

  // Next, check to make sure that all exit nodes of the loop only have
  // predecessors that are inside of the loop.  This check guarantees that the
  // loop preheader/header will dominate the exit blocks.  If the exit block has
  // predecessors from outside of the loop, split the edge now.
  SmallVector<BasicBlock*, 8> ExitBlocks;
  L->getExitBlocks(ExitBlocks);
    
  SmallSetVector<BasicBlock *, 8> ExitBlockSet(ExitBlocks.begin(),
                                               ExitBlocks.end());
  for (SmallSetVector<BasicBlock *, 8>::iterator I = ExitBlockSet.begin(),
         E = ExitBlockSet.end(); I != E; ++I) {
    BasicBlock *ExitBlock = *I;
    for (pred_iterator PI = pred_begin(ExitBlock), PE = pred_end(ExitBlock);
         PI != PE; ++PI)
      // Must be exactly this loop: no subloops, parent loops, or non-loop preds
      // allowed.
      if (!L->contains(*PI)) {
        if (RewriteLoopExitBlock(L, ExitBlock)) {
          ++NumInserted;
          Changed = true;
        }
        break;
      }
  }

  // If the header has more than two predecessors at this point (from the
  // preheader and from multiple backedges), we must adjust the loop.
  BasicBlock *LoopLatch = L->getLoopLatch();
  if (!LoopLatch) {
    // If this is really a nested loop, rip it out into a child loop.  Don't do
    // this for loops with a giant number of backedges, just factor them into a
    // common backedge instead.
    if (L->getNumBackEdges() < 8) {
      if (SeparateNestedLoop(L, LPM)) {
        ++NumNested;
        // This is a big restructuring change, reprocess the whole loop.
        Changed = true;
        // GCC doesn't tail recursion eliminate this.
        goto ReprocessLoop;
      }
    }

    // If we either couldn't, or didn't want to, identify nesting of the loops,
    // insert a new block that all backedges target, then make it jump to the
    // loop header.
    LoopLatch = InsertUniqueBackedgeBlock(L, Preheader);
    if (LoopLatch) {
      ++NumInserted;
      Changed = true;
    }
  }

  // Scan over the PHI nodes in the loop header.  Since they now have only two
  // incoming values (the loop is canonicalized), we may have simplified the PHI
  // down to 'X = phi [X, Y]', which should be replaced with 'Y'.
  PHINode *PN;
  for (BasicBlock::iterator I = L->getHeader()->begin();
       (PN = dyn_cast<PHINode>(I++)); )
    if (Value *V = PN->hasConstantValue(DT)) {
      if (AA) AA->deleteValue(PN);
      PN->replaceAllUsesWith(V);
      PN->eraseFromParent();
    }

  // If this loop has multiple exits and the exits all go to the same
  // block, attempt to merge the exits. This helps several passes, such
  // as LoopRotation, which do not support loops with multiple exits.
  // SimplifyCFG also does this (and this code uses the same utility
  // function), however this code is loop-aware, where SimplifyCFG is
  // not. That gives it the advantage of being able to hoist
  // loop-invariant instructions out of the way to open up more
  // opportunities, and the disadvantage of having the responsibility
  // to preserve dominator information.
  bool UniqueExit = true;
  if (!ExitBlocks.empty())
    for (unsigned i = 1, e = ExitBlocks.size(); i != e; ++i)
      if (ExitBlocks[i] != ExitBlocks[0]) {
        UniqueExit = false;
        break;
      }
  if (UniqueExit) {
    for (unsigned i = 0, e = ExitingBlocks.size(); i != e; ++i) {
      BasicBlock *ExitingBlock = ExitingBlocks[i];
      if (!ExitingBlock->getSinglePredecessor()) continue;
      BranchInst *BI = dyn_cast<BranchInst>(ExitingBlock->getTerminator());
      if (!BI || !BI->isConditional()) continue;
      CmpInst *CI = dyn_cast<CmpInst>(BI->getCondition());
      if (!CI || CI->getParent() != ExitingBlock) continue;

      // Attempt to hoist out all instructions except for the
      // comparison and the branch.
      bool AllInvariant = true;
      for (BasicBlock::iterator I = ExitingBlock->begin(); &*I != BI; ) {
        Instruction *Inst = I++;
        // Skip debug info intrinsics.
        if (isa<DbgInfoIntrinsic>(Inst))
          continue;
        if (Inst == CI)
          continue;
        if (!L->makeLoopInvariant(Inst, Changed,
                                  Preheader ? Preheader->getTerminator() : 0)) {
          AllInvariant = false;
          break;
        }
      }
      if (!AllInvariant) continue;

      // The block has now been cleared of all instructions except for
      // a comparison and a conditional branch. SimplifyCFG may be able
      // to fold it now.
      if (!FoldBranchToCommonDest(BI)) continue;

      // Success. The block is now dead, so remove it from the loop,
      // update the dominator tree and dominance frontier, and delete it.

      DEBUG(dbgs() << "LoopSimplify: Eliminating exiting block ";
            WriteAsOperand(dbgs(), ExitingBlock, false);
            dbgs() << "\n");

      assert(pred_begin(ExitingBlock) == pred_end(ExitingBlock));
      Changed = true;
      LI->removeBlock(ExitingBlock);

      DominanceFrontier *DF = getAnalysisIfAvailable<DominanceFrontier>();
      DomTreeNode *Node = DT->getNode(ExitingBlock);
      const std::vector<DomTreeNodeBase<BasicBlock> *> &Children =
        Node->getChildren();
      while (!Children.empty()) {
        DomTreeNode *Child = Children.front();
        DT->changeImmediateDominator(Child, Node->getIDom());
        if (DF) DF->changeImmediateDominator(Child->getBlock(),
                                             Node->getIDom()->getBlock(),
                                             DT);
      }
      DT->eraseNode(ExitingBlock);
      if (DF) DF->removeBlock(ExitingBlock);

      BI->getSuccessor(0)->removePredecessor(ExitingBlock);
      BI->getSuccessor(1)->removePredecessor(ExitingBlock);
      ExitingBlock->eraseFromParent();
    }
  }

  return Changed;
}

/// InsertPreheaderForLoop - Once we discover that a loop doesn't have a
/// preheader, this method is called to insert one.  This method has two phases:
/// preheader insertion and analysis updating.
///
BasicBlock *LoopSimplify::InsertPreheaderForLoop(Loop *L) {
  BasicBlock *Header = L->getHeader();

  // Compute the set of predecessors of the loop that are not in the loop.
  SmallVector<BasicBlock*, 8> OutsideBlocks;
  for (pred_iterator PI = pred_begin(Header), PE = pred_end(Header);
       PI != PE; ++PI) {
    BasicBlock *P = *PI;
    if (!L->contains(P)) {         // Coming in from outside the loop?
      // If the loop is branched to from an indirect branch, we won't
      // be able to fully transform the loop, because it prohibits
      // edge splitting.
      if (isa<IndirectBrInst>(P->getTerminator())) return 0;

      // Keep track of it.
      OutsideBlocks.push_back(P);
    }
  }

  // Split out the loop pre-header.
  BasicBlock *NewBB =
    SplitBlockPredecessors(Header, &OutsideBlocks[0], OutsideBlocks.size(),
                           ".preheader", this);

  DEBUG(dbgs() << "LoopSimplify: Creating pre-header ";
        WriteAsOperand(dbgs(), NewBB, false);
        dbgs() << "\n");

  // Make sure that NewBB is put someplace intelligent, which doesn't mess up
  // code layout too horribly.
  PlaceSplitBlockCarefully(NewBB, OutsideBlocks, L);

  return NewBB;
}

/// RewriteLoopExitBlock - Ensure that the loop preheader dominates all exit
/// blocks.  This method is used to split exit blocks that have predecessors
/// outside of the loop.
BasicBlock *LoopSimplify::RewriteLoopExitBlock(Loop *L, BasicBlock *Exit) {
  SmallVector<BasicBlock*, 8> LoopBlocks;
  for (pred_iterator I = pred_begin(Exit), E = pred_end(Exit); I != E; ++I) {
    BasicBlock *P = *I;
    if (L->contains(P)) {
      // Don't do this if the loop is exited via an indirect branch.
      if (isa<IndirectBrInst>(P->getTerminator())) return 0;

      LoopBlocks.push_back(P);
    }
  }

  assert(!LoopBlocks.empty() && "No edges coming in from outside the loop?");
  BasicBlock *NewBB = SplitBlockPredecessors(Exit, &LoopBlocks[0], 
                                             LoopBlocks.size(), ".loopexit",
                                             this);

  DEBUG(dbgs() << "LoopSimplify: Creating dedicated exit block ";
        WriteAsOperand(dbgs(), NewBB, false);
        dbgs() << "\n");

  return NewBB;
}

/// AddBlockAndPredsToSet - Add the specified block, and all of its
/// predecessors, to the specified set, if it's not already in there.  Stop
/// predecessor traversal when we reach StopBlock.
static void AddBlockAndPredsToSet(BasicBlock *InputBB, BasicBlock *StopBlock,
                                  std::set<BasicBlock*> &Blocks) {
  std::vector<BasicBlock *> WorkList;
  WorkList.push_back(InputBB);
  do {
    BasicBlock *BB = WorkList.back(); WorkList.pop_back();
    if (Blocks.insert(BB).second && BB != StopBlock)
      // If BB is not already processed and it is not a stop block then
      // insert its predecessor in the work list
      for (pred_iterator I = pred_begin(BB), E = pred_end(BB); I != E; ++I) {
        BasicBlock *WBB = *I;
        WorkList.push_back(WBB);
      }
  } while(!WorkList.empty());
}

/// FindPHIToPartitionLoops - The first part of loop-nestification is to find a
/// PHI node that tells us how to partition the loops.
static PHINode *FindPHIToPartitionLoops(Loop *L, DominatorTree *DT,
                                        AliasAnalysis *AA) {
  for (BasicBlock::iterator I = L->getHeader()->begin(); isa<PHINode>(I); ) {
    PHINode *PN = cast<PHINode>(I);
    ++I;
    if (Value *V = PN->hasConstantValue(DT)) {
      // This is a degenerate PHI already, don't modify it!
      PN->replaceAllUsesWith(V);
      if (AA) AA->deleteValue(PN);
      PN->eraseFromParent();
      continue;
    }

    // Scan this PHI node looking for a use of the PHI node by itself.
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
      if (PN->getIncomingValue(i) == PN &&
          L->contains(PN->getIncomingBlock(i)))
        // We found something tasty to remove.
        return PN;
  }
  return 0;
}

// PlaceSplitBlockCarefully - If the block isn't already, move the new block to
// right after some 'outside block' block.  This prevents the preheader from
// being placed inside the loop body, e.g. when the loop hasn't been rotated.
void LoopSimplify::PlaceSplitBlockCarefully(BasicBlock *NewBB,
                                       SmallVectorImpl<BasicBlock*> &SplitPreds,
                                            Loop *L) {
  // Check to see if NewBB is already well placed.
  Function::iterator BBI = NewBB; --BBI;
  for (unsigned i = 0, e = SplitPreds.size(); i != e; ++i) {
    if (&*BBI == SplitPreds[i])
      return;
  }
  
  // If it isn't already after an outside block, move it after one.  This is
  // always good as it makes the uncond branch from the outside block into a
  // fall-through.
  
  // Figure out *which* outside block to put this after.  Prefer an outside
  // block that neighbors a BB actually in the loop.
  BasicBlock *FoundBB = 0;
  for (unsigned i = 0, e = SplitPreds.size(); i != e; ++i) {
    Function::iterator BBI = SplitPreds[i];
    if (++BBI != NewBB->getParent()->end() && 
        L->contains(BBI)) {
      FoundBB = SplitPreds[i];
      break;
    }
  }
  
  // If our heuristic for a *good* bb to place this after doesn't find
  // anything, just pick something.  It's likely better than leaving it within
  // the loop.
  if (!FoundBB)
    FoundBB = SplitPreds[0];
  NewBB->moveAfter(FoundBB);
}


/// SeparateNestedLoop - If this loop has multiple backedges, try to pull one of
/// them out into a nested loop.  This is important for code that looks like
/// this:
///
///  Loop:
///     ...
///     br cond, Loop, Next
///     ...
///     br cond2, Loop, Out
///
/// To identify this common case, we look at the PHI nodes in the header of the
/// loop.  PHI nodes with unchanging values on one backedge correspond to values
/// that change in the "outer" loop, but not in the "inner" loop.
///
/// If we are able to separate out a loop, return the new outer loop that was
/// created.
///
Loop *LoopSimplify::SeparateNestedLoop(Loop *L, LPPassManager &LPM) {
  PHINode *PN = FindPHIToPartitionLoops(L, DT, AA);
  if (PN == 0) return 0;  // No known way to partition.

  // Pull out all predecessors that have varying values in the loop.  This
  // handles the case when a PHI node has multiple instances of itself as
  // arguments.
  SmallVector<BasicBlock*, 8> OuterLoopPreds;
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
    if (PN->getIncomingValue(i) != PN ||
        !L->contains(PN->getIncomingBlock(i))) {
      // We can't split indirectbr edges.
      if (isa<IndirectBrInst>(PN->getIncomingBlock(i)->getTerminator()))
        return 0;

      OuterLoopPreds.push_back(PN->getIncomingBlock(i));
    }

  DEBUG(dbgs() << "LoopSimplify: Splitting out a new outer loop\n");

  // If ScalarEvolution is around and knows anything about values in
  // this loop, tell it to forget them, because we're about to
  // substantially change it.
  if (SE)
    SE->forgetLoop(L);

  BasicBlock *Header = L->getHeader();
  BasicBlock *NewBB = SplitBlockPredecessors(Header, &OuterLoopPreds[0],
                                             OuterLoopPreds.size(),
                                             ".outer", this);

  // Make sure that NewBB is put someplace intelligent, which doesn't mess up
  // code layout too horribly.
  PlaceSplitBlockCarefully(NewBB, OuterLoopPreds, L);
  
  // Create the new outer loop.
  Loop *NewOuter = new Loop();

  // Change the parent loop to use the outer loop as its child now.
  if (Loop *Parent = L->getParentLoop())
    Parent->replaceChildLoopWith(L, NewOuter);
  else
    LI->changeTopLevelLoop(L, NewOuter);

  // L is now a subloop of our outer loop.
  NewOuter->addChildLoop(L);

  // Add the new loop to the pass manager queue.
  LPM.insertLoopIntoQueue(NewOuter);

  for (Loop::block_iterator I = L->block_begin(), E = L->block_end();
       I != E; ++I)
    NewOuter->addBlockEntry(*I);

  // Now reset the header in L, which had been moved by
  // SplitBlockPredecessors for the outer loop.
  L->moveToHeader(Header);

  // Determine which blocks should stay in L and which should be moved out to
  // the Outer loop now.
  std::set<BasicBlock*> BlocksInL;
  for (pred_iterator PI=pred_begin(Header), E = pred_end(Header); PI!=E; ++PI) {
    BasicBlock *P = *PI;
    if (DT->dominates(Header, P))
      AddBlockAndPredsToSet(P, Header, BlocksInL);
  }

  // Scan all of the loop children of L, moving them to OuterLoop if they are
  // not part of the inner loop.
  const std::vector<Loop*> &SubLoops = L->getSubLoops();
  for (size_t I = 0; I != SubLoops.size(); )
    if (BlocksInL.count(SubLoops[I]->getHeader()))
      ++I;   // Loop remains in L
    else
      NewOuter->addChildLoop(L->removeChildLoop(SubLoops.begin() + I));

  // Now that we know which blocks are in L and which need to be moved to
  // OuterLoop, move any blocks that need it.
  for (unsigned i = 0; i != L->getBlocks().size(); ++i) {
    BasicBlock *BB = L->getBlocks()[i];
    if (!BlocksInL.count(BB)) {
      // Move this block to the parent, updating the exit blocks sets
      L->removeBlockFromLoop(BB);
      if ((*LI)[BB] == L)
        LI->changeLoopFor(BB, NewOuter);
      --i;
    }
  }

  return NewOuter;
}



/// InsertUniqueBackedgeBlock - This method is called when the specified loop
/// has more than one backedge in it.  If this occurs, revector all of these
/// backedges to target a new basic block and have that block branch to the loop
/// header.  This ensures that loops have exactly one backedge.
///
BasicBlock *
LoopSimplify::InsertUniqueBackedgeBlock(Loop *L, BasicBlock *Preheader) {
  assert(L->getNumBackEdges() > 1 && "Must have > 1 backedge!");

  // Get information about the loop
  BasicBlock *Header = L->getHeader();
  Function *F = Header->getParent();

  // Unique backedge insertion currently depends on having a preheader.
  if (!Preheader)
    return 0;

  // Figure out which basic blocks contain back-edges to the loop header.
  std::vector<BasicBlock*> BackedgeBlocks;
  for (pred_iterator I = pred_begin(Header), E = pred_end(Header); I != E; ++I){
    BasicBlock *P = *I;

    // Indirectbr edges cannot be split, so we must fail if we find one.
    if (isa<IndirectBrInst>(P->getTerminator()))
      return 0;

    if (P != Preheader) BackedgeBlocks.push_back(P);
  }

  // Create and insert the new backedge block...
  BasicBlock *BEBlock = BasicBlock::Create(Header->getContext(),
                                           Header->getName()+".backedge", F);
  BranchInst *BETerminator = BranchInst::Create(Header, BEBlock);

  DEBUG(dbgs() << "LoopSimplify: Inserting unique backedge block ";
        WriteAsOperand(dbgs(), BEBlock, false);
        dbgs() << "\n");

  // Move the new backedge block to right after the last backedge block.
  Function::iterator InsertPos = BackedgeBlocks.back(); ++InsertPos;
  F->getBasicBlockList().splice(InsertPos, F->getBasicBlockList(), BEBlock);

  // Now that the block has been inserted into the function, create PHI nodes in
  // the backedge block which correspond to any PHI nodes in the header block.
  for (BasicBlock::iterator I = Header->begin(); isa<PHINode>(I); ++I) {
    PHINode *PN = cast<PHINode>(I);
    PHINode *NewPN = PHINode::Create(PN->getType(), PN->getName()+".be",
                                     BETerminator);
    NewPN->reserveOperandSpace(BackedgeBlocks.size());
    if (AA) AA->copyValue(PN, NewPN);

    // Loop over the PHI node, moving all entries except the one for the
    // preheader over to the new PHI node.
    unsigned PreheaderIdx = ~0U;
    bool HasUniqueIncomingValue = true;
    Value *UniqueValue = 0;
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
      BasicBlock *IBB = PN->getIncomingBlock(i);
      Value *IV = PN->getIncomingValue(i);
      if (IBB == Preheader) {
        PreheaderIdx = i;
      } else {
        NewPN->addIncoming(IV, IBB);
        if (HasUniqueIncomingValue) {
          if (UniqueValue == 0)
            UniqueValue = IV;
          else if (UniqueValue != IV)
            HasUniqueIncomingValue = false;
        }
      }
    }

    // Delete all of the incoming values from the old PN except the preheader's
    assert(PreheaderIdx != ~0U && "PHI has no preheader entry??");
    if (PreheaderIdx != 0) {
      PN->setIncomingValue(0, PN->getIncomingValue(PreheaderIdx));
      PN->setIncomingBlock(0, PN->getIncomingBlock(PreheaderIdx));
    }
    // Nuke all entries except the zero'th.
    for (unsigned i = 0, e = PN->getNumIncomingValues()-1; i != e; ++i)
      PN->removeIncomingValue(e-i, false);

    // Finally, add the newly constructed PHI node as the entry for the BEBlock.
    PN->addIncoming(NewPN, BEBlock);

    // As an optimization, if all incoming values in the new PhiNode (which is a
    // subset of the incoming values of the old PHI node) have the same value,
    // eliminate the PHI Node.
    if (HasUniqueIncomingValue) {
      NewPN->replaceAllUsesWith(UniqueValue);
      if (AA) AA->deleteValue(NewPN);
      BEBlock->getInstList().erase(NewPN);
    }
  }

  // Now that all of the PHI nodes have been inserted and adjusted, modify the
  // backedge blocks to just to the BEBlock instead of the header.
  for (unsigned i = 0, e = BackedgeBlocks.size(); i != e; ++i) {
    TerminatorInst *TI = BackedgeBlocks[i]->getTerminator();
    for (unsigned Op = 0, e = TI->getNumSuccessors(); Op != e; ++Op)
      if (TI->getSuccessor(Op) == Header)
        TI->setSuccessor(Op, BEBlock);
  }

  //===--- Update all analyses which we must preserve now -----------------===//

  // Update Loop Information - we know that this block is now in the current
  // loop and all parent loops.
  L->addBasicBlockToLoop(BEBlock, LI->getBase());

  // Update dominator information
  DT->splitBlock(BEBlock);
  if (DominanceFrontier *DF = getAnalysisIfAvailable<DominanceFrontier>())
    DF->splitBlock(BEBlock);

  return BEBlock;
}

void LoopSimplify::verifyAnalysis() const {
  // It used to be possible to just assert L->isLoopSimplifyForm(), however
  // with the introduction of indirectbr, there are now cases where it's
  // not possible to transform a loop as necessary. We can at least check
  // that there is an indirectbr near any time there's trouble.

  // Indirectbr can interfere with preheader and unique backedge insertion.
  if (!L->getLoopPreheader() || !L->getLoopLatch()) {
    bool HasIndBrPred = false;
    for (pred_iterator PI = pred_begin(L->getHeader()),
         PE = pred_end(L->getHeader()); PI != PE; ++PI)
      if (isa<IndirectBrInst>((*PI)->getTerminator())) {
        HasIndBrPred = true;
        break;
      }
    assert(HasIndBrPred &&
           "LoopSimplify has no excuse for missing loop header info!");
  }

  // Indirectbr can interfere with exit block canonicalization.
  if (!L->hasDedicatedExits()) {
    bool HasIndBrExiting = false;
    SmallVector<BasicBlock*, 8> ExitingBlocks;
    L->getExitingBlocks(ExitingBlocks);
    for (unsigned i = 0, e = ExitingBlocks.size(); i != e; ++i)
      if (isa<IndirectBrInst>((ExitingBlocks[i])->getTerminator())) {
        HasIndBrExiting = true;
        break;
      }
    assert(HasIndBrExiting &&
           "LoopSimplify has no excuse for missing exit block info!");
  }
}
