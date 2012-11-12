//===-- BasicBlockUtils.cpp - BasicBlock Utilities -------------------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This family of functions perform manipulations on basic blocks, and
// instructions contained within basic blocks.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Constant.h"
#include "llvm/Type.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/DataLayout.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ValueHandle.h"
#include <algorithm>
using namespace llvm;

/// DeleteDeadBlock - Delete the specified block, which must have no
/// predecessors.
void llvm::DeleteDeadBlock(BasicBlock *BB) {
  assert((pred_begin(BB) == pred_end(BB) ||
         // Can delete self loop.
         BB->getSinglePredecessor() == BB) && "Block is not dead!");
  TerminatorInst *BBTerm = BB->getTerminator();
  
  // Loop through all of our successors and make sure they know that one
  // of their predecessors is going away.
  for (unsigned i = 0, e = BBTerm->getNumSuccessors(); i != e; ++i)
    BBTerm->getSuccessor(i)->removePredecessor(BB);
  
  // Zap all the instructions in the block.
  while (!BB->empty()) {
    Instruction &I = BB->back();
    // If this instruction is used, replace uses with an arbitrary value.
    // Because control flow can't get here, we don't care what we replace the
    // value with.  Note that since this block is unreachable, and all values
    // contained within it must dominate their uses, that all uses will
    // eventually be removed (they are themselves dead).
    if (!I.use_empty())
      I.replaceAllUsesWith(UndefValue::get(I.getType()));
    BB->getInstList().pop_back();
  }
  
  // Zap the block!
  BB->eraseFromParent();
}

/// FoldSingleEntryPHINodes - We know that BB has one predecessor.  If there are
/// any single-entry PHI nodes in it, fold them away.  This handles the case
/// when all entries to the PHI nodes in a block are guaranteed equal, such as
/// when the block has exactly one predecessor.
void llvm::FoldSingleEntryPHINodes(BasicBlock *BB, Pass *P) {
  if (!isa<PHINode>(BB->begin())) return;
  
  AliasAnalysis *AA = 0;
  MemoryDependenceAnalysis *MemDep = 0;
  if (P) {
    AA = P->getAnalysisIfAvailable<AliasAnalysis>();
    MemDep = P->getAnalysisIfAvailable<MemoryDependenceAnalysis>();
  }
  
  while (PHINode *PN = dyn_cast<PHINode>(BB->begin())) {
    if (PN->getIncomingValue(0) != PN)
      PN->replaceAllUsesWith(PN->getIncomingValue(0));
    else
      PN->replaceAllUsesWith(UndefValue::get(PN->getType()));
    
    if (MemDep)
      MemDep->removeInstruction(PN);  // Memdep updates AA itself.
    else if (AA && isa<PointerType>(PN->getType()))
      AA->deleteValue(PN);
    
    PN->eraseFromParent();
  }
}


/// DeleteDeadPHIs - Examine each PHI in the given block and delete it if it
/// is dead. Also recursively delete any operands that become dead as
/// a result. This includes tracing the def-use list from the PHI to see if
/// it is ultimately unused or if it reaches an unused cycle.
bool llvm::DeleteDeadPHIs(BasicBlock *BB, const TargetLibraryInfo *TLI) {
  // Recursively deleting a PHI may cause multiple PHIs to be deleted
  // or RAUW'd undef, so use an array of WeakVH for the PHIs to delete.
  SmallVector<WeakVH, 8> PHIs;
  for (BasicBlock::iterator I = BB->begin();
       PHINode *PN = dyn_cast<PHINode>(I); ++I)
    PHIs.push_back(PN);

  bool Changed = false;
  for (unsigned i = 0, e = PHIs.size(); i != e; ++i)
    if (PHINode *PN = dyn_cast_or_null<PHINode>(PHIs[i].operator Value*()))
      Changed |= RecursivelyDeleteDeadPHINode(PN, TLI);

  return Changed;
}

/// MergeBlockIntoPredecessor - Attempts to merge a block into its predecessor,
/// if possible.  The return value indicates success or failure.
bool llvm::MergeBlockIntoPredecessor(BasicBlock *BB, Pass *P) {
  // Don't merge away blocks who have their address taken.
  if (BB->hasAddressTaken()) return false;
  
  // Can't merge if there are multiple predecessors, or no predecessors.
  BasicBlock *PredBB = BB->getUniquePredecessor();
  if (!PredBB) return false;

  // Don't break self-loops.
  if (PredBB == BB) return false;
  // Don't break invokes.
  if (isa<InvokeInst>(PredBB->getTerminator())) return false;
  
  succ_iterator SI(succ_begin(PredBB)), SE(succ_end(PredBB));
  BasicBlock *OnlySucc = BB;
  for (; SI != SE; ++SI)
    if (*SI != OnlySucc) {
      OnlySucc = 0;     // There are multiple distinct successors!
      break;
    }
  
  // Can't merge if there are multiple successors.
  if (!OnlySucc) return false;

  // Can't merge if there is PHI loop.
  for (BasicBlock::iterator BI = BB->begin(), BE = BB->end(); BI != BE; ++BI) {
    if (PHINode *PN = dyn_cast<PHINode>(BI)) {
      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
        if (PN->getIncomingValue(i) == PN)
          return false;
    } else
      break;
  }

  // Begin by getting rid of unneeded PHIs.
  if (isa<PHINode>(BB->front()))
    FoldSingleEntryPHINodes(BB, P);
  
  // Delete the unconditional branch from the predecessor...
  PredBB->getInstList().pop_back();
  
  // Make all PHI nodes that referred to BB now refer to Pred as their
  // source...
  BB->replaceAllUsesWith(PredBB);
  
  // Move all definitions in the successor to the predecessor...
  PredBB->getInstList().splice(PredBB->end(), BB->getInstList());
  
  // Inherit predecessors name if it exists.
  if (!PredBB->hasName())
    PredBB->takeName(BB);
  
  // Finally, erase the old block and update dominator info.
  if (P) {
    if (DominatorTree *DT = P->getAnalysisIfAvailable<DominatorTree>()) {
      if (DomTreeNode *DTN = DT->getNode(BB)) {
        DomTreeNode *PredDTN = DT->getNode(PredBB);
        SmallVector<DomTreeNode*, 8> Children(DTN->begin(), DTN->end());
        for (SmallVector<DomTreeNode*, 8>::iterator DI = Children.begin(),
             DE = Children.end(); DI != DE; ++DI)
          DT->changeImmediateDominator(*DI, PredDTN);

        DT->eraseNode(BB);
      }
      
      if (LoopInfo *LI = P->getAnalysisIfAvailable<LoopInfo>())
        LI->removeBlock(BB);
      
      if (MemoryDependenceAnalysis *MD =
            P->getAnalysisIfAvailable<MemoryDependenceAnalysis>())
        MD->invalidateCachedPredecessors();
    }
  }
  
  BB->eraseFromParent();
  return true;
}

/// ReplaceInstWithValue - Replace all uses of an instruction (specified by BI)
/// with a value, then remove and delete the original instruction.
///
void llvm::ReplaceInstWithValue(BasicBlock::InstListType &BIL,
                                BasicBlock::iterator &BI, Value *V) {
  Instruction &I = *BI;
  // Replaces all of the uses of the instruction with uses of the value
  I.replaceAllUsesWith(V);

  // Make sure to propagate a name if there is one already.
  if (I.hasName() && !V->hasName())
    V->takeName(&I);

  // Delete the unnecessary instruction now...
  BI = BIL.erase(BI);
}


/// ReplaceInstWithInst - Replace the instruction specified by BI with the
/// instruction specified by I.  The original instruction is deleted and BI is
/// updated to point to the new instruction.
///
void llvm::ReplaceInstWithInst(BasicBlock::InstListType &BIL,
                               BasicBlock::iterator &BI, Instruction *I) {
  assert(I->getParent() == 0 &&
         "ReplaceInstWithInst: Instruction already inserted into basic block!");

  // Insert the new instruction into the basic block...
  BasicBlock::iterator New = BIL.insert(BI, I);

  // Replace all uses of the old instruction, and delete it.
  ReplaceInstWithValue(BIL, BI, I);

  // Move BI back to point to the newly inserted instruction
  BI = New;
}

/// ReplaceInstWithInst - Replace the instruction specified by From with the
/// instruction specified by To.
///
void llvm::ReplaceInstWithInst(Instruction *From, Instruction *To) {
  BasicBlock::iterator BI(From);
  ReplaceInstWithInst(From->getParent()->getInstList(), BI, To);
}

/// GetSuccessorNumber - Search for the specified successor of basic block BB
/// and return its position in the terminator instruction's list of
/// successors.  It is an error to call this with a block that is not a
/// successor.
unsigned llvm::GetSuccessorNumber(BasicBlock *BB, BasicBlock *Succ) {
  TerminatorInst *Term = BB->getTerminator();
#ifndef NDEBUG
  unsigned e = Term->getNumSuccessors();
#endif
  for (unsigned i = 0; ; ++i) {
    assert(i != e && "Didn't find edge?");
    if (Term->getSuccessor(i) == Succ)
      return i;
  }
}

/// SplitEdge -  Split the edge connecting specified block. Pass P must 
/// not be NULL. 
BasicBlock *llvm::SplitEdge(BasicBlock *BB, BasicBlock *Succ, Pass *P) {
  unsigned SuccNum = GetSuccessorNumber(BB, Succ);
  
  // If this is a critical edge, let SplitCriticalEdge do it.
  TerminatorInst *LatchTerm = BB->getTerminator();
  if (SplitCriticalEdge(LatchTerm, SuccNum, P))
    return LatchTerm->getSuccessor(SuccNum);

  // If the edge isn't critical, then BB has a single successor or Succ has a
  // single pred.  Split the block.
  BasicBlock::iterator SplitPoint;
  if (BasicBlock *SP = Succ->getSinglePredecessor()) {
    // If the successor only has a single pred, split the top of the successor
    // block.
    assert(SP == BB && "CFG broken");
    SP = NULL;
    return SplitBlock(Succ, Succ->begin(), P);
  }
  
  // Otherwise, if BB has a single successor, split it at the bottom of the
  // block.
  assert(BB->getTerminator()->getNumSuccessors() == 1 &&
         "Should have a single succ!"); 
  return SplitBlock(BB, BB->getTerminator(), P);
}

/// SplitBlock - Split the specified block at the specified instruction - every
/// thing before SplitPt stays in Old and everything starting with SplitPt moves
/// to a new block.  The two blocks are joined by an unconditional branch and
/// the loop info is updated.
///
BasicBlock *llvm::SplitBlock(BasicBlock *Old, Instruction *SplitPt, Pass *P) {
  BasicBlock::iterator SplitIt = SplitPt;
  while (isa<PHINode>(SplitIt) || isa<LandingPadInst>(SplitIt))
    ++SplitIt;
  BasicBlock *New = Old->splitBasicBlock(SplitIt, Old->getName()+".split");

  // The new block lives in whichever loop the old one did. This preserves
  // LCSSA as well, because we force the split point to be after any PHI nodes.
  if (LoopInfo *LI = P->getAnalysisIfAvailable<LoopInfo>())
    if (Loop *L = LI->getLoopFor(Old))
      L->addBasicBlockToLoop(New, LI->getBase());

  if (DominatorTree *DT = P->getAnalysisIfAvailable<DominatorTree>()) {
    // Old dominates New. New node dominates all other nodes dominated by Old.
    if (DomTreeNode *OldNode = DT->getNode(Old)) {
      std::vector<DomTreeNode *> Children;
      for (DomTreeNode::iterator I = OldNode->begin(), E = OldNode->end();
           I != E; ++I) 
        Children.push_back(*I);

      DomTreeNode *NewNode = DT->addNewBlock(New,Old);
      for (std::vector<DomTreeNode *>::iterator I = Children.begin(),
             E = Children.end(); I != E; ++I) 
        DT->changeImmediateDominator(*I, NewNode);
    }
  }

  return New;
}

/// UpdateAnalysisInformation - Update DominatorTree, LoopInfo, and LCCSA
/// analysis information.
static void UpdateAnalysisInformation(BasicBlock *OldBB, BasicBlock *NewBB,
                                      ArrayRef<BasicBlock *> Preds,
                                      Pass *P, bool &HasLoopExit) {
  if (!P) return;

  LoopInfo *LI = P->getAnalysisIfAvailable<LoopInfo>();
  Loop *L = LI ? LI->getLoopFor(OldBB) : 0;

  // If we need to preserve loop analyses, collect some information about how
  // this split will affect loops.
  bool IsLoopEntry = !!L;
  bool SplitMakesNewLoopHeader = false;
  if (LI) {
    bool PreserveLCSSA = P->mustPreserveAnalysisID(LCSSAID);
    for (ArrayRef<BasicBlock*>::iterator
           i = Preds.begin(), e = Preds.end(); i != e; ++i) {
      BasicBlock *Pred = *i;

      // If we need to preserve LCSSA, determine if any of the preds is a loop
      // exit.
      if (PreserveLCSSA)
        if (Loop *PL = LI->getLoopFor(Pred))
          if (!PL->contains(OldBB))
            HasLoopExit = true;

      // If we need to preserve LoopInfo, note whether any of the preds crosses
      // an interesting loop boundary.
      if (!L) continue;
      if (L->contains(Pred))
        IsLoopEntry = false;
      else
        SplitMakesNewLoopHeader = true;
    }
  }

  // Update dominator tree if available.
  DominatorTree *DT = P->getAnalysisIfAvailable<DominatorTree>();
  if (DT)
    DT->splitBlock(NewBB);

  if (!L) return;

  if (IsLoopEntry) {
    // Add the new block to the nearest enclosing loop (and not an adjacent
    // loop). To find this, examine each of the predecessors and determine which
    // loops enclose them, and select the most-nested loop which contains the
    // loop containing the block being split.
    Loop *InnermostPredLoop = 0;
    for (ArrayRef<BasicBlock*>::iterator
           i = Preds.begin(), e = Preds.end(); i != e; ++i) {
      BasicBlock *Pred = *i;
      if (Loop *PredLoop = LI->getLoopFor(Pred)) {
        // Seek a loop which actually contains the block being split (to avoid
        // adjacent loops).
        while (PredLoop && !PredLoop->contains(OldBB))
          PredLoop = PredLoop->getParentLoop();

        // Select the most-nested of these loops which contains the block.
        if (PredLoop && PredLoop->contains(OldBB) &&
            (!InnermostPredLoop ||
             InnermostPredLoop->getLoopDepth() < PredLoop->getLoopDepth()))
          InnermostPredLoop = PredLoop;
      }
    }

    if (InnermostPredLoop)
      InnermostPredLoop->addBasicBlockToLoop(NewBB, LI->getBase());
  } else {
    L->addBasicBlockToLoop(NewBB, LI->getBase());
    if (SplitMakesNewLoopHeader)
      L->moveToHeader(NewBB);
  }
}

/// UpdatePHINodes - Update the PHI nodes in OrigBB to include the values coming
/// from NewBB. This also updates AliasAnalysis, if available.
static void UpdatePHINodes(BasicBlock *OrigBB, BasicBlock *NewBB,
                           ArrayRef<BasicBlock*> Preds, BranchInst *BI,
                           Pass *P, bool HasLoopExit) {
  // Otherwise, create a new PHI node in NewBB for each PHI node in OrigBB.
  AliasAnalysis *AA = P ? P->getAnalysisIfAvailable<AliasAnalysis>() : 0;
  for (BasicBlock::iterator I = OrigBB->begin(); isa<PHINode>(I); ) {
    PHINode *PN = cast<PHINode>(I++);

    // Check to see if all of the values coming in are the same.  If so, we
    // don't need to create a new PHI node, unless it's needed for LCSSA.
    Value *InVal = 0;
    if (!HasLoopExit) {
      InVal = PN->getIncomingValueForBlock(Preds[0]);
      for (unsigned i = 1, e = Preds.size(); i != e; ++i)
        if (InVal != PN->getIncomingValueForBlock(Preds[i])) {
          InVal = 0;
          break;
        }
    }

    if (InVal) {
      // If all incoming values for the new PHI would be the same, just don't
      // make a new PHI.  Instead, just remove the incoming values from the old
      // PHI.
      for (unsigned i = 0, e = Preds.size(); i != e; ++i)
        PN->removeIncomingValue(Preds[i], false);
    } else {
      // If the values coming into the block are not the same, we need a PHI.
      // Create the new PHI node, insert it into NewBB at the end of the block
      PHINode *NewPHI =
        PHINode::Create(PN->getType(), Preds.size(), PN->getName() + ".ph", BI);
      if (AA) AA->copyValue(PN, NewPHI);
      
      // Move all of the PHI values for 'Preds' to the new PHI.
      for (unsigned i = 0, e = Preds.size(); i != e; ++i) {
        Value *V = PN->removeIncomingValue(Preds[i], false);
        NewPHI->addIncoming(V, Preds[i]);
      }

      InVal = NewPHI;
    }

    // Add an incoming value to the PHI node in the loop for the preheader
    // edge.
    PN->addIncoming(InVal, NewBB);
  }
}

/// SplitBlockPredecessors - This method transforms BB by introducing a new
/// basic block into the function, and moving some of the predecessors of BB to
/// be predecessors of the new block.  The new predecessors are indicated by the
/// Preds array, which has NumPreds elements in it.  The new block is given a
/// suffix of 'Suffix'.
///
/// This currently updates the LLVM IR, AliasAnalysis, DominatorTree,
/// LoopInfo, and LCCSA but no other analyses. In particular, it does not
/// preserve LoopSimplify (because it's complicated to handle the case where one
/// of the edges being split is an exit of a loop with other exits).
///
BasicBlock *llvm::SplitBlockPredecessors(BasicBlock *BB, 
                                         ArrayRef<BasicBlock*> Preds,
                                         const char *Suffix, Pass *P) {
  // Create new basic block, insert right before the original block.
  BasicBlock *NewBB = BasicBlock::Create(BB->getContext(), BB->getName()+Suffix,
                                         BB->getParent(), BB);
  
  // The new block unconditionally branches to the old block.
  BranchInst *BI = BranchInst::Create(BB, NewBB);
  
  // Move the edges from Preds to point to NewBB instead of BB.
  for (unsigned i = 0, e = Preds.size(); i != e; ++i) {
    // This is slightly more strict than necessary; the minimum requirement
    // is that there be no more than one indirectbr branching to BB. And
    // all BlockAddress uses would need to be updated.
    assert(!isa<IndirectBrInst>(Preds[i]->getTerminator()) &&
           "Cannot split an edge from an IndirectBrInst");
    Preds[i]->getTerminator()->replaceUsesOfWith(BB, NewBB);
  }

  // Insert a new PHI node into NewBB for every PHI node in BB and that new PHI
  // node becomes an incoming value for BB's phi node.  However, if the Preds
  // list is empty, we need to insert dummy entries into the PHI nodes in BB to
  // account for the newly created predecessor.
  if (Preds.size() == 0) {
    // Insert dummy values as the incoming value.
    for (BasicBlock::iterator I = BB->begin(); isa<PHINode>(I); ++I)
      cast<PHINode>(I)->addIncoming(UndefValue::get(I->getType()), NewBB);
    return NewBB;
  }

  // Update DominatorTree, LoopInfo, and LCCSA analysis information.
  bool HasLoopExit = false;
  UpdateAnalysisInformation(BB, NewBB, Preds, P, HasLoopExit);

  // Update the PHI nodes in BB with the values coming from NewBB.
  UpdatePHINodes(BB, NewBB, Preds, BI, P, HasLoopExit);
  return NewBB;
}

/// SplitLandingPadPredecessors - This method transforms the landing pad,
/// OrigBB, by introducing two new basic blocks into the function. One of those
/// new basic blocks gets the predecessors listed in Preds. The other basic
/// block gets the remaining predecessors of OrigBB. The landingpad instruction
/// OrigBB is clone into both of the new basic blocks. The new blocks are given
/// the suffixes 'Suffix1' and 'Suffix2', and are returned in the NewBBs vector.
/// 
/// This currently updates the LLVM IR, AliasAnalysis, DominatorTree,
/// DominanceFrontier, LoopInfo, and LCCSA but no other analyses. In particular,
/// it does not preserve LoopSimplify (because it's complicated to handle the
/// case where one of the edges being split is an exit of a loop with other
/// exits).
/// 
void llvm::SplitLandingPadPredecessors(BasicBlock *OrigBB,
                                       ArrayRef<BasicBlock*> Preds,
                                       const char *Suffix1, const char *Suffix2,
                                       Pass *P,
                                       SmallVectorImpl<BasicBlock*> &NewBBs) {
  assert(OrigBB->isLandingPad() && "Trying to split a non-landing pad!");

  // Create a new basic block for OrigBB's predecessors listed in Preds. Insert
  // it right before the original block.
  BasicBlock *NewBB1 = BasicBlock::Create(OrigBB->getContext(),
                                          OrigBB->getName() + Suffix1,
                                          OrigBB->getParent(), OrigBB);
  NewBBs.push_back(NewBB1);

  // The new block unconditionally branches to the old block.
  BranchInst *BI1 = BranchInst::Create(OrigBB, NewBB1);

  // Move the edges from Preds to point to NewBB1 instead of OrigBB.
  for (unsigned i = 0, e = Preds.size(); i != e; ++i) {
    // This is slightly more strict than necessary; the minimum requirement
    // is that there be no more than one indirectbr branching to BB. And
    // all BlockAddress uses would need to be updated.
    assert(!isa<IndirectBrInst>(Preds[i]->getTerminator()) &&
           "Cannot split an edge from an IndirectBrInst");
    Preds[i]->getTerminator()->replaceUsesOfWith(OrigBB, NewBB1);
  }

  // Update DominatorTree, LoopInfo, and LCCSA analysis information.
  bool HasLoopExit = false;
  UpdateAnalysisInformation(OrigBB, NewBB1, Preds, P, HasLoopExit);

  // Update the PHI nodes in OrigBB with the values coming from NewBB1.
  UpdatePHINodes(OrigBB, NewBB1, Preds, BI1, P, HasLoopExit);

  // Move the remaining edges from OrigBB to point to NewBB2.
  SmallVector<BasicBlock*, 8> NewBB2Preds;
  for (pred_iterator i = pred_begin(OrigBB), e = pred_end(OrigBB);
       i != e; ) {
    BasicBlock *Pred = *i++;
    if (Pred == NewBB1) continue;
    assert(!isa<IndirectBrInst>(Pred->getTerminator()) &&
           "Cannot split an edge from an IndirectBrInst");
    NewBB2Preds.push_back(Pred);
    e = pred_end(OrigBB);
  }

  BasicBlock *NewBB2 = 0;
  if (!NewBB2Preds.empty()) {
    // Create another basic block for the rest of OrigBB's predecessors.
    NewBB2 = BasicBlock::Create(OrigBB->getContext(),
                                OrigBB->getName() + Suffix2,
                                OrigBB->getParent(), OrigBB);
    NewBBs.push_back(NewBB2);

    // The new block unconditionally branches to the old block.
    BranchInst *BI2 = BranchInst::Create(OrigBB, NewBB2);

    // Move the remaining edges from OrigBB to point to NewBB2.
    for (SmallVectorImpl<BasicBlock*>::iterator
           i = NewBB2Preds.begin(), e = NewBB2Preds.end(); i != e; ++i)
      (*i)->getTerminator()->replaceUsesOfWith(OrigBB, NewBB2);

    // Update DominatorTree, LoopInfo, and LCCSA analysis information.
    HasLoopExit = false;
    UpdateAnalysisInformation(OrigBB, NewBB2, NewBB2Preds, P, HasLoopExit);

    // Update the PHI nodes in OrigBB with the values coming from NewBB2.
    UpdatePHINodes(OrigBB, NewBB2, NewBB2Preds, BI2, P, HasLoopExit);
  }

  LandingPadInst *LPad = OrigBB->getLandingPadInst();
  Instruction *Clone1 = LPad->clone();
  Clone1->setName(Twine("lpad") + Suffix1);
  NewBB1->getInstList().insert(NewBB1->getFirstInsertionPt(), Clone1);

  if (NewBB2) {
    Instruction *Clone2 = LPad->clone();
    Clone2->setName(Twine("lpad") + Suffix2);
    NewBB2->getInstList().insert(NewBB2->getFirstInsertionPt(), Clone2);

    // Create a PHI node for the two cloned landingpad instructions.
    PHINode *PN = PHINode::Create(LPad->getType(), 2, "lpad.phi", LPad);
    PN->addIncoming(Clone1, NewBB1);
    PN->addIncoming(Clone2, NewBB2);
    LPad->replaceAllUsesWith(PN);
    LPad->eraseFromParent();
  } else {
    // There is no second clone. Just replace the landing pad with the first
    // clone.
    LPad->replaceAllUsesWith(Clone1);
    LPad->eraseFromParent();
  }
}

/// FindFunctionBackedges - Analyze the specified function to find all of the
/// loop backedges in the function and return them.  This is a relatively cheap
/// (compared to computing dominators and loop info) analysis.
///
/// The output is added to Result, as pairs of <from,to> edge info.
void llvm::FindFunctionBackedges(const Function &F,
     SmallVectorImpl<std::pair<const BasicBlock*,const BasicBlock*> > &Result) {
  const BasicBlock *BB = &F.getEntryBlock();
  if (succ_begin(BB) == succ_end(BB))
    return;
  
  SmallPtrSet<const BasicBlock*, 8> Visited;
  SmallVector<std::pair<const BasicBlock*, succ_const_iterator>, 8> VisitStack;
  SmallPtrSet<const BasicBlock*, 8> InStack;
  
  Visited.insert(BB);
  VisitStack.push_back(std::make_pair(BB, succ_begin(BB)));
  InStack.insert(BB);
  do {
    std::pair<const BasicBlock*, succ_const_iterator> &Top = VisitStack.back();
    const BasicBlock *ParentBB = Top.first;
    succ_const_iterator &I = Top.second;
    
    bool FoundNew = false;
    while (I != succ_end(ParentBB)) {
      BB = *I++;
      if (Visited.insert(BB)) {
        FoundNew = true;
        break;
      }
      // Successor is in VisitStack, it's a back edge.
      if (InStack.count(BB))
        Result.push_back(std::make_pair(ParentBB, BB));
    }
    
    if (FoundNew) {
      // Go down one level if there is a unvisited successor.
      InStack.insert(BB);
      VisitStack.push_back(std::make_pair(BB, succ_begin(BB)));
    } else {
      // Go up one level.
      InStack.erase(VisitStack.pop_back_val().first);
    }
  } while (!VisitStack.empty()); 
}

/// FoldReturnIntoUncondBranch - This method duplicates the specified return
/// instruction into a predecessor which ends in an unconditional branch. If
/// the return instruction returns a value defined by a PHI, propagate the
/// right value into the return. It returns the new return instruction in the
/// predecessor.
ReturnInst *llvm::FoldReturnIntoUncondBranch(ReturnInst *RI, BasicBlock *BB,
                                             BasicBlock *Pred) {
  Instruction *UncondBranch = Pred->getTerminator();
  // Clone the return and add it to the end of the predecessor.
  Instruction *NewRet = RI->clone();
  Pred->getInstList().push_back(NewRet);
      
  // If the return instruction returns a value, and if the value was a
  // PHI node in "BB", propagate the right value into the return.
  for (User::op_iterator i = NewRet->op_begin(), e = NewRet->op_end();
       i != e; ++i) {
    Value *V = *i;
    Instruction *NewBC = 0;
    if (BitCastInst *BCI = dyn_cast<BitCastInst>(V)) {
      // Return value might be bitcasted. Clone and insert it before the
      // return instruction.
      V = BCI->getOperand(0);
      NewBC = BCI->clone();
      Pred->getInstList().insert(NewRet, NewBC);
      *i = NewBC;
    }
    if (PHINode *PN = dyn_cast<PHINode>(V)) {
      if (PN->getParent() == BB) {
        if (NewBC)
          NewBC->setOperand(0, PN->getIncomingValueForBlock(Pred));
        else
          *i = PN->getIncomingValueForBlock(Pred);
      }
    }
  }
      
  // Update any PHI nodes in the returning block to realize that we no
  // longer branch to them.
  BB->removePredecessor(Pred);
  UncondBranch->eraseFromParent();
  return cast<ReturnInst>(NewRet);
}

/// SplitBlockAndInsertIfThen - Split the containing block at the
/// specified instruction - everything before and including Cmp stays
/// in the old basic block, and everything after Cmp is moved to a
/// new block. The two blocks are connected by a conditional branch
/// (with value of Cmp being the condition).
/// Before:
///   Head
///   Cmp
///   Tail
/// After:
///   Head
///   Cmp
///   if (Cmp)
///     ThenBlock
///   Tail
///
/// If Unreachable is true, then ThenBlock ends with
/// UnreachableInst, otherwise it branches to Tail.
/// Returns the NewBasicBlock's terminator.

TerminatorInst *llvm::SplitBlockAndInsertIfThen(Instruction *Cmp,
    bool Unreachable, MDNode *BranchWeights) {
  Instruction *SplitBefore = Cmp->getNextNode();
  BasicBlock *Head = SplitBefore->getParent();
  BasicBlock *Tail = Head->splitBasicBlock(SplitBefore);
  TerminatorInst *HeadOldTerm = Head->getTerminator();
  LLVMContext &C = Head->getContext();
  BasicBlock *ThenBlock = BasicBlock::Create(C, "", Head->getParent(), Tail);
  TerminatorInst *CheckTerm;
  if (Unreachable)
    CheckTerm = new UnreachableInst(C, ThenBlock);
  else
    CheckTerm = BranchInst::Create(Tail, ThenBlock);
  BranchInst *HeadNewTerm =
    BranchInst::Create(/*ifTrue*/ThenBlock, /*ifFalse*/Tail, Cmp);
  HeadNewTerm->setMetadata(LLVMContext::MD_prof, BranchWeights);
  ReplaceInstWithInst(HeadOldTerm, HeadNewTerm);
  return CheckTerm;
}
