//===-- LCSSA.cpp - Convert loops into loop-closed SSA form ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass transforms loops by placing phi nodes at the end of the loops for
// all values that are live across the loop boundary.  For example, it turns
// the left into the right code:
// 
// for (...)                for (...)
//   if (c)                   if (c)
//     X1 = ...                 X1 = ...
//   else                     else
//     X2 = ...                 X2 = ...
//   X3 = phi(X1, X2)         X3 = phi(X1, X2)
// ... = X3 + 4             X4 = phi(X3)
//                          ... = X4 + 4
//
// This is still valid LLVM; the extra phi nodes are purely redundant, and will
// be trivially eliminated by InstCombine.  The major benefit of this 
// transformation is that it makes many other loop optimizations, such as 
// LoopUnswitching, simpler.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "lcssa"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Constants.h"
#include "llvm/Pass.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/PredIteratorCache.h"
using namespace llvm;

STATISTIC(NumLCSSA, "Number of live out of a loop variables");

namespace {
  struct LCSSA : public LoopPass {
    static char ID; // Pass identification, replacement for typeid
    LCSSA() : LoopPass(ID) {
      initializeLCSSAPass(*PassRegistry::getPassRegistry());
    }

    // Cached analysis information for the current function.
    DominatorTree *DT;
    LoopInfo *LI;
    ScalarEvolution *SE;
    std::vector<BasicBlock*> LoopBlocks;
    PredIteratorCache PredCache;
    Loop *L;
    
    virtual bool runOnLoop(Loop *L, LPPassManager &LPM);

    /// This transformation requires natural loop information & requires that
    /// loop preheaders be inserted into the CFG.  It maintains both of these,
    /// as well as the CFG.  It also requires dominator information.
    ///
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();

      AU.addRequired<DominatorTree>();
      AU.addRequired<LoopInfo>();
      AU.addPreservedID(LoopSimplifyID);
      AU.addPreserved<ScalarEvolution>();
    }
  private:
    bool ProcessInstruction(Instruction *Inst,
                            const SmallVectorImpl<BasicBlock*> &ExitBlocks);
    
    /// verifyAnalysis() - Verify loop nest.
    virtual void verifyAnalysis() const {
      // Check the special guarantees that LCSSA makes.
      assert(L->isLCSSAForm(*DT) && "LCSSA form not preserved!");
    }

    /// inLoop - returns true if the given block is within the current loop
    bool inLoop(BasicBlock *B) const {
      return std::binary_search(LoopBlocks.begin(), LoopBlocks.end(), B);
    }
  };
}
  
char LCSSA::ID = 0;
INITIALIZE_PASS_BEGIN(LCSSA, "lcssa", "Loop-Closed SSA Form Pass", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTree)
INITIALIZE_PASS_DEPENDENCY(LoopInfo)
INITIALIZE_PASS_END(LCSSA, "lcssa", "Loop-Closed SSA Form Pass", false, false)

Pass *llvm::createLCSSAPass() { return new LCSSA(); }
char &llvm::LCSSAID = LCSSA::ID;


/// BlockDominatesAnExit - Return true if the specified block dominates at least
/// one of the blocks in the specified list.
static bool BlockDominatesAnExit(BasicBlock *BB,
                                 const SmallVectorImpl<BasicBlock*> &ExitBlocks,
                                 DominatorTree *DT) {
  DomTreeNode *DomNode = DT->getNode(BB);
  for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i)
    if (DT->dominates(DomNode, DT->getNode(ExitBlocks[i])))
      return true;

  return false;
}


/// runOnFunction - Process all loops in the function, inner-most out.
bool LCSSA::runOnLoop(Loop *TheLoop, LPPassManager &LPM) {
  L = TheLoop;
  
  DT = &getAnalysis<DominatorTree>();
  LI = &getAnalysis<LoopInfo>();
  SE = getAnalysisIfAvailable<ScalarEvolution>();

  // Get the set of exiting blocks.
  SmallVector<BasicBlock*, 8> ExitBlocks;
  L->getExitBlocks(ExitBlocks);
  
  if (ExitBlocks.empty())
    return false;
  
  // Speed up queries by creating a sorted vector of blocks.
  LoopBlocks.clear();
  LoopBlocks.insert(LoopBlocks.end(), L->block_begin(), L->block_end());
  array_pod_sort(LoopBlocks.begin(), LoopBlocks.end());
  
  // Look at all the instructions in the loop, checking to see if they have uses
  // outside the loop.  If so, rewrite those uses.
  bool MadeChange = false;
  
  for (Loop::block_iterator BBI = L->block_begin(), E = L->block_end();
       BBI != E; ++BBI) {
    BasicBlock *BB = *BBI;
    
    // For large loops, avoid use-scanning by using dominance information:  In
    // particular, if a block does not dominate any of the loop exits, then none
    // of the values defined in the block could be used outside the loop.
    if (!BlockDominatesAnExit(BB, ExitBlocks, DT))
      continue;
    
    for (BasicBlock::iterator I = BB->begin(), E = BB->end();
         I != E; ++I) {
      // Reject two common cases fast: instructions with no uses (like stores)
      // and instructions with one use that is in the same block as this.
      if (I->use_empty() ||
          (I->hasOneUse() && I->use_back()->getParent() == BB &&
           !isa<PHINode>(I->use_back())))
        continue;
      
      MadeChange |= ProcessInstruction(I, ExitBlocks);
    }
  }

  // If we modified the code, remove any caches about the loop from SCEV to
  // avoid dangling entries.
  // FIXME: This is a big hammer, can we clear the cache more selectively?
  if (SE && MadeChange)
    SE->forgetLoop(L);
  
  assert(L->isLCSSAForm(*DT));
  PredCache.clear();

  return MadeChange;
}

/// isExitBlock - Return true if the specified block is in the list.
static bool isExitBlock(BasicBlock *BB,
                        const SmallVectorImpl<BasicBlock*> &ExitBlocks) {
  for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i)
    if (ExitBlocks[i] == BB)
      return true;
  return false;
}

/// ProcessInstruction - Given an instruction in the loop, check to see if it
/// has any uses that are outside the current loop.  If so, insert LCSSA PHI
/// nodes and rewrite the uses.
bool LCSSA::ProcessInstruction(Instruction *Inst,
                               const SmallVectorImpl<BasicBlock*> &ExitBlocks) {
  SmallVector<Use*, 16> UsesToRewrite;
  
  BasicBlock *InstBB = Inst->getParent();
  
  for (Value::use_iterator UI = Inst->use_begin(), E = Inst->use_end();
       UI != E; ++UI) {
    User *U = *UI;
    BasicBlock *UserBB = cast<Instruction>(U)->getParent();
    if (PHINode *PN = dyn_cast<PHINode>(U))
      UserBB = PN->getIncomingBlock(UI);
    
    if (InstBB != UserBB && !inLoop(UserBB))
      UsesToRewrite.push_back(&UI.getUse());
  }

  // If there are no uses outside the loop, exit with no change.
  if (UsesToRewrite.empty()) return false;
  
  ++NumLCSSA; // We are applying the transformation

  // Invoke instructions are special in that their result value is not available
  // along their unwind edge. The code below tests to see whether DomBB dominates
  // the value, so adjust DomBB to the normal destination block, which is
  // effectively where the value is first usable.
  BasicBlock *DomBB = Inst->getParent();
  if (InvokeInst *Inv = dyn_cast<InvokeInst>(Inst))
    DomBB = Inv->getNormalDest();

  DomTreeNode *DomNode = DT->getNode(DomBB);

  SmallVector<PHINode*, 16> AddedPHIs;

  SSAUpdater SSAUpdate;
  SSAUpdate.Initialize(Inst->getType(), Inst->getName());
  
  // Insert the LCSSA phi's into all of the exit blocks dominated by the
  // value, and add them to the Phi's map.
  for (SmallVectorImpl<BasicBlock*>::const_iterator BBI = ExitBlocks.begin(),
      BBE = ExitBlocks.end(); BBI != BBE; ++BBI) {
    BasicBlock *ExitBB = *BBI;
    if (!DT->dominates(DomNode, DT->getNode(ExitBB))) continue;
    
    // If we already inserted something for this BB, don't reprocess it.
    if (SSAUpdate.HasValueForBlock(ExitBB)) continue;
    
    PHINode *PN = PHINode::Create(Inst->getType(),
                                  PredCache.GetNumPreds(ExitBB),
                                  Inst->getName()+".lcssa",
                                  ExitBB->begin());

    // Add inputs from inside the loop for this PHI.
    for (BasicBlock **PI = PredCache.GetPreds(ExitBB); *PI; ++PI) {
      PN->addIncoming(Inst, *PI);

      // If the exit block has a predecessor not within the loop, arrange for
      // the incoming value use corresponding to that predecessor to be
      // rewritten in terms of a different LCSSA PHI.
      if (!inLoop(*PI))
        UsesToRewrite.push_back(
          &PN->getOperandUse(
            PN->getOperandNumForIncomingValue(PN->getNumIncomingValues()-1)));
    }

    AddedPHIs.push_back(PN);
    
    // Remember that this phi makes the value alive in this block.
    SSAUpdate.AddAvailableValue(ExitBB, PN);
  }

  // Rewrite all uses outside the loop in terms of the new PHIs we just
  // inserted.
  for (unsigned i = 0, e = UsesToRewrite.size(); i != e; ++i) {
    // If this use is in an exit block, rewrite to use the newly inserted PHI.
    // This is required for correctness because SSAUpdate doesn't handle uses in
    // the same block.  It assumes the PHI we inserted is at the end of the
    // block.
    Instruction *User = cast<Instruction>(UsesToRewrite[i]->getUser());
    BasicBlock *UserBB = User->getParent();
    if (PHINode *PN = dyn_cast<PHINode>(User))
      UserBB = PN->getIncomingBlock(*UsesToRewrite[i]);

    if (isa<PHINode>(UserBB->begin()) &&
        isExitBlock(UserBB, ExitBlocks)) {
      // Tell the VHs that the uses changed. This updates SCEV's caches.
      if (UsesToRewrite[i]->get()->hasValueHandle())
        ValueHandleBase::ValueIsRAUWd(*UsesToRewrite[i], UserBB->begin());
      UsesToRewrite[i]->set(UserBB->begin());
      continue;
    }
    
    // Otherwise, do full PHI insertion.
    SSAUpdate.RewriteUse(*UsesToRewrite[i]);
  }

  // Remove PHI nodes that did not have any uses rewritten.
  for (unsigned i = 0, e = AddedPHIs.size(); i != e; ++i) {
    if (AddedPHIs[i]->use_empty())
      AddedPHIs[i]->eraseFromParent();
  }
  
  return true;
}

