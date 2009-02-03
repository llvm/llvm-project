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
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Compiler.h"
#include <algorithm>
#include <map>
using namespace llvm;

STATISTIC(NumLCSSA, "Number of live out of a loop variables");

namespace {
  struct VISIBILITY_HIDDEN LCSSA : public LoopPass {
    static char ID; // Pass identification, replacement for typeid
    LCSSA() : LoopPass(&ID) {}

    // Cached analysis information for the current function.
    LoopInfo *LI;
    DominatorTree *DT;
    std::vector<BasicBlock*> LoopBlocks;
    
    virtual bool runOnLoop(Loop *L, LPPassManager &LPM);

    void ProcessInstruction(Instruction* Instr,
                            const SmallVector<BasicBlock*, 8>& exitBlocks);
    
    /// This transformation requires natural loop information & requires that
    /// loop preheaders be inserted into the CFG.  It maintains both of these,
    /// as well as the CFG.  It also requires dominator information.
    ///
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequiredID(LoopSimplifyID);
      AU.addPreservedID(LoopSimplifyID);
      AU.addRequired<LoopInfo>();
      AU.addPreserved<LoopInfo>();
      AU.addRequired<DominatorTree>();
      AU.addPreserved<ScalarEvolution>();
      AU.addPreserved<DominatorTree>();

      // Request DominanceFrontier now, even though LCSSA does
      // not use it. This allows Pass Manager to schedule Dominance
      // Frontier early enough such that one LPPassManager can handle
      // multiple loop transformation passes.
      AU.addRequired<DominanceFrontier>(); 
      AU.addPreserved<DominanceFrontier>();
    }
  private:
    void getLoopValuesUsedOutsideLoop(Loop *L,
                                      SetVector<Instruction*> &AffectedValues);

    Value *GetValueForBlock(DomTreeNode *BB, Instruction *OrigInst,
                            DenseMap<DomTreeNode*, Value*> &Phis);

    /// inLoop - returns true if the given block is within the current loop
    bool inLoop(BasicBlock* B) {
      return std::binary_search(LoopBlocks.begin(), LoopBlocks.end(), B);
    }
  };
}
  
char LCSSA::ID = 0;
static RegisterPass<LCSSA> X("lcssa", "Loop-Closed SSA Form Pass");

Pass *llvm::createLCSSAPass() { return new LCSSA(); }
const PassInfo *const llvm::LCSSAID = &X;

/// runOnFunction - Process all loops in the function, inner-most out.
bool LCSSA::runOnLoop(Loop *L, LPPassManager &LPM) {
  
  LI = &LPM.getAnalysis<LoopInfo>();
  DT = &getAnalysis<DominatorTree>();

  // Speed up queries by creating a sorted list of blocks
  LoopBlocks.clear();
  LoopBlocks.insert(LoopBlocks.end(), L->block_begin(), L->block_end());
  std::sort(LoopBlocks.begin(), LoopBlocks.end());
  
  SetVector<Instruction*> AffectedValues;
  getLoopValuesUsedOutsideLoop(L, AffectedValues);
  
  // If no values are affected, we can save a lot of work, since we know that
  // nothing will be changed.
  if (AffectedValues.empty())
    return false;
  
  SmallVector<BasicBlock*, 8> exitBlocks;
  L->getExitBlocks(exitBlocks);  
  
  // Iterate over all affected values for this loop and insert Phi nodes
  // for them in the appropriate exit blocks
  
  for (SetVector<Instruction*>::iterator I = AffectedValues.begin(),
       E = AffectedValues.end(); I != E; ++I)
    ProcessInstruction(*I, exitBlocks);
  
  assert(L->isLCSSAForm());
  
  return true;
}

/// processInstruction - Given a live-out instruction, insert LCSSA Phi nodes,
/// eliminate all out-of-loop uses.
void LCSSA::ProcessInstruction(Instruction *Instr,
                               const SmallVector<BasicBlock*, 8>& exitBlocks) {
  ++NumLCSSA; // We are applying the transformation

  // Keep track of the blocks that have the value available already.
  DenseMap<DomTreeNode*, Value*> Phis;

  DomTreeNode *InstrNode = DT->getNode(Instr->getParent());

  // Insert the LCSSA phi's into the exit blocks (dominated by the value), and
  // add them to the Phi's map.
  for (SmallVector<BasicBlock*, 8>::const_iterator BBI = exitBlocks.begin(),
      BBE = exitBlocks.end(); BBI != BBE; ++BBI) {
    BasicBlock *BB = *BBI;
    DomTreeNode *ExitBBNode = DT->getNode(BB);
    Value *&Phi = Phis[ExitBBNode];
    if (!Phi && DT->dominates(InstrNode, ExitBBNode)) {
      PHINode *PN = PHINode::Create(Instr->getType(), Instr->getName()+".lcssa",
                                    BB->begin());
      PN->reserveOperandSpace(std::distance(pred_begin(BB), pred_end(BB)));

      // Remember that this phi makes the value alive in this block.
      Phi = PN;

      // Add inputs from inside the loop for this PHI.
      for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI)
        PN->addIncoming(Instr, *PI);
    }
  }
  
  
  // Record all uses of Instr outside the loop.  We need to rewrite these.  The
  // LCSSA phis won't be included because they use the value in the loop.
  for (Value::use_iterator UI = Instr->use_begin(), E = Instr->use_end();
       UI != E;) {
    BasicBlock *UserBB = cast<Instruction>(*UI)->getParent();
    if (PHINode *P = dyn_cast<PHINode>(*UI)) {
      UserBB = P->getIncomingBlock(UI);
    }
    
    // If the user is in the loop, don't rewrite it!
    if (UserBB == Instr->getParent() || inLoop(UserBB)) {
      ++UI;
      continue;
    }
    
    // Otherwise, patch up uses of the value with the appropriate LCSSA Phi,
    // inserting PHI nodes into join points where needed.
    Value *Val = GetValueForBlock(DT->getNode(UserBB), Instr, Phis);
    
    // Preincrement the iterator to avoid invalidating it when we change the
    // value.
    Use &U = UI.getUse();
    ++UI;
    U.set(Val);
  }
}

/// getLoopValuesUsedOutsideLoop - Return any values defined in the loop that
/// are used by instructions outside of it.
void LCSSA::getLoopValuesUsedOutsideLoop(Loop *L,
                                      SetVector<Instruction*> &AffectedValues) {
  // FIXME: For large loops, we may be able to avoid a lot of use-scanning
  // by using dominance information.  In particular, if a block does not
  // dominate any of the loop exits, then none of the values defined in the
  // block could be used outside the loop.
  for (Loop::block_iterator BB = L->block_begin(), E = L->block_end();
       BB != E; ++BB) {
    for (BasicBlock::iterator I = (*BB)->begin(), E = (*BB)->end(); I != E; ++I)
      for (Value::use_iterator UI = I->use_begin(), E = I->use_end(); UI != E;
           ++UI) {
        BasicBlock *UserBB = cast<Instruction>(*UI)->getParent();
        if (PHINode* p = dyn_cast<PHINode>(*UI)) {
          UserBB = p->getIncomingBlock(UI);
        }
        
        if (*BB != UserBB && !inLoop(UserBB)) {
          AffectedValues.insert(I);
          break;
        }
      }
  }
}

/// GetValueForBlock - Get the value to use within the specified basic block.
/// available values are in Phis.
Value *LCSSA::GetValueForBlock(DomTreeNode *BB, Instruction *OrigInst,
                               DenseMap<DomTreeNode*, Value*> &Phis) {
  // If there is no dominator info for this BB, it is unreachable.
  if (BB == 0)
    return UndefValue::get(OrigInst->getType());
                                 
  // If we have already computed this value, return the previously computed val.
  if (Phis.count(BB)) return Phis[BB];

  DomTreeNode *IDom = BB->getIDom();

  // Otherwise, there are two cases: we either have to insert a PHI node or we
  // don't.  We need to insert a PHI node if this block is not dominated by one
  // of the exit nodes from the loop (the loop could have multiple exits, and
  // though the value defined *inside* the loop dominated all its uses, each
  // exit by itself may not dominate all the uses).
  //
  // The simplest way to check for this condition is by checking to see if the
  // idom is in the loop.  If so, we *know* that none of the exit blocks
  // dominate this block.  Note that we *know* that the block defining the
  // original instruction is in the idom chain, because if it weren't, then the
  // original value didn't dominate this use.
  if (!inLoop(IDom->getBlock())) {
    // Idom is not in the loop, we must still be "below" the exit block and must
    // be fully dominated by the value live in the idom.
    Value* val = GetValueForBlock(IDom, OrigInst, Phis);
    Phis.insert(std::make_pair(BB, val));
    return val;
  }
  
  BasicBlock *BBN = BB->getBlock();
  
  // Otherwise, the idom is the loop, so we need to insert a PHI node.  Do so
  // now, then get values to fill in the incoming values for the PHI.
  PHINode *PN = PHINode::Create(OrigInst->getType(),
                                OrigInst->getName() + ".lcssa", BBN->begin());
  PN->reserveOperandSpace(std::distance(pred_begin(BBN), pred_end(BBN)));
  Phis.insert(std::make_pair(BB, PN));
                                 
  // Fill in the incoming values for the block.
  for (pred_iterator PI = pred_begin(BBN), E = pred_end(BBN); PI != E; ++PI)
    PN->addIncoming(GetValueForBlock(DT->getNode(*PI), OrigInst, Phis), *PI);
  return PN;
}

