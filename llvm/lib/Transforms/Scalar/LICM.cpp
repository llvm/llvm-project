//===-- LICM.cpp - Loop Invariant Code Motion Pass ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass performs loop invariant code motion, attempting to remove as much
// code from the body of a loop as possible.  It does this by either hoisting
// code into the preheader block, or by sinking code to the exit blocks if it is
// safe.  This pass also promotes must-aliased memory locations in the loop to
// live in registers, thus hoisting and sinking "invariant" loads and stores.
//
// This pass uses alias analysis for two purposes:
//
//  1. Moving loop invariant loads and calls out of loops.  If we can determine
//     that a load or call inside of a loop never aliases anything stored to,
//     we can hoist it or sink it like any other instruction.
//  2. Scalar Promotion of Memory - If there is a store instruction inside of
//     the loop, we try to move the store to happen AFTER the loop instead of
//     inside of the loop.  This can only happen if a few conditions are true:
//       A. The pointer stored through is loop invariant
//       B. There are no stores or loads in the loop which _may_ alias the
//          pointer.  There are no calls in the loop which mod/ref the pointer.
//     If these conditions are true, we can promote the loads and stores in the
//     loop of the pointer to use a temporary alloca'd variable.  We then use
//     the SSAUpdater to construct the appropriate SSA form for the value.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "licm"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Instructions.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
#include <algorithm>
using namespace llvm;

STATISTIC(NumSunk      , "Number of instructions sunk out of loop");
STATISTIC(NumHoisted   , "Number of instructions hoisted out of loop");
STATISTIC(NumMovedLoads, "Number of load insts hoisted or sunk");
STATISTIC(NumMovedCalls, "Number of call insts hoisted or sunk");
STATISTIC(NumPromoted  , "Number of memory locations promoted to registers");

static cl::opt<bool>
DisablePromotion("disable-licm-promotion", cl::Hidden,
                 cl::desc("Disable memory promotion in LICM pass"));

namespace {
  struct LICM : public LoopPass {
    static char ID; // Pass identification, replacement for typeid
    LICM() : LoopPass(ID) {}

    virtual bool runOnLoop(Loop *L, LPPassManager &LPM);

    /// This transformation requires natural loop information & requires that
    /// loop preheaders be inserted into the CFG...
    ///
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<DominatorTree>();
      AU.addRequired<LoopInfo>();
      AU.addRequiredID(LoopSimplifyID);
      AU.addRequired<AliasAnalysis>();
      AU.addPreserved<AliasAnalysis>();
      AU.addPreserved<ScalarEvolution>();
      AU.addPreservedID(LoopSimplifyID);
    }

    bool doFinalization() {
      assert(LoopToAliasSetMap.empty() && "Didn't free loop alias sets");
      return false;
    }

  private:
    AliasAnalysis *AA;       // Current AliasAnalysis information
    LoopInfo      *LI;       // Current LoopInfo
    DominatorTree *DT;       // Dominator Tree for the current Loop.

    // State that is updated as we process loops.
    bool Changed;            // Set to true when we change anything.
    BasicBlock *Preheader;   // The preheader block of the current loop...
    Loop *CurLoop;           // The current loop we are working on...
    AliasSetTracker *CurAST; // AliasSet information for the current loop...
    DenseMap<Loop*, AliasSetTracker*> LoopToAliasSetMap;

    /// cloneBasicBlockAnalysis - Simple Analysis hook. Clone alias set info.
    void cloneBasicBlockAnalysis(BasicBlock *From, BasicBlock *To, Loop *L);

    /// deleteAnalysisValue - Simple Analysis hook. Delete value V from alias
    /// set.
    void deleteAnalysisValue(Value *V, Loop *L);

    /// SinkRegion - Walk the specified region of the CFG (defined by all blocks
    /// dominated by the specified block, and that are in the current loop) in
    /// reverse depth first order w.r.t the DominatorTree.  This allows us to
    /// visit uses before definitions, allowing us to sink a loop body in one
    /// pass without iteration.
    ///
    void SinkRegion(DomTreeNode *N);

    /// HoistRegion - Walk the specified region of the CFG (defined by all
    /// blocks dominated by the specified block, and that are in the current
    /// loop) in depth first order w.r.t the DominatorTree.  This allows us to
    /// visit definitions before uses, allowing us to hoist a loop body in one
    /// pass without iteration.
    ///
    void HoistRegion(DomTreeNode *N);

    /// inSubLoop - Little predicate that returns true if the specified basic
    /// block is in a subloop of the current one, not the current one itself.
    ///
    bool inSubLoop(BasicBlock *BB) {
      assert(CurLoop->contains(BB) && "Only valid if BB is IN the loop");
      for (Loop::iterator I = CurLoop->begin(), E = CurLoop->end(); I != E; ++I)
        if ((*I)->contains(BB))
          return true;  // A subloop actually contains this block!
      return false;
    }

    /// isExitBlockDominatedByBlockInLoop - This method checks to see if the
    /// specified exit block of the loop is dominated by the specified block
    /// that is in the body of the loop.  We use these constraints to
    /// dramatically limit the amount of the dominator tree that needs to be
    /// searched.
    bool isExitBlockDominatedByBlockInLoop(BasicBlock *ExitBlock,
                                           BasicBlock *BlockInLoop) const {
      // If the block in the loop is the loop header, it must be dominated!
      BasicBlock *LoopHeader = CurLoop->getHeader();
      if (BlockInLoop == LoopHeader)
        return true;

      DomTreeNode *BlockInLoopNode = DT->getNode(BlockInLoop);
      DomTreeNode *IDom            = DT->getNode(ExitBlock);

      // Because the exit block is not in the loop, we know we have to get _at
      // least_ its immediate dominator.
      IDom = IDom->getIDom();
      
      while (IDom && IDom != BlockInLoopNode) {
        // If we have got to the header of the loop, then the instructions block
        // did not dominate the exit node, so we can't hoist it.
        if (IDom->getBlock() == LoopHeader)
          return false;

        // Get next Immediate Dominator.
        IDom = IDom->getIDom();
      };

      return true;
    }

    /// sink - When an instruction is found to only be used outside of the loop,
    /// this function moves it to the exit blocks and patches up SSA form as
    /// needed.
    ///
    void sink(Instruction &I);

    /// hoist - When an instruction is found to only use loop invariant operands
    /// that is safe to hoist, this instruction is called to do the dirty work.
    ///
    void hoist(Instruction &I);

    /// isSafeToExecuteUnconditionally - Only sink or hoist an instruction if it
    /// is not a trapping instruction or if it is a trapping instruction and is
    /// guaranteed to execute.
    ///
    bool isSafeToExecuteUnconditionally(Instruction &I);

    /// pointerInvalidatedByLoop - Return true if the body of this loop may
    /// store into the memory location pointed to by V.
    ///
    bool pointerInvalidatedByLoop(Value *V, unsigned Size) {
      // Check to see if any of the basic blocks in CurLoop invalidate *V.
      return CurAST->getAliasSetForPointer(V, Size).isMod();
    }

    bool canSinkOrHoistInst(Instruction &I);
    bool isLoopInvariantInst(Instruction &I);
    bool isNotUsedInLoop(Instruction &I);

    void PromoteAliasSet(AliasSet &AS);
  };
}

char LICM::ID = 0;
INITIALIZE_PASS(LICM, "licm", "Loop Invariant Code Motion", false, false);

Pass *llvm::createLICMPass() { return new LICM(); }

/// Hoist expressions out of the specified loop. Note, alias info for inner
/// loop is not preserved so it is not a good idea to run LICM multiple 
/// times on one loop.
///
bool LICM::runOnLoop(Loop *L, LPPassManager &LPM) {
  Changed = false;

  // Get our Loop and Alias Analysis information...
  LI = &getAnalysis<LoopInfo>();
  AA = &getAnalysis<AliasAnalysis>();
  DT = &getAnalysis<DominatorTree>();

  CurAST = new AliasSetTracker(*AA);
  // Collect Alias info from subloops.
  for (Loop::iterator LoopItr = L->begin(), LoopItrE = L->end();
       LoopItr != LoopItrE; ++LoopItr) {
    Loop *InnerL = *LoopItr;
    AliasSetTracker *InnerAST = LoopToAliasSetMap[InnerL];
    assert(InnerAST && "Where is my AST?");

    // What if InnerLoop was modified by other passes ?
    CurAST->add(*InnerAST);
    
    // Once we've incorporated the inner loop's AST into ours, we don't need the
    // subloop's anymore.
    delete InnerAST;
    LoopToAliasSetMap.erase(InnerL);
  }
  
  CurLoop = L;

  // Get the preheader block to move instructions into...
  Preheader = L->getLoopPreheader();

  // Loop over the body of this loop, looking for calls, invokes, and stores.
  // Because subloops have already been incorporated into AST, we skip blocks in
  // subloops.
  //
  for (Loop::block_iterator I = L->block_begin(), E = L->block_end();
       I != E; ++I) {
    BasicBlock *BB = *I;
    if (LI->getLoopFor(BB) == L)        // Ignore blocks in subloops.
      CurAST->add(*BB);                 // Incorporate the specified basic block
  }

  // We want to visit all of the instructions in this loop... that are not parts
  // of our subloops (they have already had their invariants hoisted out of
  // their loop, into this loop, so there is no need to process the BODIES of
  // the subloops).
  //
  // Traverse the body of the loop in depth first order on the dominator tree so
  // that we are guaranteed to see definitions before we see uses.  This allows
  // us to sink instructions in one pass, without iteration.  After sinking
  // instructions, we perform another pass to hoist them out of the loop.
  //
  if (L->hasDedicatedExits())
    SinkRegion(DT->getNode(L->getHeader()));
  if (Preheader)
    HoistRegion(DT->getNode(L->getHeader()));

  // Now that all loop invariants have been removed from the loop, promote any
  // memory references to scalars that we can.
  if (!DisablePromotion && Preheader && L->hasDedicatedExits()) {
    // Loop over all of the alias sets in the tracker object.
    for (AliasSetTracker::iterator I = CurAST->begin(), E = CurAST->end();
         I != E; ++I)
      PromoteAliasSet(*I);
  }
  
  // Clear out loops state information for the next iteration
  CurLoop = 0;
  Preheader = 0;

  // If this loop is nested inside of another one, save the alias information
  // for when we process the outer loop.
  if (L->getParentLoop())
    LoopToAliasSetMap[L] = CurAST;
  else
    delete CurAST;
  return Changed;
}

/// SinkRegion - Walk the specified region of the CFG (defined by all blocks
/// dominated by the specified block, and that are in the current loop) in
/// reverse depth first order w.r.t the DominatorTree.  This allows us to visit
/// uses before definitions, allowing us to sink a loop body in one pass without
/// iteration.
///
void LICM::SinkRegion(DomTreeNode *N) {
  assert(N != 0 && "Null dominator tree node?");
  BasicBlock *BB = N->getBlock();

  // If this subregion is not in the top level loop at all, exit.
  if (!CurLoop->contains(BB)) return;

  // We are processing blocks in reverse dfo, so process children first.
  const std::vector<DomTreeNode*> &Children = N->getChildren();
  for (unsigned i = 0, e = Children.size(); i != e; ++i)
    SinkRegion(Children[i]);

  // Only need to process the contents of this block if it is not part of a
  // subloop (which would already have been processed).
  if (inSubLoop(BB)) return;

  for (BasicBlock::iterator II = BB->end(); II != BB->begin(); ) {
    Instruction &I = *--II;
    
    // If the instruction is dead, we would try to sink it because it isn't used
    // in the loop, instead, just delete it.
    if (isInstructionTriviallyDead(&I)) {
      DEBUG(dbgs() << "LICM deleting dead inst: " << I << '\n');
      ++II;
      CurAST->deleteValue(&I);
      I.eraseFromParent();
      Changed = true;
      continue;
    }

    // Check to see if we can sink this instruction to the exit blocks
    // of the loop.  We can do this if the all users of the instruction are
    // outside of the loop.  In this case, it doesn't even matter if the
    // operands of the instruction are loop invariant.
    //
    if (isNotUsedInLoop(I) && canSinkOrHoistInst(I)) {
      ++II;
      sink(I);
    }
  }
}

/// HoistRegion - Walk the specified region of the CFG (defined by all blocks
/// dominated by the specified block, and that are in the current loop) in depth
/// first order w.r.t the DominatorTree.  This allows us to visit definitions
/// before uses, allowing us to hoist a loop body in one pass without iteration.
///
void LICM::HoistRegion(DomTreeNode *N) {
  assert(N != 0 && "Null dominator tree node?");
  BasicBlock *BB = N->getBlock();

  // If this subregion is not in the top level loop at all, exit.
  if (!CurLoop->contains(BB)) return;

  // Only need to process the contents of this block if it is not part of a
  // subloop (which would already have been processed).
  if (!inSubLoop(BB))
    for (BasicBlock::iterator II = BB->begin(), E = BB->end(); II != E; ) {
      Instruction &I = *II++;

      // Try constant folding this instruction.  If all the operands are
      // constants, it is technically hoistable, but it would be better to just
      // fold it.
      if (Constant *C = ConstantFoldInstruction(&I)) {
        DEBUG(dbgs() << "LICM folding inst: " << I << "  --> " << *C << '\n');
        CurAST->copyValue(&I, C);
        CurAST->deleteValue(&I);
        I.replaceAllUsesWith(C);
        I.eraseFromParent();
        continue;
      }
      
      // Try hoisting the instruction out to the preheader.  We can only do this
      // if all of the operands of the instruction are loop invariant and if it
      // is safe to hoist the instruction.
      //
      if (isLoopInvariantInst(I) && canSinkOrHoistInst(I) &&
          isSafeToExecuteUnconditionally(I))
        hoist(I);
    }

  const std::vector<DomTreeNode*> &Children = N->getChildren();
  for (unsigned i = 0, e = Children.size(); i != e; ++i)
    HoistRegion(Children[i]);
}

/// canSinkOrHoistInst - Return true if the hoister and sinker can handle this
/// instruction.
///
bool LICM::canSinkOrHoistInst(Instruction &I) {
  // Loads have extra constraints we have to verify before we can hoist them.
  if (LoadInst *LI = dyn_cast<LoadInst>(&I)) {
    if (LI->isVolatile())
      return false;        // Don't hoist volatile loads!

    // Loads from constant memory are always safe to move, even if they end up
    // in the same alias set as something that ends up being modified.
    if (AA->pointsToConstantMemory(LI->getOperand(0)))
      return true;
    
    // Don't hoist loads which have may-aliased stores in loop.
    unsigned Size = 0;
    if (LI->getType()->isSized())
      Size = AA->getTypeStoreSize(LI->getType());
    return !pointerInvalidatedByLoop(LI->getOperand(0), Size);
  } else if (CallInst *CI = dyn_cast<CallInst>(&I)) {
    // Handle obvious cases efficiently.
    AliasAnalysis::ModRefBehavior Behavior = AA->getModRefBehavior(CI);
    if (Behavior == AliasAnalysis::DoesNotAccessMemory)
      return true;
    else if (Behavior == AliasAnalysis::OnlyReadsMemory) {
      // If this call only reads from memory and there are no writes to memory
      // in the loop, we can hoist or sink the call as appropriate.
      bool FoundMod = false;
      for (AliasSetTracker::iterator I = CurAST->begin(), E = CurAST->end();
           I != E; ++I) {
        AliasSet &AS = *I;
        if (!AS.isForwardingAliasSet() && AS.isMod()) {
          FoundMod = true;
          break;
        }
      }
      if (!FoundMod) return true;
    }

    // FIXME: This should use mod/ref information to see if we can hoist or sink
    // the call.

    return false;
  }

  // Otherwise these instructions are hoistable/sinkable
  return isa<BinaryOperator>(I) || isa<CastInst>(I) ||
         isa<SelectInst>(I) || isa<GetElementPtrInst>(I) || isa<CmpInst>(I) ||
         isa<InsertElementInst>(I) || isa<ExtractElementInst>(I) ||
         isa<ShuffleVectorInst>(I);
}

/// isNotUsedInLoop - Return true if the only users of this instruction are
/// outside of the loop.  If this is true, we can sink the instruction to the
/// exit blocks of the loop.
///
bool LICM::isNotUsedInLoop(Instruction &I) {
  for (Value::use_iterator UI = I.use_begin(), E = I.use_end(); UI != E; ++UI) {
    Instruction *User = cast<Instruction>(*UI);
    if (PHINode *PN = dyn_cast<PHINode>(User)) {
      // PHI node uses occur in predecessor blocks!
      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
        if (PN->getIncomingValue(i) == &I)
          if (CurLoop->contains(PN->getIncomingBlock(i)))
            return false;
    } else if (CurLoop->contains(User)) {
      return false;
    }
  }
  return true;
}


/// isLoopInvariantInst - Return true if all operands of this instruction are
/// loop invariant.  We also filter out non-hoistable instructions here just for
/// efficiency.
///
bool LICM::isLoopInvariantInst(Instruction &I) {
  // The instruction is loop invariant if all of its operands are loop-invariant
  for (unsigned i = 0, e = I.getNumOperands(); i != e; ++i)
    if (!CurLoop->isLoopInvariant(I.getOperand(i)))
      return false;

  // If we got this far, the instruction is loop invariant!
  return true;
}

/// sink - When an instruction is found to only be used outside of the loop,
/// this function moves it to the exit blocks and patches up SSA form as needed.
/// This method is guaranteed to remove the original instruction from its
/// position, and may either delete it or move it to outside of the loop.
///
void LICM::sink(Instruction &I) {
  DEBUG(dbgs() << "LICM sinking instruction: " << I << "\n");

  SmallVector<BasicBlock*, 8> ExitBlocks;
  CurLoop->getUniqueExitBlocks(ExitBlocks);

  if (isa<LoadInst>(I)) ++NumMovedLoads;
  else if (isa<CallInst>(I)) ++NumMovedCalls;
  ++NumSunk;
  Changed = true;

  // The case where there is only a single exit node of this loop is common
  // enough that we handle it as a special (more efficient) case.  It is more
  // efficient to handle because there are no PHI nodes that need to be placed.
  if (ExitBlocks.size() == 1) {
    if (!isExitBlockDominatedByBlockInLoop(ExitBlocks[0], I.getParent())) {
      // Instruction is not used, just delete it.
      CurAST->deleteValue(&I);
      // If I has users in unreachable blocks, eliminate.
      // If I is not void type then replaceAllUsesWith undef.
      // This allows ValueHandlers and custom metadata to adjust itself.
      if (!I.use_empty())
        I.replaceAllUsesWith(UndefValue::get(I.getType()));
      I.eraseFromParent();
    } else {
      // Move the instruction to the start of the exit block, after any PHI
      // nodes in it.
      I.moveBefore(ExitBlocks[0]->getFirstNonPHI());

      // This instruction is no longer in the AST for the current loop, because
      // we just sunk it out of the loop.  If we just sunk it into an outer
      // loop, we will rediscover the operation when we process it.
      CurAST->deleteValue(&I);
    }
    return;
  }
  
  if (ExitBlocks.empty()) {
    // The instruction is actually dead if there ARE NO exit blocks.
    CurAST->deleteValue(&I);
    // If I has users in unreachable blocks, eliminate.
    // If I is not void type then replaceAllUsesWith undef.
    // This allows ValueHandlers and custom metadata to adjust itself.
    if (!I.use_empty())
      I.replaceAllUsesWith(UndefValue::get(I.getType()));
    I.eraseFromParent();
    return;
  }
  
  // Otherwise, if we have multiple exits, use the SSAUpdater to do all of the
  // hard work of inserting PHI nodes as necessary.
  SmallVector<PHINode*, 8> NewPHIs;
  SSAUpdater SSA(&NewPHIs);
  
  if (!I.use_empty())
    SSA.Initialize(I.getType(), I.getName());
  
  // Insert a copy of the instruction in each exit block of the loop that is
  // dominated by the instruction.  Each exit block is known to only be in the
  // ExitBlocks list once.
  BasicBlock *InstOrigBB = I.getParent();
  unsigned NumInserted = 0;
  
  for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i) {
    BasicBlock *ExitBlock = ExitBlocks[i];
    
    if (!isExitBlockDominatedByBlockInLoop(ExitBlock, InstOrigBB))
      continue;
    
    // Insert the code after the last PHI node.
    BasicBlock::iterator InsertPt = ExitBlock->getFirstNonPHI();
    
    // If this is the first exit block processed, just move the original
    // instruction, otherwise clone the original instruction and insert
    // the copy.
    Instruction *New;
    if (NumInserted++ == 0) {
      I.moveBefore(InsertPt);
      New = &I;
    } else {
      New = I.clone();
      if (!I.getName().empty())
        New->setName(I.getName()+".le");
      ExitBlock->getInstList().insert(InsertPt, New);
    }
    
    // Now that we have inserted the instruction, inform SSAUpdater.
    if (!I.use_empty())
      SSA.AddAvailableValue(ExitBlock, New);
  }
  
  // If the instruction doesn't dominate any exit blocks, it must be dead.
  if (NumInserted == 0) {
    CurAST->deleteValue(&I);
    if (!I.use_empty())
      I.replaceAllUsesWith(UndefValue::get(I.getType()));
    I.eraseFromParent();
    return;
  }
  
  // Next, rewrite uses of the instruction, inserting PHI nodes as needed.
  for (Value::use_iterator UI = I.use_begin(), UE = I.use_end(); UI != UE; ) {
    // Grab the use before incrementing the iterator.
    Use &U = UI.getUse();
    // Increment the iterator before removing the use from the list.
    ++UI;
    SSA.RewriteUseAfterInsertions(U);
  }
  
  // Update CurAST for NewPHIs if I had pointer type.
  if (I.getType()->isPointerTy())
    for (unsigned i = 0, e = NewPHIs.size(); i != e; ++i)
      CurAST->copyValue(&I, NewPHIs[i]);
  
  // Finally, remove the instruction from CurAST.  It is no longer in the loop.
  CurAST->deleteValue(&I);
}

/// hoist - When an instruction is found to only use loop invariant operands
/// that is safe to hoist, this instruction is called to do the dirty work.
///
void LICM::hoist(Instruction &I) {
  DEBUG(dbgs() << "LICM hoisting to " << Preheader->getName() << ": "
        << I << "\n");

  // Move the new node to the Preheader, before its terminator.
  I.moveBefore(Preheader->getTerminator());

  if (isa<LoadInst>(I)) ++NumMovedLoads;
  else if (isa<CallInst>(I)) ++NumMovedCalls;
  ++NumHoisted;
  Changed = true;
}

/// isSafeToExecuteUnconditionally - Only sink or hoist an instruction if it is
/// not a trapping instruction or if it is a trapping instruction and is
/// guaranteed to execute.
///
bool LICM::isSafeToExecuteUnconditionally(Instruction &Inst) {
  // If it is not a trapping instruction, it is always safe to hoist.
  if (Inst.isSafeToSpeculativelyExecute())
    return true;

  // Otherwise we have to check to make sure that the instruction dominates all
  // of the exit blocks.  If it doesn't, then there is a path out of the loop
  // which does not execute this instruction, so we can't hoist it.

  // If the instruction is in the header block for the loop (which is very
  // common), it is always guaranteed to dominate the exit blocks.  Since this
  // is a common case, and can save some work, check it now.
  if (Inst.getParent() == CurLoop->getHeader())
    return true;

  // Get the exit blocks for the current loop.
  SmallVector<BasicBlock*, 8> ExitBlocks;
  CurLoop->getExitBlocks(ExitBlocks);

  // For each exit block, get the DT node and walk up the DT until the
  // instruction's basic block is found or we exit the loop.
  for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i)
    if (!isExitBlockDominatedByBlockInLoop(ExitBlocks[i], Inst.getParent()))
      return false;

  return true;
}

/// PromoteAliasSet - Try to promote memory values to scalars by sinking
/// stores out of the loop and moving loads to before the loop.  We do this by
/// looping over the stores in the loop, looking for stores to Must pointers
/// which are loop invariant.
///
void LICM::PromoteAliasSet(AliasSet &AS) {
  // We can promote this alias set if it has a store, if it is a "Must" alias
  // set, if the pointer is loop invariant, and if we are not eliminating any
  // volatile loads or stores.
  if (AS.isForwardingAliasSet() || !AS.isMod() || !AS.isMustAlias() ||
      AS.isVolatile() || !CurLoop->isLoopInvariant(AS.begin()->getValue()))
    return;
  
  assert(!AS.empty() &&
         "Must alias set should have at least one pointer element in it!");
  Value *SomePtr = AS.begin()->getValue();

  // It isn't safe to promote a load/store from the loop if the load/store is
  // conditional.  For example, turning:
  //
  //    for () { if (c) *P += 1; }
  //
  // into:
  //
  //    tmp = *P;  for () { if (c) tmp +=1; } *P = tmp;
  //
  // is not safe, because *P may only be valid to access if 'c' is true.
  // 
  // It is safe to promote P if all uses are direct load/stores and if at
  // least one is guaranteed to be executed.
  bool GuaranteedToExecute = false;
  
  SmallVector<Instruction*, 64> LoopUses;
  SmallPtrSet<Value*, 4> PointerMustAliases;

  // Check that all of the pointers in the alias set have the same type.  We
  // cannot (yet) promote a memory location that is loaded and stored in
  // different sizes.
  for (AliasSet::iterator ASI = AS.begin(), E = AS.end(); ASI != E; ++ASI) {
    Value *ASIV = ASI->getValue();
    PointerMustAliases.insert(ASIV);
    
    // Check that all of the pointers in the alias set have the same type.  We
    // cannot (yet) promote a memory location that is loaded and stored in
    // different sizes.
    if (SomePtr->getType() != ASIV->getType())
      return;
    
    for (Value::use_iterator UI = ASIV->use_begin(), UE = ASIV->use_end();
         UI != UE; ++UI) {
      // Ignore instructions that are outside the loop.
      Instruction *Use = dyn_cast<Instruction>(*UI);
      if (!Use || !CurLoop->contains(Use))
        continue;
      
      // If there is an non-load/store instruction in the loop, we can't promote
      // it.
      if (isa<LoadInst>(Use))
        assert(!cast<LoadInst>(Use)->isVolatile() && "AST broken");
      else if (isa<StoreInst>(Use)) {
        assert(!cast<StoreInst>(Use)->isVolatile() && "AST broken");
        if (Use->getOperand(0) == ASIV) return;
      } else
        return; // Not a load or store.
      
      if (!GuaranteedToExecute)
        GuaranteedToExecute = isSafeToExecuteUnconditionally(*Use);
      
      LoopUses.push_back(Use);
    }
  }
  
  // If there isn't a guaranteed-to-execute instruction, we can't promote.
  if (!GuaranteedToExecute)
    return;
  
  // Otherwise, this is safe to promote, lets do it!
  DEBUG(dbgs() << "LICM: Promoting value stored to in loop: " <<*SomePtr<<'\n');  
  Changed = true;
  ++NumPromoted;

  // We use the SSAUpdater interface to insert phi nodes as required.
  SmallVector<PHINode*, 16> NewPHIs;
  SSAUpdater SSA(&NewPHIs);
  
  // It wants to know some value of the same type as what we'll be inserting.
  Value *SomeValue;
  if (isa<LoadInst>(LoopUses[0]))
    SomeValue = LoopUses[0];
  else
    SomeValue = cast<StoreInst>(LoopUses[0])->getOperand(0);
  SSA.Initialize(SomeValue->getType(), SomeValue->getName());

  // First step: bucket up uses of the pointers by the block they occur in.
  // This is important because we have to handle multiple defs/uses in a block
  // ourselves: SSAUpdater is purely for cross-block references.
  // FIXME: Want a TinyVector<Instruction*> since there is usually 0/1 element.
  DenseMap<BasicBlock*, std::vector<Instruction*> > UsesByBlock;
  for (unsigned i = 0, e = LoopUses.size(); i != e; ++i) {
    Instruction *User = LoopUses[i];
    UsesByBlock[User->getParent()].push_back(User);
  }
  
  // Okay, now we can iterate over all the blocks in the loop with uses,
  // processing them.  Keep track of which loads are loading a live-in value.
  SmallVector<LoadInst*, 32> LiveInLoads;
  DenseMap<Value*, Value*> ReplacedLoads;
  
  for (unsigned LoopUse = 0, e = LoopUses.size(); LoopUse != e; ++LoopUse) {
    Instruction *User = LoopUses[LoopUse];
    std::vector<Instruction*> &BlockUses = UsesByBlock[User->getParent()];
    
    // If this block has already been processed, ignore this repeat use.
    if (BlockUses.empty()) continue;
    
    // Okay, this is the first use in the block.  If this block just has a
    // single user in it, we can rewrite it trivially.
    if (BlockUses.size() == 1) {
      // If it is a store, it is a trivial def of the value in the block.
      if (isa<StoreInst>(User)) {
        SSA.AddAvailableValue(User->getParent(),
                              cast<StoreInst>(User)->getOperand(0));
      } else {
        // Otherwise it is a load, queue it to rewrite as a live-in load.
        LiveInLoads.push_back(cast<LoadInst>(User));
      }
      BlockUses.clear();
      continue;
    }
    
    // Otherwise, check to see if this block is all loads.  If so, we can queue
    // them all as live in loads.
    bool HasStore = false;
    for (unsigned i = 0, e = BlockUses.size(); i != e; ++i) {
      if (isa<StoreInst>(BlockUses[i])) {
        HasStore = true;
        break;
      }
    }
    
    if (!HasStore) {
      for (unsigned i = 0, e = BlockUses.size(); i != e; ++i)
        LiveInLoads.push_back(cast<LoadInst>(BlockUses[i]));
      BlockUses.clear();
      continue;
    }

    // Otherwise, we have mixed loads and stores (or just a bunch of stores).
    // Since SSAUpdater is purely for cross-block values, we need to determine
    // the order of these instructions in the block.  If the first use in the
    // block is a load, then it uses the live in value.  The last store defines
    // the live out value.  We handle this by doing a linear scan of the block.
    BasicBlock *BB = User->getParent();
    Value *StoredValue = 0;
    for (BasicBlock::iterator II = BB->begin(), E = BB->end(); II != E; ++II) {
      if (LoadInst *L = dyn_cast<LoadInst>(II)) {
        // If this is a load from an unrelated pointer, ignore it.
        if (!PointerMustAliases.count(L->getOperand(0))) continue;

        // If we haven't seen a store yet, this is a live in use, otherwise
        // use the stored value.
        if (StoredValue) {
          L->replaceAllUsesWith(StoredValue);
          ReplacedLoads[L] = StoredValue;
        } else {
          LiveInLoads.push_back(L);
        }
        continue;
      }
      
      if (StoreInst *S = dyn_cast<StoreInst>(II)) {
        // If this is a store to an unrelated pointer, ignore it.
        if (!PointerMustAliases.count(S->getOperand(1))) continue;

        // Remember that this is the active value in the block.
        StoredValue = S->getOperand(0);
      }
    }
    
    // The last stored value that happened is the live-out for the block.
    assert(StoredValue && "Already checked that there is a store in block");
    SSA.AddAvailableValue(BB, StoredValue);
    BlockUses.clear();
  }
  
  // Now that all the intra-loop values are classified, set up the preheader.
  // It gets a load of the pointer we're promoting, and it is the live-out value
  // from the preheader.
  LoadInst *PreheaderLoad = new LoadInst(SomePtr,SomePtr->getName()+".promoted",
                                         Preheader->getTerminator());
  SSA.AddAvailableValue(Preheader, PreheaderLoad);

  // Now that the preheader is good to go, set up the exit blocks.  Each exit
  // block gets a store of the live-out values that feed them.  Since we've
  // already told the SSA updater about the defs in the loop and the preheader
  // definition, it is all set and we can start using it.
  SmallVector<BasicBlock*, 8> ExitBlocks;
  CurLoop->getUniqueExitBlocks(ExitBlocks);
  for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i) {
    BasicBlock *ExitBlock = ExitBlocks[i];
    Value *LiveInValue = SSA.GetValueInMiddleOfBlock(ExitBlock);
    Instruction *InsertPos = ExitBlock->getFirstNonPHI();
    new StoreInst(LiveInValue, SomePtr, InsertPos);
  }

  // Okay, now we rewrite all loads that use live-in values in the loop,
  // inserting PHI nodes as necessary.
  for (unsigned i = 0, e = LiveInLoads.size(); i != e; ++i) {
    LoadInst *ALoad = LiveInLoads[i];
    Value *NewVal = SSA.GetValueInMiddleOfBlock(ALoad->getParent());
    ALoad->replaceAllUsesWith(NewVal);
    CurAST->copyValue(ALoad, NewVal);
    ReplacedLoads[ALoad] = NewVal;
  }
  
  // If the preheader load is itself a pointer, we need to tell alias analysis
  // about the new pointer we created in the preheader block and about any PHI
  // nodes that just got inserted.
  if (PreheaderLoad->getType()->isPointerTy()) {
    // Copy any value stored to or loaded from a must-alias of the pointer.
    CurAST->copyValue(SomeValue, PreheaderLoad);
    
    for (unsigned i = 0, e = NewPHIs.size(); i != e; ++i)
      CurAST->copyValue(SomeValue, NewPHIs[i]);
  }
  
  // Now that everything is rewritten, delete the old instructions from the body
  // of the loop.  They should all be dead now.
  for (unsigned i = 0, e = LoopUses.size(); i != e; ++i) {
    Instruction *User = LoopUses[i];
    
    // If this is a load that still has uses, then the load must have been added
    // as a live value in the SSAUpdate data structure for a block (e.g. because
    // the loaded value was stored later).  In this case, we need to recursively
    // propagate the updates until we get to the real value.
    if (!User->use_empty()) {
      Value *NewVal = ReplacedLoads[User];
      assert(NewVal && "not a replaced load?");
      
      // Propagate down to the ultimate replacee.  The intermediately loads
      // could theoretically already have been deleted, so we don't want to
      // dereference the Value*'s.
      DenseMap<Value*, Value*>::iterator RLI = ReplacedLoads.find(NewVal);
      while (RLI != ReplacedLoads.end()) {
        NewVal = RLI->second;
        RLI = ReplacedLoads.find(NewVal);
      }
      
      User->replaceAllUsesWith(NewVal);
      CurAST->copyValue(User, NewVal);
    }
    
    CurAST->deleteValue(User);
    User->eraseFromParent();
  }
  
  // fwew, we're done!
}


/// cloneBasicBlockAnalysis - Simple Analysis hook. Clone alias set info.
void LICM::cloneBasicBlockAnalysis(BasicBlock *From, BasicBlock *To, Loop *L) {
  AliasSetTracker *AST = LoopToAliasSetMap.lookup(L);
  if (!AST)
    return;

  AST->copyValue(From, To);
}

/// deleteAnalysisValue - Simple Analysis hook. Delete value V from alias
/// set.
void LICM::deleteAnalysisValue(Value *V, Loop *L) {
  AliasSetTracker *AST = LoopToAliasSetMap.lookup(L);
  if (!AST)
    return;

  AST->deleteValue(V);
}
