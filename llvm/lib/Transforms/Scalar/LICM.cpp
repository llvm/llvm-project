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
//     the mem2reg functionality to construct the appropriate SSA form for the
//     variable.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "licm"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/CommandLine.h"
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

// This feature is currently disabled by default because CodeGen is not yet capable
// of rematerializing these constants in PIC mode, so it can lead to degraded
// performance. Compile test/CodeGen/X86/remat-constant.ll with
// -relocation-model=pic to see an example of this.
static cl::opt<bool>
EnableLICMConstantMotion("enable-licm-constant-variables", cl::Hidden,
                         cl::desc("Enable hoisting/sinking of constant "
                                  "global variables"));

namespace {
  struct VISIBILITY_HIDDEN LICM : public LoopPass {
    static char ID; // Pass identification, replacement for typeid
    LICM() : LoopPass(&ID) {}

    virtual bool runOnLoop(Loop *L, LPPassManager &LPM);

    /// This transformation requires natural loop information & requires that
    /// loop preheaders be inserted into the CFG...
    ///
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequiredID(LoopSimplifyID);
      AU.addRequired<LoopInfo>();
      AU.addRequired<DominatorTree>();
      AU.addRequired<DominanceFrontier>();  // For scalar promotion (mem2reg)
      AU.addRequired<AliasAnalysis>();
      AU.addPreserved<ScalarEvolution>();
      AU.addPreserved<DominanceFrontier>();
    }

    bool doFinalization() {
      // Free the values stored in the map
      for (std::map<Loop *, AliasSetTracker *>::iterator
             I = LoopToAliasMap.begin(), E = LoopToAliasMap.end(); I != E; ++I)
        delete I->second;

      LoopToAliasMap.clear();
      return false;
    }

  private:
    // Various analyses that we use...
    AliasAnalysis *AA;       // Current AliasAnalysis information
    LoopInfo      *LI;       // Current LoopInfo
    DominatorTree *DT;       // Dominator Tree for the current Loop...
    DominanceFrontier *DF;   // Current Dominance Frontier

    // State that is updated as we process loops
    bool Changed;            // Set to true when we change anything.
    BasicBlock *Preheader;   // The preheader block of the current loop...
    Loop *CurLoop;           // The current loop we are working on...
    AliasSetTracker *CurAST; // AliasSet information for the current loop...
    std::map<Loop *, AliasSetTracker *> LoopToAliasMap;

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
      do {
        // Get next Immediate Dominator.
        IDom = IDom->getIDom();

        // If we have got to the header of the loop, then the instructions block
        // did not dominate the exit node, so we can't hoist it.
        if (IDom->getBlock() == LoopHeader)
          return false;

      } while (IDom != BlockInLoopNode);

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

    /// PromoteValuesInLoop - Look at the stores in the loop and promote as many
    /// to scalars as we can.
    ///
    void PromoteValuesInLoop();

    /// FindPromotableValuesInLoop - Check the current loop for stores to
    /// definite pointers, which are not loaded and stored through may aliases.
    /// If these are found, create an alloca for the value, add it to the
    /// PromotedValues list, and keep track of the mapping from value to
    /// alloca...
    ///
    void FindPromotableValuesInLoop(
                   std::vector<std::pair<AllocaInst*, Value*> > &PromotedValues,
                                    std::map<Value*, AllocaInst*> &Val2AlMap);
  };
}

char LICM::ID = 0;
static RegisterPass<LICM> X("licm", "Loop Invariant Code Motion");

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
  DF = &getAnalysis<DominanceFrontier>();
  DT = &getAnalysis<DominatorTree>();

  CurAST = new AliasSetTracker(*AA);
  // Collect Alias info from subloops
  for (Loop::iterator LoopItr = L->begin(), LoopItrE = L->end();
       LoopItr != LoopItrE; ++LoopItr) {
    Loop *InnerL = *LoopItr;
    AliasSetTracker *InnerAST = LoopToAliasMap[InnerL];
    assert (InnerAST && "Where is my AST?");

    // What if InnerLoop was modified by other passes ?
    CurAST->add(*InnerAST);
  }
  
  CurLoop = L;

  // Get the preheader block to move instructions into...
  Preheader = L->getLoopPreheader();
  assert(Preheader&&"Preheader insertion pass guarantees we have a preheader!");

  // Loop over the body of this loop, looking for calls, invokes, and stores.
  // Because subloops have already been incorporated into AST, we skip blocks in
  // subloops.
  //
  for (Loop::block_iterator I = L->block_begin(), E = L->block_end();
       I != E; ++I) {
    BasicBlock *BB = *I;
    if (LI->getLoopFor(BB) == L)        // Ignore blocks in subloops...
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
  SinkRegion(DT->getNode(L->getHeader()));
  HoistRegion(DT->getNode(L->getHeader()));

  // Now that all loop invariants have been removed from the loop, promote any
  // memory references to scalars that we can...
  if (!DisablePromotion)
    PromoteValuesInLoop();

  // Clear out loops state information for the next iteration
  CurLoop = 0;
  Preheader = 0;

  LoopToAliasMap[L] = CurAST;
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

  // We are processing blocks in reverse dfo, so process children first...
  const std::vector<DomTreeNode*> &Children = N->getChildren();
  for (unsigned i = 0, e = Children.size(); i != e; ++i)
    SinkRegion(Children[i]);

  // Only need to process the contents of this block if it is not part of a
  // subloop (which would already have been processed).
  if (inSubLoop(BB)) return;

  for (BasicBlock::iterator II = BB->end(); II != BB->begin(); ) {
    Instruction &I = *--II;

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
    if (EnableLICMConstantMotion &&
        AA->pointsToConstantMemory(LI->getOperand(0)))
      return true;
    
    // Don't hoist loads which have may-aliased stores in loop.
    unsigned Size = 0;
    if (LI->getType()->isSized())
      Size = AA->getTargetData().getTypeStoreSize(LI->getType());
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
    } else if (CurLoop->contains(User->getParent())) {
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
  DOUT << "LICM sinking instruction: " << I;

  SmallVector<BasicBlock*, 8> ExitBlocks;
  CurLoop->getExitBlocks(ExitBlocks);

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
      if (!I.use_empty())  // If I has users in unreachable blocks, eliminate.
        I.replaceAllUsesWith(UndefValue::get(I.getType()));
      I.eraseFromParent();
    } else {
      // Move the instruction to the start of the exit block, after any PHI
      // nodes in it.
      I.removeFromParent();

      BasicBlock::iterator InsertPt = ExitBlocks[0]->getFirstNonPHI();
      ExitBlocks[0]->getInstList().insert(InsertPt, &I);
    }
  } else if (ExitBlocks.empty()) {
    // The instruction is actually dead if there ARE NO exit blocks.
    CurAST->deleteValue(&I);
    if (!I.use_empty())  // If I has users in unreachable blocks, eliminate.
      I.replaceAllUsesWith(UndefValue::get(I.getType()));
    I.eraseFromParent();
  } else {
    // Otherwise, if we have multiple exits, use the PromoteMem2Reg function to
    // do all of the hard work of inserting PHI nodes as necessary.  We convert
    // the value into a stack object to get it to do this.

    // Firstly, we create a stack object to hold the value...
    AllocaInst *AI = 0;

    if (I.getType() != Type::VoidTy) {
      AI = new AllocaInst(I.getType(), 0, I.getName(),
                          I.getParent()->getParent()->getEntryBlock().begin());
      CurAST->add(AI);
    }

    // Secondly, insert load instructions for each use of the instruction
    // outside of the loop.
    while (!I.use_empty()) {
      Instruction *U = cast<Instruction>(I.use_back());

      // If the user is a PHI Node, we actually have to insert load instructions
      // in all predecessor blocks, not in the PHI block itself!
      if (PHINode *UPN = dyn_cast<PHINode>(U)) {
        // Only insert into each predecessor once, so that we don't have
        // different incoming values from the same block!
        std::map<BasicBlock*, Value*> InsertedBlocks;
        for (unsigned i = 0, e = UPN->getNumIncomingValues(); i != e; ++i)
          if (UPN->getIncomingValue(i) == &I) {
            BasicBlock *Pred = UPN->getIncomingBlock(i);
            Value *&PredVal = InsertedBlocks[Pred];
            if (!PredVal) {
              // Insert a new load instruction right before the terminator in
              // the predecessor block.
              PredVal = new LoadInst(AI, "", Pred->getTerminator());
              CurAST->add(cast<LoadInst>(PredVal));
            }

            UPN->setIncomingValue(i, PredVal);
          }

      } else {
        LoadInst *L = new LoadInst(AI, "", U);
        U->replaceUsesOfWith(&I, L);
        CurAST->add(L);
      }
    }

    // Thirdly, insert a copy of the instruction in each exit block of the loop
    // that is dominated by the instruction, storing the result into the memory
    // location.  Be careful not to insert the instruction into any particular
    // basic block more than once.
    std::set<BasicBlock*> InsertedBlocks;
    BasicBlock *InstOrigBB = I.getParent();

    for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i) {
      BasicBlock *ExitBlock = ExitBlocks[i];

      if (isExitBlockDominatedByBlockInLoop(ExitBlock, InstOrigBB)) {
        // If we haven't already processed this exit block, do so now.
        if (InsertedBlocks.insert(ExitBlock).second) {
          // Insert the code after the last PHI node...
          BasicBlock::iterator InsertPt = ExitBlock->getFirstNonPHI();

          // If this is the first exit block processed, just move the original
          // instruction, otherwise clone the original instruction and insert
          // the copy.
          Instruction *New;
          if (InsertedBlocks.size() == 1) {
            I.removeFromParent();
            ExitBlock->getInstList().insert(InsertPt, &I);
            New = &I;
          } else {
            New = I.clone();
            CurAST->copyValue(&I, New);
            if (!I.getName().empty())
              New->setName(I.getName()+".le");
            ExitBlock->getInstList().insert(InsertPt, New);
          }

          // Now that we have inserted the instruction, store it into the alloca
          if (AI) new StoreInst(New, AI, InsertPt);
        }
      }
    }

    // If the instruction doesn't dominate any exit blocks, it must be dead.
    if (InsertedBlocks.empty()) {
      CurAST->deleteValue(&I);
      I.eraseFromParent();
    }

    // Finally, promote the fine value to SSA form.
    if (AI) {
      std::vector<AllocaInst*> Allocas;
      Allocas.push_back(AI);
      PromoteMemToReg(Allocas, *DT, *DF, CurAST);
    }
  }
}

/// hoist - When an instruction is found to only use loop invariant operands
/// that is safe to hoist, this instruction is called to do the dirty work.
///
void LICM::hoist(Instruction &I) {
  DOUT << "LICM hoisting to " << Preheader->getName() << ": " << I;

  // Remove the instruction from its current basic block... but don't delete the
  // instruction.
  I.removeFromParent();

  // Insert the new node in Preheader, before the terminator.
  Preheader->getInstList().insert(Preheader->getTerminator(), &I);

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
  if (!Inst.isTrapping()) return true;

  // Otherwise we have to check to make sure that the instruction dominates all
  // of the exit blocks.  If it doesn't, then there is a path out of the loop
  // which does not execute this instruction, so we can't hoist it.

  // If the instruction is in the header block for the loop (which is very
  // common), it is always guaranteed to dominate the exit blocks.  Since this
  // is a common case, and can save some work, check it now.
  if (Inst.getParent() == CurLoop->getHeader())
    return true;

  // It's always safe to load from a global or alloca.
  if (isa<LoadInst>(Inst))
    if (isa<AllocationInst>(Inst.getOperand(0)) ||
        isa<GlobalVariable>(Inst.getOperand(0)))
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


/// PromoteValuesInLoop - Try to promote memory values to scalars by sinking
/// stores out of the loop and moving loads to before the loop.  We do this by
/// looping over the stores in the loop, looking for stores to Must pointers
/// which are loop invariant.  We promote these memory locations to use allocas
/// instead.  These allocas can easily be raised to register values by the
/// PromoteMem2Reg functionality.
///
void LICM::PromoteValuesInLoop() {
  // PromotedValues - List of values that are promoted out of the loop.  Each
  // value has an alloca instruction for it, and a canonical version of the
  // pointer.
  std::vector<std::pair<AllocaInst*, Value*> > PromotedValues;
  std::map<Value*, AllocaInst*> ValueToAllocaMap; // Map of ptr to alloca

  FindPromotableValuesInLoop(PromotedValues, ValueToAllocaMap);
  if (ValueToAllocaMap.empty()) return;   // If there are values to promote.

  Changed = true;
  NumPromoted += PromotedValues.size();

  std::vector<Value*> PointerValueNumbers;

  // Emit a copy from the value into the alloca'd value in the loop preheader
  TerminatorInst *LoopPredInst = Preheader->getTerminator();
  for (unsigned i = 0, e = PromotedValues.size(); i != e; ++i) {
    Value *Ptr = PromotedValues[i].second;

    // If we are promoting a pointer value, update alias information for the
    // inserted load.
    Value *LoadValue = 0;
    if (isa<PointerType>(cast<PointerType>(Ptr->getType())->getElementType())) {
      // Locate a load or store through the pointer, and assign the same value
      // to LI as we are loading or storing.  Since we know that the value is
      // stored in this loop, this will always succeed.
      for (Value::use_iterator UI = Ptr->use_begin(), E = Ptr->use_end();
           UI != E; ++UI)
        if (LoadInst *LI = dyn_cast<LoadInst>(*UI)) {
          LoadValue = LI;
          break;
        } else if (StoreInst *SI = dyn_cast<StoreInst>(*UI)) {
          if (SI->getOperand(1) == Ptr) {
            LoadValue = SI->getOperand(0);
            break;
          }
        }
      assert(LoadValue && "No store through the pointer found!");
      PointerValueNumbers.push_back(LoadValue);  // Remember this for later.
    }

    // Load from the memory we are promoting.
    LoadInst *LI = new LoadInst(Ptr, Ptr->getName()+".promoted", LoopPredInst);

    if (LoadValue) CurAST->copyValue(LoadValue, LI);

    // Store into the temporary alloca.
    new StoreInst(LI, PromotedValues[i].first, LoopPredInst);
  }

  // Scan the basic blocks in the loop, replacing uses of our pointers with
  // uses of the allocas in question.
  //
  for (Loop::block_iterator I = CurLoop->block_begin(),
         E = CurLoop->block_end(); I != E; ++I) {
    BasicBlock *BB = *I;
    // Rewrite all loads and stores in the block of the pointer...
    for (BasicBlock::iterator II = BB->begin(), E = BB->end(); II != E; ++II) {
      if (LoadInst *L = dyn_cast<LoadInst>(II)) {
        std::map<Value*, AllocaInst*>::iterator
          I = ValueToAllocaMap.find(L->getOperand(0));
        if (I != ValueToAllocaMap.end())
          L->setOperand(0, I->second);    // Rewrite load instruction...
      } else if (StoreInst *S = dyn_cast<StoreInst>(II)) {
        std::map<Value*, AllocaInst*>::iterator
          I = ValueToAllocaMap.find(S->getOperand(1));
        if (I != ValueToAllocaMap.end())
          S->setOperand(1, I->second);    // Rewrite store instruction...
      }
    }
  }

  // Now that the body of the loop uses the allocas instead of the original
  // memory locations, insert code to copy the alloca value back into the
  // original memory location on all exits from the loop.  Note that we only
  // want to insert one copy of the code in each exit block, though the loop may
  // exit to the same block more than once.
  //
  SmallPtrSet<BasicBlock*, 16> ProcessedBlocks;

  SmallVector<BasicBlock*, 8> ExitBlocks;
  CurLoop->getExitBlocks(ExitBlocks);
  for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i) {
    if (!ProcessedBlocks.insert(ExitBlocks[i]))
      continue;
  
    // Copy all of the allocas into their memory locations.
    BasicBlock::iterator BI = ExitBlocks[i]->getFirstNonPHI();
    Instruction *InsertPos = BI;
    unsigned PVN = 0;
    for (unsigned i = 0, e = PromotedValues.size(); i != e; ++i) {
      // Load from the alloca.
      LoadInst *LI = new LoadInst(PromotedValues[i].first, "", InsertPos);

      // If this is a pointer type, update alias info appropriately.
      if (isa<PointerType>(LI->getType()))
        CurAST->copyValue(PointerValueNumbers[PVN++], LI);

      // Store into the memory we promoted.
      new StoreInst(LI, PromotedValues[i].second, InsertPos);
    }
  }

  // Now that we have done the deed, use the mem2reg functionality to promote
  // all of the new allocas we just created into real SSA registers.
  //
  std::vector<AllocaInst*> PromotedAllocas;
  PromotedAllocas.reserve(PromotedValues.size());
  for (unsigned i = 0, e = PromotedValues.size(); i != e; ++i)
    PromotedAllocas.push_back(PromotedValues[i].first);
  PromoteMemToReg(PromotedAllocas, *DT, *DF, CurAST);
}

/// FindPromotableValuesInLoop - Check the current loop for stores to definite
/// pointers, which are not loaded and stored through may aliases and are safe
/// for promotion.  If these are found, create an alloca for the value, add it 
/// to the PromotedValues list, and keep track of the mapping from value to 
/// alloca. 
void LICM::FindPromotableValuesInLoop(
                   std::vector<std::pair<AllocaInst*, Value*> > &PromotedValues,
                             std::map<Value*, AllocaInst*> &ValueToAllocaMap) {
  Instruction *FnStart = CurLoop->getHeader()->getParent()->begin()->begin();

  SmallVector<BasicBlock*, 4> ExitingBlocks;
  CurLoop->getExitingBlocks(ExitingBlocks);

  // Loop over all of the alias sets in the tracker object.
  for (AliasSetTracker::iterator I = CurAST->begin(), E = CurAST->end();
       I != E; ++I) {
    AliasSet &AS = *I;
    // We can promote this alias set if it has a store, if it is a "Must" alias
    // set, if the pointer is loop invariant, and if we are not eliminating any
    // volatile loads or stores.
    if (AS.isForwardingAliasSet() || !AS.isMod() || !AS.isMustAlias() ||
        AS.isVolatile() || !CurLoop->isLoopInvariant(AS.begin()->first))
      continue;
    
    assert(!AS.empty() &&
           "Must alias set should have at least one pointer element in it!");
    Value *V = AS.begin()->first;

    // Check that all of the pointers in the alias set have the same type.  We
    // cannot (yet) promote a memory location that is loaded and stored in
    // different sizes.
    {
      bool PointerOk = true;
      for (AliasSet::iterator I = AS.begin(), E = AS.end(); I != E; ++I)
        if (V->getType() != I->first->getType()) {
          PointerOk = false;
          break;
        }
      if (!PointerOk)
        continue;
    }

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
    bool InvalidInst = false;
    for (Value::use_iterator UI = V->use_begin(), UE = V->use_end();
         UI != UE; ++UI) {
      // Ignore instructions not in this loop.
      Instruction *Use = dyn_cast<Instruction>(*UI);
      if (!Use || !CurLoop->contains(Use->getParent()))
        continue;

      if (!isa<LoadInst>(Use) && !isa<StoreInst>(Use)) {
        InvalidInst = true;
        break;
      }
      
      if (!GuaranteedToExecute)
        GuaranteedToExecute = isSafeToExecuteUnconditionally(*Use);
    }

    // If there is an non-load/store instruction in the loop, we can't promote
    // it.  If there isn't a guaranteed-to-execute instruction, we can't
    // promote.
    if (InvalidInst || !GuaranteedToExecute)
      continue;
    
    const Type *Ty = cast<PointerType>(V->getType())->getElementType();
    AllocaInst *AI = new AllocaInst(Ty, 0, V->getName()+".tmp", FnStart);
    PromotedValues.push_back(std::make_pair(AI, V));

    // Update the AST and alias analysis.
    CurAST->copyValue(V, AI);

    for (AliasSet::iterator I = AS.begin(), E = AS.end(); I != E; ++I)
      ValueToAllocaMap.insert(std::make_pair(I->first, AI));

    DOUT << "LICM: Promoting value: " << *V << "\n";
  }
}

/// cloneBasicBlockAnalysis - Simple Analysis hook. Clone alias set info.
void LICM::cloneBasicBlockAnalysis(BasicBlock *From, BasicBlock *To, Loop *L) {
  AliasSetTracker *AST = LoopToAliasMap[L];
  if (!AST)
    return;

  AST->copyValue(From, To);
}

/// deleteAnalysisValue - Simple Analysis hook. Delete value V from alias
/// set.
void LICM::deleteAnalysisValue(Value *V, Loop *L) {
  AliasSetTracker *AST = LoopToAliasMap[L];
  if (!AST)
    return;

  AST->deleteValue(V);
}
