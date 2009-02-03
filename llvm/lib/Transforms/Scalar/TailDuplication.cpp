//===- TailDuplication.cpp - Simplify CFG through tail duplication --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass performs a limited form of tail duplication, intended to simplify
// CFGs by removing some unconditional branches.  This pass is necessary to
// straighten out loops created by the C front-end, but also is capable of
// making other code nicer.  After this pass is run, the CFG simplify pass
// should be run to clean up the mess.
//
// This pass could be enhanced in the future to use profile information to be
// more aggressive.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "tailduplicate"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Constant.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Pass.h"
#include "llvm/Type.h"
#include "llvm/Support/CFG.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <map>
using namespace llvm;

STATISTIC(NumEliminated, "Number of unconditional branches eliminated");

static cl::opt<unsigned>
TailDupThreshold("taildup-threshold",
                 cl::desc("Max block size to tail duplicate"),
                 cl::init(1), cl::Hidden);

namespace {
  class VISIBILITY_HIDDEN TailDup : public FunctionPass {
    bool runOnFunction(Function &F);
  public:
    static char ID; // Pass identification, replacement for typeid
    TailDup() : FunctionPass(&ID) {}

  private:
    inline bool shouldEliminateUnconditionalBranch(TerminatorInst *, unsigned);
    inline void eliminateUnconditionalBranch(BranchInst *BI);
    SmallPtrSet<BasicBlock*, 4> CycleDetector;
  };
}

char TailDup::ID = 0;
static RegisterPass<TailDup> X("tailduplicate", "Tail Duplication");

// Public interface to the Tail Duplication pass
FunctionPass *llvm::createTailDuplicationPass() { return new TailDup(); }

/// runOnFunction - Top level algorithm - Loop over each unconditional branch in
/// the function, eliminating it if it looks attractive enough.  CycleDetector
/// prevents infinite loops by checking that we aren't redirecting a branch to
/// a place it already pointed to earlier; see PR 2323.
bool TailDup::runOnFunction(Function &F) {
  bool Changed = false;
  CycleDetector.clear();
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ) {
    if (shouldEliminateUnconditionalBranch(I->getTerminator(),
                                           TailDupThreshold)) {
      eliminateUnconditionalBranch(cast<BranchInst>(I->getTerminator()));
      Changed = true;
    } else {
      ++I;
      CycleDetector.clear();
    }
  }
  return Changed;
}

/// shouldEliminateUnconditionalBranch - Return true if this branch looks
/// attractive to eliminate.  We eliminate the branch if the destination basic
/// block has <= 5 instructions in it, not counting PHI nodes.  In practice,
/// since one of these is a terminator instruction, this means that we will add
/// up to 4 instructions to the new block.
///
/// We don't count PHI nodes in the count since they will be removed when the
/// contents of the block are copied over.
///
bool TailDup::shouldEliminateUnconditionalBranch(TerminatorInst *TI,
                                                 unsigned Threshold) {
  BranchInst *BI = dyn_cast<BranchInst>(TI);
  if (!BI || !BI->isUnconditional()) return false;  // Not an uncond branch!

  BasicBlock *Dest = BI->getSuccessor(0);
  if (Dest == BI->getParent()) return false;        // Do not loop infinitely!

  // Do not inline a block if we will just get another branch to the same block!
  TerminatorInst *DTI = Dest->getTerminator();
  if (BranchInst *DBI = dyn_cast<BranchInst>(DTI))
    if (DBI->isUnconditional() && DBI->getSuccessor(0) == Dest)
      return false;                                 // Do not loop infinitely!

  // FIXME: DemoteRegToStack cannot yet demote invoke instructions to the stack,
  // because doing so would require breaking critical edges.  This should be
  // fixed eventually.
  if (!DTI->use_empty())
    return false;

  // Do not bother with blocks with only a single predecessor: simplify
  // CFG will fold these two blocks together!
  pred_iterator PI = pred_begin(Dest), PE = pred_end(Dest);
  ++PI;
  if (PI == PE) return false;  // Exactly one predecessor!

  BasicBlock::iterator I = Dest->getFirstNonPHI();

  for (unsigned Size = 0; I != Dest->end(); ++I) {
    if (Size == Threshold) return false;  // The block is too large.
    
    // Don't tail duplicate call instructions.  They are very large compared to
    // other instructions.
    if (isa<CallInst>(I) || isa<InvokeInst>(I)) return false;

    // Allso alloca and malloc.
    if (isa<AllocationInst>(I)) return false;

    // Some vector instructions can expand into a number of instructions.
    if (isa<ShuffleVectorInst>(I) || isa<ExtractElementInst>(I) ||
        isa<InsertElementInst>(I)) return false;
    
    // Only count instructions that are not debugger intrinsics.
    if (!isa<DbgInfoIntrinsic>(I)) ++Size;
  }

  // Do not tail duplicate a block that has thousands of successors into a block
  // with a single successor if the block has many other predecessors.  This can
  // cause an N^2 explosion in CFG edges (and PHI node entries), as seen in
  // cases that have a large number of indirect gotos.
  unsigned NumSuccs = DTI->getNumSuccessors();
  if (NumSuccs > 8) {
    unsigned TooMany = 128;
    if (NumSuccs >= TooMany) return false;
    TooMany = TooMany/NumSuccs;
    for (; PI != PE; ++PI)
      if (TooMany-- == 0) return false;
  }
  
  // If this unconditional branch is a fall-through, be careful about
  // tail duplicating it.  In particular, we don't want to taildup it if the
  // original block will still be there after taildup is completed: doing so
  // would eliminate the fall-through, requiring unconditional branches.
  Function::iterator DestI = Dest;
  if (&*--DestI == BI->getParent()) {
    // The uncond branch is a fall-through.  Tail duplication of the block is
    // will eliminate the fall-through-ness and end up cloning the terminator
    // at the end of the Dest block.  Since the original Dest block will
    // continue to exist, this means that one or the other will not be able to
    // fall through.  One typical example that this helps with is code like:
    // if (a)
    //   foo();
    // if (b)
    //   foo();
    // Cloning the 'if b' block into the end of the first foo block is messy.
    
    // The messy case is when the fall-through block falls through to other
    // blocks.  This is what we would be preventing if we cloned the block.
    DestI = Dest;
    if (++DestI != Dest->getParent()->end()) {
      BasicBlock *DestSucc = DestI;
      // If any of Dest's successors are fall-throughs, don't do this xform.
      for (succ_iterator SI = succ_begin(Dest), SE = succ_end(Dest);
           SI != SE; ++SI)
        if (*SI == DestSucc)
          return false;
    }
  }

  // Finally, check that we haven't redirected to this target block earlier;
  // there are cases where we loop forever if we don't check this (PR 2323).
  if (!CycleDetector.insert(Dest))
    return false;

  return true;
}

/// FindObviousSharedDomOf - We know there is a branch from SrcBlock to
/// DestBlock, and that SrcBlock is not the only predecessor of DstBlock.  If we
/// can find a predecessor of SrcBlock that is a dominator of both SrcBlock and
/// DstBlock, return it.
static BasicBlock *FindObviousSharedDomOf(BasicBlock *SrcBlock,
                                          BasicBlock *DstBlock) {
  // SrcBlock must have a single predecessor.
  pred_iterator PI = pred_begin(SrcBlock), PE = pred_end(SrcBlock);
  if (PI == PE || ++PI != PE) return 0;

  BasicBlock *SrcPred = *pred_begin(SrcBlock);

  // Look at the predecessors of DstBlock.  One of them will be SrcBlock.  If
  // there is only one other pred, get it, otherwise we can't handle it.
  PI = pred_begin(DstBlock); PE = pred_end(DstBlock);
  BasicBlock *DstOtherPred = 0;
  if (*PI == SrcBlock) {
    if (++PI == PE) return 0;
    DstOtherPred = *PI;
    if (++PI != PE) return 0;
  } else {
    DstOtherPred = *PI;
    if (++PI == PE || *PI != SrcBlock || ++PI != PE) return 0;
  }

  // We can handle two situations here: "if then" and "if then else" blocks.  An
  // 'if then' situation is just where DstOtherPred == SrcPred.
  if (DstOtherPred == SrcPred)
    return SrcPred;

  // Check to see if we have an "if then else" situation, which means that
  // DstOtherPred will have a single predecessor and it will be SrcPred.
  PI = pred_begin(DstOtherPred); PE = pred_end(DstOtherPred);
  if (PI != PE && *PI == SrcPred) {
    if (++PI != PE) return 0;  // Not a single pred.
    return SrcPred;  // Otherwise, it's an "if then" situation.  Return the if.
  }

  // Otherwise, this is something we can't handle.
  return 0;
}


/// eliminateUnconditionalBranch - Clone the instructions from the destination
/// block into the source block, eliminating the specified unconditional branch.
/// If the destination block defines values used by successors of the dest
/// block, we may need to insert PHI nodes.
///
void TailDup::eliminateUnconditionalBranch(BranchInst *Branch) {
  BasicBlock *SourceBlock = Branch->getParent();
  BasicBlock *DestBlock = Branch->getSuccessor(0);
  assert(SourceBlock != DestBlock && "Our predicate is broken!");

  DOUT << "TailDuplication[" << SourceBlock->getParent()->getName()
       << "]: Eliminating branch: " << *Branch;

  // See if we can avoid duplicating code by moving it up to a dominator of both
  // blocks.
  if (BasicBlock *DomBlock = FindObviousSharedDomOf(SourceBlock, DestBlock)) {
    DOUT << "Found shared dominator: " << DomBlock->getName() << "\n";

    // If there are non-phi instructions in DestBlock that have no operands
    // defined in DestBlock, and if the instruction has no side effects, we can
    // move the instruction to DomBlock instead of duplicating it.
    BasicBlock::iterator BBI = DestBlock->getFirstNonPHI();
    while (!isa<TerminatorInst>(BBI)) {
      Instruction *I = BBI++;

      bool CanHoist = !I->isTrapping() && !I->mayWriteToMemory();
      if (CanHoist) {
        for (unsigned op = 0, e = I->getNumOperands(); op != e; ++op)
          if (Instruction *OpI = dyn_cast<Instruction>(I->getOperand(op)))
            if (OpI->getParent() == DestBlock ||
                (isa<InvokeInst>(OpI) && OpI->getParent() == DomBlock)) {
              CanHoist = false;
              break;
            }
        if (CanHoist) {
          // Remove from DestBlock, move right before the term in DomBlock.
          DestBlock->getInstList().remove(I);
          DomBlock->getInstList().insert(DomBlock->getTerminator(), I);
          DOUT << "Hoisted: " << *I;
        }
      }
    }
  }

  // Tail duplication can not update SSA properties correctly if the values
  // defined in the duplicated tail are used outside of the tail itself.  For
  // this reason, we spill all values that are used outside of the tail to the
  // stack.
  for (BasicBlock::iterator I = DestBlock->begin(); I != DestBlock->end(); ++I)
    if (I->isUsedOutsideOfBlock(DestBlock)) {
      // We found a use outside of the tail.  Create a new stack slot to
      // break this inter-block usage pattern.
      DemoteRegToStack(*I);
    }

  // We are going to have to map operands from the original block B to the new
  // copy of the block B'.  If there are PHI nodes in the DestBlock, these PHI
  // nodes also define part of this mapping.  Loop over these PHI nodes, adding
  // them to our mapping.
  //
  std::map<Value*, Value*> ValueMapping;

  BasicBlock::iterator BI = DestBlock->begin();
  bool HadPHINodes = isa<PHINode>(BI);
  for (; PHINode *PN = dyn_cast<PHINode>(BI); ++BI)
    ValueMapping[PN] = PN->getIncomingValueForBlock(SourceBlock);

  // Clone the non-phi instructions of the dest block into the source block,
  // keeping track of the mapping...
  //
  for (; BI != DestBlock->end(); ++BI) {
    Instruction *New = BI->clone();
    New->setName(BI->getName());
    SourceBlock->getInstList().push_back(New);
    ValueMapping[BI] = New;
  }

  // Now that we have built the mapping information and cloned all of the
  // instructions (giving us a new terminator, among other things), walk the new
  // instructions, rewriting references of old instructions to use new
  // instructions.
  //
  BI = Branch; ++BI;  // Get an iterator to the first new instruction
  for (; BI != SourceBlock->end(); ++BI)
    for (unsigned i = 0, e = BI->getNumOperands(); i != e; ++i)
      if (Value *Remapped = ValueMapping[BI->getOperand(i)])
        BI->setOperand(i, Remapped);

  // Next we check to see if any of the successors of DestBlock had PHI nodes.
  // If so, we need to add entries to the PHI nodes for SourceBlock now.
  for (succ_iterator SI = succ_begin(DestBlock), SE = succ_end(DestBlock);
       SI != SE; ++SI) {
    BasicBlock *Succ = *SI;
    for (BasicBlock::iterator PNI = Succ->begin(); isa<PHINode>(PNI); ++PNI) {
      PHINode *PN = cast<PHINode>(PNI);
      // Ok, we have a PHI node.  Figure out what the incoming value was for the
      // DestBlock.
      Value *IV = PN->getIncomingValueForBlock(DestBlock);

      // Remap the value if necessary...
      if (Value *MappedIV = ValueMapping[IV])
        IV = MappedIV;
      PN->addIncoming(IV, SourceBlock);
    }
  }

  // Next, remove the old branch instruction, and any PHI node entries that we
  // had.
  BI = Branch; ++BI;  // Get an iterator to the first new instruction
  DestBlock->removePredecessor(SourceBlock); // Remove entries in PHI nodes...
  SourceBlock->getInstList().erase(Branch);  // Destroy the uncond branch...

  // Final step: now that we have finished everything up, walk the cloned
  // instructions one last time, constant propagating and DCE'ing them, because
  // they may not be needed anymore.
  //
  if (HadPHINodes) {
    while (BI != SourceBlock->end()) {
      Instruction *Inst = BI++;
      if (isInstructionTriviallyDead(Inst))
        Inst->eraseFromParent();
      else if (Constant *C = ConstantFoldInstruction(Inst)) {
        Inst->replaceAllUsesWith(C);
        Inst->eraseFromParent();
      }
    }
  }

  ++NumEliminated;  // We just killed a branch!
}
