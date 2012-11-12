//===-- BasicBlock.cpp - Implement BasicBlock related methods -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the BasicBlock class for the VMCore library.
//
//===----------------------------------------------------------------------===//

#include "llvm/BasicBlock.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/LLVMContext.h"
#include "llvm/Type.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/LeakDetector.h"
#include "SymbolTableListTraitsImpl.h"
#include <algorithm>
using namespace llvm;

ValueSymbolTable *BasicBlock::getValueSymbolTable() {
  if (Function *F = getParent())
    return &F->getValueSymbolTable();
  return 0;
}

LLVMContext &BasicBlock::getContext() const {
  return getType()->getContext();
}

// Explicit instantiation of SymbolTableListTraits since some of the methods
// are not in the public header file...
template class llvm::SymbolTableListTraits<Instruction, BasicBlock>;


BasicBlock::BasicBlock(LLVMContext &C, const Twine &Name, Function *NewParent,
                       BasicBlock *InsertBefore)
  : Value(Type::getLabelTy(C), Value::BasicBlockVal), Parent(0) {

  // Make sure that we get added to a function
  LeakDetector::addGarbageObject(this);

  if (InsertBefore) {
    assert(NewParent &&
           "Cannot insert block before another block with no function!");
    NewParent->getBasicBlockList().insert(InsertBefore, this);
  } else if (NewParent) {
    NewParent->getBasicBlockList().push_back(this);
  }

  setName(Name);
}


BasicBlock::~BasicBlock() {
  // If the address of the block is taken and it is being deleted (e.g. because
  // it is dead), this means that there is either a dangling constant expr
  // hanging off the block, or an undefined use of the block (source code
  // expecting the address of a label to keep the block alive even though there
  // is no indirect branch).  Handle these cases by zapping the BlockAddress
  // nodes.  There are no other possible uses at this point.
  if (hasAddressTaken()) {
    assert(!use_empty() && "There should be at least one blockaddress!");
    Constant *Replacement =
      ConstantInt::get(llvm::Type::getInt32Ty(getContext()), 1);
    while (!use_empty()) {
      BlockAddress *BA = cast<BlockAddress>(use_back());
      BA->replaceAllUsesWith(ConstantExpr::getIntToPtr(Replacement,
                                                       BA->getType()));
      BA->destroyConstant();
    }
  }

  assert(getParent() == 0 && "BasicBlock still linked into the program!");
  dropAllReferences();
  InstList.clear();
}

void BasicBlock::setParent(Function *parent) {
  if (getParent())
    LeakDetector::addGarbageObject(this);

  // Set Parent=parent, updating instruction symtab entries as appropriate.
  InstList.setSymTabObject(&Parent, parent);

  if (getParent())
    LeakDetector::removeGarbageObject(this);
}

void BasicBlock::removeFromParent() {
  getParent()->getBasicBlockList().remove(this);
}

void BasicBlock::eraseFromParent() {
  getParent()->getBasicBlockList().erase(this);
}

/// moveBefore - Unlink this basic block from its current function and
/// insert it into the function that MovePos lives in, right before MovePos.
void BasicBlock::moveBefore(BasicBlock *MovePos) {
  MovePos->getParent()->getBasicBlockList().splice(MovePos,
                       getParent()->getBasicBlockList(), this);
}

/// moveAfter - Unlink this basic block from its current function and
/// insert it into the function that MovePos lives in, right after MovePos.
void BasicBlock::moveAfter(BasicBlock *MovePos) {
  Function::iterator I = MovePos;
  MovePos->getParent()->getBasicBlockList().splice(++I,
                                       getParent()->getBasicBlockList(), this);
}


TerminatorInst *BasicBlock::getTerminator() {
  if (InstList.empty()) return 0;
  return dyn_cast<TerminatorInst>(&InstList.back());
}

const TerminatorInst *BasicBlock::getTerminator() const {
  if (InstList.empty()) return 0;
  return dyn_cast<TerminatorInst>(&InstList.back());
}

Instruction* BasicBlock::getFirstNonPHI() {
  BasicBlock::iterator i = begin();
  // All valid basic blocks should have a terminator,
  // which is not a PHINode. If we have an invalid basic
  // block we'll get an assertion failure when dereferencing
  // a past-the-end iterator.
  while (isa<PHINode>(i)) ++i;
  return &*i;
}

Instruction* BasicBlock::getFirstNonPHIOrDbg() {
  BasicBlock::iterator i = begin();
  // All valid basic blocks should have a terminator,
  // which is not a PHINode. If we have an invalid basic
  // block we'll get an assertion failure when dereferencing
  // a past-the-end iterator.
  while (isa<PHINode>(i) || isa<DbgInfoIntrinsic>(i)) ++i;
  return &*i;
}

Instruction* BasicBlock::getFirstNonPHIOrDbgOrLifetime() {
  // All valid basic blocks should have a terminator,
  // which is not a PHINode. If we have an invalid basic
  // block we'll get an assertion failure when dereferencing
  // a past-the-end iterator.
  BasicBlock::iterator i = begin();
  for (;; ++i) {
    if (isa<PHINode>(i) || isa<DbgInfoIntrinsic>(i))
      continue;

    const IntrinsicInst *II = dyn_cast<IntrinsicInst>(i);
    if (!II)
      break;
    if (II->getIntrinsicID() != Intrinsic::lifetime_start &&
        II->getIntrinsicID() != Intrinsic::lifetime_end)
      break;
  }
  return &*i;
}

BasicBlock::iterator BasicBlock::getFirstInsertionPt() {
  iterator InsertPt = getFirstNonPHI();
  if (isa<LandingPadInst>(InsertPt)) ++InsertPt;
  return InsertPt;
}

void BasicBlock::dropAllReferences() {
  for(iterator I = begin(), E = end(); I != E; ++I)
    I->dropAllReferences();
}

/// getSinglePredecessor - If this basic block has a single predecessor block,
/// return the block, otherwise return a null pointer.
BasicBlock *BasicBlock::getSinglePredecessor() {
  pred_iterator PI = pred_begin(this), E = pred_end(this);
  if (PI == E) return 0;         // No preds.
  BasicBlock *ThePred = *PI;
  ++PI;
  return (PI == E) ? ThePred : 0 /*multiple preds*/;
}

/// getUniquePredecessor - If this basic block has a unique predecessor block,
/// return the block, otherwise return a null pointer.
/// Note that unique predecessor doesn't mean single edge, there can be
/// multiple edges from the unique predecessor to this block (for example
/// a switch statement with multiple cases having the same destination).
BasicBlock *BasicBlock::getUniquePredecessor() {
  pred_iterator PI = pred_begin(this), E = pred_end(this);
  if (PI == E) return 0; // No preds.
  BasicBlock *PredBB = *PI;
  ++PI;
  for (;PI != E; ++PI) {
    if (*PI != PredBB)
      return 0;
    // The same predecessor appears multiple times in the predecessor list.
    // This is OK.
  }
  return PredBB;
}

/// removePredecessor - This method is used to notify a BasicBlock that the
/// specified Predecessor of the block is no longer able to reach it.  This is
/// actually not used to update the Predecessor list, but is actually used to
/// update the PHI nodes that reside in the block.  Note that this should be
/// called while the predecessor still refers to this block.
///
void BasicBlock::removePredecessor(BasicBlock *Pred,
                                   bool DontDeleteUselessPHIs) {
  assert((hasNUsesOrMore(16)||// Reduce cost of this assertion for complex CFGs.
          find(pred_begin(this), pred_end(this), Pred) != pred_end(this)) &&
         "removePredecessor: BB is not a predecessor!");

  if (InstList.empty()) return;
  PHINode *APN = dyn_cast<PHINode>(&front());
  if (!APN) return;   // Quick exit.

  // If there are exactly two predecessors, then we want to nuke the PHI nodes
  // altogether.  However, we cannot do this, if this in this case:
  //
  //  Loop:
  //    %x = phi [X, Loop]
  //    %x2 = add %x, 1         ;; This would become %x2 = add %x2, 1
  //    br Loop                 ;; %x2 does not dominate all uses
  //
  // This is because the PHI node input is actually taken from the predecessor
  // basic block.  The only case this can happen is with a self loop, so we
  // check for this case explicitly now.
  //
  unsigned max_idx = APN->getNumIncomingValues();
  assert(max_idx != 0 && "PHI Node in block with 0 predecessors!?!?!");
  if (max_idx == 2) {
    BasicBlock *Other = APN->getIncomingBlock(APN->getIncomingBlock(0) == Pred);

    // Disable PHI elimination!
    if (this == Other) max_idx = 3;
  }

  // <= Two predecessors BEFORE I remove one?
  if (max_idx <= 2 && !DontDeleteUselessPHIs) {
    // Yup, loop through and nuke the PHI nodes
    while (PHINode *PN = dyn_cast<PHINode>(&front())) {
      // Remove the predecessor first.
      PN->removeIncomingValue(Pred, !DontDeleteUselessPHIs);

      // If the PHI _HAD_ two uses, replace PHI node with its now *single* value
      if (max_idx == 2) {
        if (PN->getIncomingValue(0) != PN)
          PN->replaceAllUsesWith(PN->getIncomingValue(0));
        else
          // We are left with an infinite loop with no entries: kill the PHI.
          PN->replaceAllUsesWith(UndefValue::get(PN->getType()));
        getInstList().pop_front();    // Remove the PHI node
      }

      // If the PHI node already only had one entry, it got deleted by
      // removeIncomingValue.
    }
  } else {
    // Okay, now we know that we need to remove predecessor #pred_idx from all
    // PHI nodes.  Iterate over each PHI node fixing them up
    PHINode *PN;
    for (iterator II = begin(); (PN = dyn_cast<PHINode>(II)); ) {
      ++II;
      PN->removeIncomingValue(Pred, false);
      // If all incoming values to the Phi are the same, we can replace the Phi
      // with that value.
      Value* PNV = 0;
      if (!DontDeleteUselessPHIs && (PNV = PN->hasConstantValue()))
        if (PNV != PN) {
          PN->replaceAllUsesWith(PNV);
          PN->eraseFromParent();
        }
    }
  }
}


/// splitBasicBlock - This splits a basic block into two at the specified
/// instruction.  Note that all instructions BEFORE the specified iterator stay
/// as part of the original basic block, an unconditional branch is added to
/// the new BB, and the rest of the instructions in the BB are moved to the new
/// BB, including the old terminator.  This invalidates the iterator.
///
/// Note that this only works on well formed basic blocks (must have a
/// terminator), and 'I' must not be the end of instruction list (which would
/// cause a degenerate basic block to be formed, having a terminator inside of
/// the basic block).
///
BasicBlock *BasicBlock::splitBasicBlock(iterator I, const Twine &BBName) {
  assert(getTerminator() && "Can't use splitBasicBlock on degenerate BB!");
  assert(I != InstList.end() &&
         "Trying to get me to create degenerate basic block!");

  BasicBlock *InsertBefore = llvm::next(Function::iterator(this))
                               .getNodePtrUnchecked();
  BasicBlock *New = BasicBlock::Create(getContext(), BBName,
                                       getParent(), InsertBefore);

  // Move all of the specified instructions from the original basic block into
  // the new basic block.
  New->getInstList().splice(New->end(), this->getInstList(), I, end());

  // Add a branch instruction to the newly formed basic block.
  BranchInst::Create(New, this);

  // Now we must loop through all of the successors of the New block (which
  // _were_ the successors of the 'this' block), and update any PHI nodes in
  // successors.  If there were PHI nodes in the successors, then they need to
  // know that incoming branches will be from New, not from Old.
  //
  for (succ_iterator I = succ_begin(New), E = succ_end(New); I != E; ++I) {
    // Loop over any phi nodes in the basic block, updating the BB field of
    // incoming values...
    BasicBlock *Successor = *I;
    PHINode *PN;
    for (BasicBlock::iterator II = Successor->begin();
         (PN = dyn_cast<PHINode>(II)); ++II) {
      int IDX = PN->getBasicBlockIndex(this);
      while (IDX != -1) {
        PN->setIncomingBlock((unsigned)IDX, New);
        IDX = PN->getBasicBlockIndex(this);
      }
    }
  }
  return New;
}

void BasicBlock::replaceSuccessorsPhiUsesWith(BasicBlock *New) {
  TerminatorInst *TI = getTerminator();
  if (!TI)
    // Cope with being called on a BasicBlock that doesn't have a terminator
    // yet. Clang's CodeGenFunction::EmitReturnBlock() likes to do this.
    return;
  for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i) {
    BasicBlock *Succ = TI->getSuccessor(i);
    // N.B. Succ might not be a complete BasicBlock, so don't assume
    // that it ends with a non-phi instruction.
    for (iterator II = Succ->begin(), IE = Succ->end(); II != IE; ++II) {
      PHINode *PN = dyn_cast<PHINode>(II);
      if (!PN)
        break;
      int i;
      while ((i = PN->getBasicBlockIndex(this)) >= 0)
        PN->setIncomingBlock(i, New);
    }
  }
}

/// isLandingPad - Return true if this basic block is a landing pad. I.e., it's
/// the destination of the 'unwind' edge of an invoke instruction.
bool BasicBlock::isLandingPad() const {
  return isa<LandingPadInst>(getFirstNonPHI());
}

/// getLandingPadInst() - Return the landingpad instruction associated with
/// the landing pad.
LandingPadInst *BasicBlock::getLandingPadInst() {
  return dyn_cast<LandingPadInst>(getFirstNonPHI());
}
const LandingPadInst *BasicBlock::getLandingPadInst() const {
  return dyn_cast<LandingPadInst>(getFirstNonPHI());
}
