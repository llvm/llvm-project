//===-- BasicBlock.cpp - Implement BasicBlock related methods -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the BasicBlock class for the IR library.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/BasicBlock.h"
#include "SymbolTableListTraitsImpl.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"
#include <algorithm>

using namespace llvm;

ValueSymbolTable *BasicBlock::getValueSymbolTable() {
  if (Function *F = getParent())
    return F->getValueSymbolTable();
  return nullptr;
}

LLVMContext &BasicBlock::getContext() const {
  return getType()->getContext();
}

template <> void llvm::invalidateParentIListOrdering(BasicBlock *BB) {
  BB->invalidateOrders();
}

// Explicit instantiation of SymbolTableListTraits since some of the methods
// are not in the public header file...
template class llvm::SymbolTableListTraits<Instruction>;

BasicBlock::BasicBlock(LLVMContext &C, const Twine &Name, Function *NewParent,
                       BasicBlock *InsertBefore)
  : Value(Type::getLabelTy(C), Value::BasicBlockVal), Parent(nullptr) {

  if (NewParent)
    insertInto(NewParent, InsertBefore);
  else
    assert(!InsertBefore &&
           "Cannot insert block before another block with no function!");

  setName(Name);
}

void BasicBlock::insertInto(Function *NewParent, BasicBlock *InsertBefore) {
  assert(NewParent && "Expected a parent");
  assert(!Parent && "Already has a parent");

  if (InsertBefore)
    NewParent->getBasicBlockList().insert(InsertBefore->getIterator(), this);
  else
    NewParent->getBasicBlockList().push_back(this);
}

BasicBlock::~BasicBlock() {
  validateInstrOrdering();

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
      BlockAddress *BA = cast<BlockAddress>(user_back());
      BA->replaceAllUsesWith(ConstantExpr::getIntToPtr(Replacement,
                                                       BA->getType()));
      BA->destroyConstant();
    }
  }

  assert(getParent() == nullptr && "BasicBlock still linked into the program!");
  dropAllReferences();
  InstList.clear();
}

void BasicBlock::setParent(Function *parent) {
  // Set Parent=parent, updating instruction symtab entries as appropriate.
  InstList.setSymTabObject(&Parent, parent);
}

iterator_range<filter_iterator<BasicBlock::const_iterator,
                               std::function<bool(const Instruction &)>>>
BasicBlock::instructionsWithoutDebug() const {
  std::function<bool(const Instruction &)> Fn = [](const Instruction &I) {
    return !isa<DbgInfoIntrinsic>(I);
  };
  return make_filter_range(*this, Fn);
}

iterator_range<filter_iterator<BasicBlock::iterator,
                               std::function<bool(Instruction &)>>>
BasicBlock::instructionsWithoutDebug() {
  std::function<bool(Instruction &)> Fn = [](Instruction &I) {
    return !isa<DbgInfoIntrinsic>(I);
  };
  return make_filter_range(*this, Fn);
}

filter_iterator<BasicBlock::const_iterator,
                std::function<bool(const Instruction &)>>::difference_type
BasicBlock::sizeWithoutDebug() const {
  return std::distance(instructionsWithoutDebug().begin(),
                       instructionsWithoutDebug().end());
}

void BasicBlock::removeFromParent() {
  getParent()->getBasicBlockList().remove(getIterator());
}

iplist<BasicBlock>::iterator BasicBlock::eraseFromParent() {
  return getParent()->getBasicBlockList().erase(getIterator());
}

/// Unlink this basic block from its current function and
/// insert it into the function that MovePos lives in, right before MovePos.
void BasicBlock::moveBefore(BasicBlock *MovePos) {
  MovePos->getParent()->getBasicBlockList().splice(
      MovePos->getIterator(), getParent()->getBasicBlockList(), getIterator());
}

/// Unlink this basic block from its current function and
/// insert it into the function that MovePos lives in, right after MovePos.
void BasicBlock::moveAfter(BasicBlock *MovePos) {
  MovePos->getParent()->getBasicBlockList().splice(
      ++MovePos->getIterator(), getParent()->getBasicBlockList(),
      getIterator());
}

const Module *BasicBlock::getModule() const {
  return getParent()->getParent();
}

const Instruction *BasicBlock::getTerminator() const {
  if (InstList.empty() || !InstList.back().isTerminator())
    return nullptr;
  return &InstList.back();
}

const CallInst *BasicBlock::getTerminatingMustTailCall() const {
  if (InstList.empty())
    return nullptr;
  const ReturnInst *RI = dyn_cast<ReturnInst>(&InstList.back());
  if (!RI || RI == &InstList.front())
    return nullptr;

  const Instruction *Prev = RI->getPrevNode();
  if (!Prev)
    return nullptr;

  if (Value *RV = RI->getReturnValue()) {
    if (RV != Prev)
      return nullptr;

    // Look through the optional bitcast.
    if (auto *BI = dyn_cast<BitCastInst>(Prev)) {
      RV = BI->getOperand(0);
      Prev = BI->getPrevNode();
      if (!Prev || RV != Prev)
        return nullptr;
    }
  }

  if (auto *CI = dyn_cast<CallInst>(Prev)) {
    if (CI->isMustTailCall())
      return CI;
  }
  return nullptr;
}

const CallInst *BasicBlock::getTerminatingDeoptimizeCall() const {
  if (InstList.empty())
    return nullptr;
  auto *RI = dyn_cast<ReturnInst>(&InstList.back());
  if (!RI || RI == &InstList.front())
    return nullptr;

  if (auto *CI = dyn_cast_or_null<CallInst>(RI->getPrevNode()))
    if (Function *F = CI->getCalledFunction())
      if (F->getIntrinsicID() == Intrinsic::experimental_deoptimize)
        return CI;

  return nullptr;
}

const CallInst *BasicBlock::getPostdominatingDeoptimizeCall() const {
  const BasicBlock* BB = this;
  SmallPtrSet<const BasicBlock *, 8> Visited;
  Visited.insert(BB);
  while (auto *Succ = BB->getUniqueSuccessor()) {
    if (!Visited.insert(Succ).second)
      return nullptr;
    BB = Succ;
  }
  return BB->getTerminatingDeoptimizeCall();
}

const Instruction* BasicBlock::getFirstNonPHI() const {
  for (const Instruction &I : *this)
    if (!isa<PHINode>(I))
      return &I;
  return nullptr;
}

const Instruction* BasicBlock::getFirstNonPHIOrDbg() const {
  for (const Instruction &I : *this)
    if (!isa<PHINode>(I) && !isa<DbgInfoIntrinsic>(I))
      return &I;
  return nullptr;
}

const Instruction* BasicBlock::getFirstNonPHIOrDbgOrLifetime() const {
  for (const Instruction &I : *this) {
    if (isa<PHINode>(I) || isa<DbgInfoIntrinsic>(I))
      continue;

    if (I.isLifetimeStartOrEnd())
      continue;

    return &I;
  }
  return nullptr;
}

BasicBlock::const_iterator BasicBlock::getFirstInsertionPt() const {
  const Instruction *FirstNonPHI = getFirstNonPHI();
  if (!FirstNonPHI)
    return end();

  const_iterator InsertPt = FirstNonPHI->getIterator();
  if (InsertPt->isEHPad()) ++InsertPt;
  return InsertPt;
}

void BasicBlock::dropAllReferences() {
  for (Instruction &I : *this)
    I.dropAllReferences();
}

/// If this basic block has a single predecessor block,
/// return the block, otherwise return a null pointer.
const BasicBlock *BasicBlock::getSinglePredecessor() const {
  const_pred_iterator PI = pred_begin(this), E = pred_end(this);
  if (PI == E) return nullptr;         // No preds.
  const BasicBlock *ThePred = *PI;
  ++PI;
  return (PI == E) ? ThePred : nullptr /*multiple preds*/;
}

/// If this basic block has a unique predecessor block,
/// return the block, otherwise return a null pointer.
/// Note that unique predecessor doesn't mean single edge, there can be
/// multiple edges from the unique predecessor to this block (for example
/// a switch statement with multiple cases having the same destination).
const BasicBlock *BasicBlock::getUniquePredecessor() const {
  const_pred_iterator PI = pred_begin(this), E = pred_end(this);
  if (PI == E) return nullptr; // No preds.
  const BasicBlock *PredBB = *PI;
  ++PI;
  for (;PI != E; ++PI) {
    if (*PI != PredBB)
      return nullptr;
    // The same predecessor appears multiple times in the predecessor list.
    // This is OK.
  }
  return PredBB;
}

bool BasicBlock::hasNPredecessors(unsigned N) const {
  return hasNItems(pred_begin(this), pred_end(this), N);
}

bool BasicBlock::hasNPredecessorsOrMore(unsigned N) const {
  return hasNItemsOrMore(pred_begin(this), pred_end(this), N);
}

const BasicBlock *BasicBlock::getSingleSuccessor() const {
  const_succ_iterator SI = succ_begin(this), E = succ_end(this);
  if (SI == E) return nullptr; // no successors
  const BasicBlock *TheSucc = *SI;
  ++SI;
  return (SI == E) ? TheSucc : nullptr /* multiple successors */;
}

const BasicBlock *BasicBlock::getUniqueSuccessor() const {
  const_succ_iterator SI = succ_begin(this), E = succ_end(this);
  if (SI == E) return nullptr; // No successors
  const BasicBlock *SuccBB = *SI;
  ++SI;
  for (;SI != E; ++SI) {
    if (*SI != SuccBB)
      return nullptr;
    // The same successor appears multiple times in the successor list.
    // This is OK.
  }
  return SuccBB;
}

iterator_range<BasicBlock::phi_iterator> BasicBlock::phis() {
  PHINode *P = empty() ? nullptr : dyn_cast<PHINode>(&*begin());
  return make_range<phi_iterator>(P, nullptr);
}

/// Update PHI nodes in this BasicBlock before removal of predecessor \p Pred.
/// Note that this function does not actually remove the predecessor.
///
/// If \p KeepOneInputPHIs is true then don't remove PHIs that are left with
/// zero or one incoming values, and don't simplify PHIs with all incoming
/// values the same.
///
/// If \p MaybeDeadInstrs is not nullptr then whenever we drop a reference to
/// an instruction, append it to the vector. The caller should check whether
/// these instructions are now trivially dead, and if so delete them.
void BasicBlock::removePredecessor(
    BasicBlock *Pred, bool KeepOneInputPHIs,
    SmallVectorImpl<WeakTrackingVH> *MaybeDeadInstrs) {
  // Use hasNUsesOrMore to bound the cost of this assertion for complex CFGs.
  assert((hasNUsesOrMore(16) ||
          find(pred_begin(this), pred_end(this), Pred) != pred_end(this)) &&
         "Pred is not a predecessor!");

  // Return early if there are no PHI nodes to update.
  if (!isa<PHINode>(begin()))
    return;
  unsigned NumPreds = cast<PHINode>(front()).getNumIncomingValues();

  // Update all PHI nodes.
  for (iterator II = begin(); isa<PHINode>(II);) {
    PHINode *PN = cast<PHINode>(II++);
    Value *V = PN->removeIncomingValue(Pred, !KeepOneInputPHIs);
    if (MaybeDeadInstrs) {
      if (auto *I = dyn_cast<Instruction>(V))
        MaybeDeadInstrs->push_back(I);
    }
    if (!KeepOneInputPHIs) {
      // If we have a single predecessor, removeIncomingValue erased the PHI
      // node itself.
      if (NumPreds > 1) {
        if (Value *PNV = PN->hasConstantValue()) {
          // Replace the PHI node with its constant value.
          PN->replaceAllUsesWith(PNV);
          PN->eraseFromParent();
        }
      }
    }
  }
}

bool BasicBlock::canSplitPredecessors() const {
  const Instruction *FirstNonPHI = getFirstNonPHI();
  if (isa<LandingPadInst>(FirstNonPHI))
    return true;
  // This is perhaps a little conservative because constructs like
  // CleanupBlockInst are pretty easy to split.  However, SplitBlockPredecessors
  // cannot handle such things just yet.
  if (FirstNonPHI->isEHPad())
    return false;
  return true;
}

bool BasicBlock::isLegalToHoistInto() const {
  auto *Term = getTerminator();
  // No terminator means the block is under construction.
  if (!Term)
    return true;

  // If the block has no successors, there can be no instructions to hoist.
  assert(Term->getNumSuccessors() > 0);

  // Instructions should not be hoisted across exception handling boundaries.
  return !Term->isExceptionalTerminator();
}

/// This splits a basic block into two at the specified
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

  BasicBlock *New = BasicBlock::Create(getContext(), BBName, getParent(),
                                       this->getNextNode());

  // Save DebugLoc of split point before invalidating iterator.
  DebugLoc Loc = I->getDebugLoc();
  // Move all of the specified instructions from the original basic block into
  // the new basic block.
  New->getInstList().splice(New->end(), this->getInstList(), I, end());

  // Add a branch instruction to the newly formed basic block.
  BranchInst *BI = BranchInst::Create(New, this);
  BI->setDebugLoc(Loc);

  // Now we must loop through all of the successors of the New block (which
  // _were_ the successors of the 'this' block), and update any PHI nodes in
  // successors.  If there were PHI nodes in the successors, then they need to
  // know that incoming branches will be from New, not from Old (this).
  //
  New->replaceSuccessorsPhiUsesWith(this, New);
  return New;
}

void BasicBlock::replacePhiUsesWith(BasicBlock *Old, BasicBlock *New) {
  // N.B. This might not be a complete BasicBlock, so don't assume
  // that it ends with a non-phi instruction.
  for (iterator II = begin(), IE = end(); II != IE; ++II) {
    PHINode *PN = dyn_cast<PHINode>(II);
    if (!PN)
      break;
    PN->replaceIncomingBlockWith(Old, New);
  }
}

void BasicBlock::replaceSuccessorsPhiUsesWith(BasicBlock *Old,
                                              BasicBlock *New) {
  Instruction *TI = getTerminator();
  if (!TI)
    // Cope with being called on a BasicBlock that doesn't have a terminator
    // yet. Clang's CodeGenFunction::EmitReturnBlock() likes to do this.
    return;
  llvm::for_each(successors(TI), [Old, New](BasicBlock *Succ) {
    Succ->replacePhiUsesWith(Old, New);
  });
}

void BasicBlock::replaceSuccessorsPhiUsesWith(BasicBlock *New) {
  this->replaceSuccessorsPhiUsesWith(this, New);
}

/// Return true if this basic block is a landing pad. I.e., it's
/// the destination of the 'unwind' edge of an invoke instruction.
bool BasicBlock::isLandingPad() const {
  return isa<LandingPadInst>(getFirstNonPHI());
}

/// Return the landingpad instruction associated with the landing pad.
const LandingPadInst *BasicBlock::getLandingPadInst() const {
  return dyn_cast<LandingPadInst>(getFirstNonPHI());
}

Optional<uint64_t> BasicBlock::getIrrLoopHeaderWeight() const {
  const Instruction *TI = getTerminator();
  if (MDNode *MDIrrLoopHeader =
      TI->getMetadata(LLVMContext::MD_irr_loop)) {
    MDString *MDName = cast<MDString>(MDIrrLoopHeader->getOperand(0));
    if (MDName->getString().equals("loop_header_weight")) {
      auto *CI = mdconst::extract<ConstantInt>(MDIrrLoopHeader->getOperand(1));
      return Optional<uint64_t>(CI->getValue().getZExtValue());
    }
  }
  return Optional<uint64_t>();
}

BasicBlock::iterator llvm::skipDebugIntrinsics(BasicBlock::iterator It) {
  while (isa<DbgInfoIntrinsic>(It))
    ++It;
  return It;
}

void BasicBlock::renumberInstructions() {
  unsigned Order = 0;
  for (Instruction &I : *this)
    I.Order = Order++;

  // Set the bit to indicate that the instruction order valid and cached.
  BasicBlockBits Bits = getBasicBlockBits();
  Bits.InstrOrderValid = true;
  setBasicBlockBits(Bits);
}

#ifndef NDEBUG
/// In asserts builds, this checks the numbering. In non-asserts builds, it
/// is defined as a no-op inline function in BasicBlock.h.
void BasicBlock::validateInstrOrdering() const {
  if (!isInstrOrderValid())
    return;
  const Instruction *Prev = nullptr;
  for (const Instruction &I : *this) {
    assert((!Prev || Prev->comesBefore(&I)) &&
           "cached instruction ordering is incorrect");
    Prev = &I;
  }
}
#endif
