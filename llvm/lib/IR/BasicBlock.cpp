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
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugProgramInstruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"

#include "LLVMContextImpl.h"

using namespace llvm;

#define DEBUG_TYPE "ir"
STATISTIC(NumInstrRenumberings, "Number of renumberings across all blocks");

cl::opt<bool>
    UseNewDbgInfoFormat("experimental-debuginfo-iterators",
                        cl::desc("Enable communicating debuginfo positions "
                                 "through iterators, eliminating intrinsics"),
                        cl::init(true));

DPMarker *BasicBlock::createMarker(Instruction *I) {
  assert(IsNewDbgInfoFormat &&
         "Tried to create a marker in a non new debug-info block!");
  if (I->DbgMarker)
    return I->DbgMarker;
  DPMarker *Marker = new DPMarker();
  Marker->MarkedInstr = I;
  I->DbgMarker = Marker;
  return Marker;
}

DPMarker *BasicBlock::createMarker(InstListType::iterator It) {
  assert(IsNewDbgInfoFormat &&
         "Tried to create a marker in a non new debug-info block!");
  if (It != end())
    return createMarker(&*It);
  DPMarker *DPM = getTrailingDPValues();
  if (DPM)
    return DPM;
  DPM = new DPMarker();
  setTrailingDPValues(DPM);
  return DPM;
}

void BasicBlock::convertToNewDbgValues() {
  IsNewDbgInfoFormat = true;

  // Iterate over all instructions in the instruction list, collecting dbg.value
  // instructions and converting them to DPValues. Once we find a "real"
  // instruction, attach all those DPValues to a DPMarker in that instruction.
  SmallVector<DbgRecord *, 4> DPVals;
  for (Instruction &I : make_early_inc_range(InstList)) {
    assert(!I.DbgMarker && "DbgMarker already set on old-format instrs?");
    if (DbgVariableIntrinsic *DVI = dyn_cast<DbgVariableIntrinsic>(&I)) {
      // Convert this dbg.value to a DPValue.
      DPValue *Value = new DPValue(DVI);
      DPVals.push_back(Value);
      DVI->eraseFromParent();
      continue;
    }

    if (DbgLabelInst *DLI = dyn_cast<DbgLabelInst>(&I)) {
      DPVals.push_back(new DPLabel(DLI->getLabel(), DLI->getDebugLoc()));
      DLI->eraseFromParent();
      continue;
    }

    if (DPVals.empty())
      continue;

    // Create a marker to store DPValues in.
    createMarker(&I);
    DPMarker *Marker = I.DbgMarker;

    for (DbgRecord *DPV : DPVals)
      Marker->insertDPValue(DPV, false);

    DPVals.clear();
  }
}

void BasicBlock::convertFromNewDbgValues() {
  invalidateOrders();
  IsNewDbgInfoFormat = false;

  // Iterate over the block, finding instructions annotated with DPMarkers.
  // Convert any attached DPValues to dbg.values and insert ahead of the
  // instruction.
  for (auto &Inst : *this) {
    if (!Inst.DbgMarker)
      continue;

    DPMarker &Marker = *Inst.DbgMarker;
    for (DbgRecord &DR : Marker.getDbgValueRange())
      InstList.insert(Inst.getIterator(),
                      DR.createDebugIntrinsic(getModule(), nullptr));

    Marker.eraseFromParent();
  }

  // Assume no trailing DPValues: we could technically create them at the end
  // of the block, after a terminator, but this would be non-cannonical and
  // indicates that something else is broken somewhere.
  assert(!getTrailingDPValues());
}

#ifndef NDEBUG
void BasicBlock::dumpDbgValues() const {
  for (auto &Inst : *this) {
    if (!Inst.DbgMarker)
      continue;

    dbgs() << "@ " << Inst.DbgMarker << " ";
    Inst.DbgMarker->dump();
  };
}
#endif

void BasicBlock::setIsNewDbgInfoFormat(bool NewFlag) {
  if (NewFlag && !IsNewDbgInfoFormat)
    convertToNewDbgValues();
  else if (!NewFlag && IsNewDbgInfoFormat)
    convertFromNewDbgValues();
}

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
template class llvm::SymbolTableListTraits<Instruction,
                                           ilist_iterator_bits<true>>;

BasicBlock::BasicBlock(LLVMContext &C, const Twine &Name, Function *NewParent,
                       BasicBlock *InsertBefore)
    : Value(Type::getLabelTy(C), Value::BasicBlockVal),
      IsNewDbgInfoFormat(false), Parent(nullptr) {

  if (NewParent)
    insertInto(NewParent, InsertBefore);
  else
    assert(!InsertBefore &&
           "Cannot insert block before another block with no function!");

  setName(Name);
  if (NewParent)
    setIsNewDbgInfoFormat(NewParent->IsNewDbgInfoFormat);
}

void BasicBlock::insertInto(Function *NewParent, BasicBlock *InsertBefore) {
  assert(NewParent && "Expected a parent");
  assert(!Parent && "Already has a parent");

  if (InsertBefore)
    NewParent->insert(InsertBefore->getIterator(), this);
  else
    NewParent->insert(NewParent->end(), this);

  setIsNewDbgInfoFormat(NewParent->IsNewDbgInfoFormat);
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
  for (auto &Inst : *this) {
    if (!Inst.DbgMarker)
      continue;
    Inst.DbgMarker->eraseFromParent();
  }
  InstList.clear();
}

void BasicBlock::setParent(Function *parent) {
  // Set Parent=parent, updating instruction symtab entries as appropriate.
  InstList.setSymTabObject(&Parent, parent);
}

iterator_range<filter_iterator<BasicBlock::const_iterator,
                               std::function<bool(const Instruction &)>>>
BasicBlock::instructionsWithoutDebug(bool SkipPseudoOp) const {
  std::function<bool(const Instruction &)> Fn = [=](const Instruction &I) {
    return !isa<DbgInfoIntrinsic>(I) &&
           !(SkipPseudoOp && isa<PseudoProbeInst>(I));
  };
  return make_filter_range(*this, Fn);
}

iterator_range<
    filter_iterator<BasicBlock::iterator, std::function<bool(Instruction &)>>>
BasicBlock::instructionsWithoutDebug(bool SkipPseudoOp) {
  std::function<bool(Instruction &)> Fn = [=](Instruction &I) {
    return !isa<DbgInfoIntrinsic>(I) &&
           !(SkipPseudoOp && isa<PseudoProbeInst>(I));
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

void BasicBlock::moveBefore(SymbolTableList<BasicBlock>::iterator MovePos) {
  getParent()->splice(MovePos, getParent(), getIterator());
}

void BasicBlock::moveAfter(BasicBlock *MovePos) {
  MovePos->getParent()->splice(++MovePos->getIterator(), getParent(),
                               getIterator());
}

const Module *BasicBlock::getModule() const {
  return getParent()->getParent();
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

const Instruction *BasicBlock::getFirstMayFaultInst() const {
  if (InstList.empty())
    return nullptr;
  for (const Instruction &I : *this)
    if (isa<LoadInst>(I) || isa<StoreInst>(I) || isa<CallBase>(I))
      return &I;
  return nullptr;
}

const Instruction* BasicBlock::getFirstNonPHI() const {
  for (const Instruction &I : *this)
    if (!isa<PHINode>(I))
      return &I;
  return nullptr;
}

BasicBlock::const_iterator BasicBlock::getFirstNonPHIIt() const {
  const Instruction *I = getFirstNonPHI();
  BasicBlock::const_iterator It = I->getIterator();
  // Set the head-inclusive bit to indicate that this iterator includes
  // any debug-info at the start of the block. This is a no-op unless the
  // appropriate CMake flag is set.
  It.setHeadBit(true);
  return It;
}

const Instruction *BasicBlock::getFirstNonPHIOrDbg(bool SkipPseudoOp) const {
  for (const Instruction &I : *this) {
    if (isa<PHINode>(I) || isa<DbgInfoIntrinsic>(I))
      continue;

    if (SkipPseudoOp && isa<PseudoProbeInst>(I))
      continue;

    return &I;
  }
  return nullptr;
}

const Instruction *
BasicBlock::getFirstNonPHIOrDbgOrLifetime(bool SkipPseudoOp) const {
  for (const Instruction &I : *this) {
    if (isa<PHINode>(I) || isa<DbgInfoIntrinsic>(I))
      continue;

    if (I.isLifetimeStartOrEnd())
      continue;

    if (SkipPseudoOp && isa<PseudoProbeInst>(I))
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
  // Set the head-inclusive bit to indicate that this iterator includes
  // any debug-info at the start of the block. This is a no-op unless the
  // appropriate CMake flag is set.
  InsertPt.setHeadBit(true);
  return InsertPt;
}

BasicBlock::const_iterator BasicBlock::getFirstNonPHIOrDbgOrAlloca() const {
  const Instruction *FirstNonPHI = getFirstNonPHI();
  if (!FirstNonPHI)
    return end();

  const_iterator InsertPt = FirstNonPHI->getIterator();
  if (InsertPt->isEHPad())
    ++InsertPt;

  if (isEntryBlock()) {
    const_iterator End = end();
    while (InsertPt != End &&
           (isa<AllocaInst>(*InsertPt) || isa<DbgInfoIntrinsic>(*InsertPt) ||
            isa<PseudoProbeInst>(*InsertPt))) {
      if (const AllocaInst *AI = dyn_cast<AllocaInst>(&*InsertPt)) {
        if (!AI->isStaticAlloca())
          break;
      }
      ++InsertPt;
    }
  }
  return InsertPt;
}

void BasicBlock::dropAllReferences() {
  for (Instruction &I : *this)
    I.dropAllReferences();
}

const BasicBlock *BasicBlock::getSinglePredecessor() const {
  const_pred_iterator PI = pred_begin(this), E = pred_end(this);
  if (PI == E) return nullptr;         // No preds.
  const BasicBlock *ThePred = *PI;
  ++PI;
  return (PI == E) ? ThePred : nullptr /*multiple preds*/;
}

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

void BasicBlock::removePredecessor(BasicBlock *Pred,
                                   bool KeepOneInputPHIs) {
  // Use hasNUsesOrMore to bound the cost of this assertion for complex CFGs.
  assert((hasNUsesOrMore(16) || llvm::is_contained(predecessors(this), Pred)) &&
         "Pred is not a predecessor!");

  // Return early if there are no PHI nodes to update.
  if (empty() || !isa<PHINode>(begin()))
    return;

  unsigned NumPreds = cast<PHINode>(front()).getNumIncomingValues();
  for (PHINode &Phi : make_early_inc_range(phis())) {
    Phi.removeIncomingValue(Pred, !KeepOneInputPHIs);
    if (KeepOneInputPHIs)
      continue;

    // If we have a single predecessor, removeIncomingValue may have erased the
    // PHI node itself.
    if (NumPreds == 1)
      continue;

    // Try to replace the PHI node with a constant value.
    if (Value *PhiConstant = Phi.hasConstantValue()) {
      Phi.replaceAllUsesWith(PhiConstant);
      Phi.eraseFromParent();
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

  // Instructions should not be hoisted across special terminators, which may
  // have side effects or return values.
  return !Term->isSpecialTerminator();
}

bool BasicBlock::isEntryBlock() const {
  const Function *F = getParent();
  assert(F && "Block must have a parent function to use this API");
  return this == &F->getEntryBlock();
}

BasicBlock *BasicBlock::splitBasicBlock(iterator I, const Twine &BBName,
                                        bool Before) {
  if (Before)
    return splitBasicBlockBefore(I, BBName);

  assert(getTerminator() && "Can't use splitBasicBlock on degenerate BB!");
  assert(I != InstList.end() &&
         "Trying to get me to create degenerate basic block!");

  BasicBlock *New = BasicBlock::Create(getContext(), BBName, getParent(),
                                       this->getNextNode());

  // Save DebugLoc of split point before invalidating iterator.
  DebugLoc Loc = I->getStableDebugLoc();
  // Move all of the specified instructions from the original basic block into
  // the new basic block.
  New->splice(New->end(), this, I, end());

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

BasicBlock *BasicBlock::splitBasicBlockBefore(iterator I, const Twine &BBName) {
  assert(getTerminator() &&
         "Can't use splitBasicBlockBefore on degenerate BB!");
  assert(I != InstList.end() &&
         "Trying to get me to create degenerate basic block!");

  assert((!isa<PHINode>(*I) || getSinglePredecessor()) &&
         "cannot split on multi incoming phis");

  BasicBlock *New = BasicBlock::Create(getContext(), BBName, getParent(), this);
  // Save DebugLoc of split point before invalidating iterator.
  DebugLoc Loc = I->getDebugLoc();
  // Move all of the specified instructions from the original basic block into
  // the new basic block.
  New->splice(New->end(), this, begin(), I);

  // Loop through all of the predecessors of the 'this' block (which will be the
  // predecessors of the New block), replace the specified successor 'this'
  // block to point at the New block and update any PHI nodes in 'this' block.
  // If there were PHI nodes in 'this' block, the PHI nodes are updated
  // to reflect that the incoming branches will be from the New block and not
  // from predecessors of the 'this' block.
  // Save predecessors to separate vector before modifying them.
  SmallVector<BasicBlock *, 4> Predecessors;
  for (BasicBlock *Pred : predecessors(this))
    Predecessors.push_back(Pred);
  for (BasicBlock *Pred : Predecessors) {
    Instruction *TI = Pred->getTerminator();
    TI->replaceSuccessorWith(this, New);
    this->replacePhiUsesWith(Pred, New);
  }
  // Add a branch instruction from  "New" to "this" Block.
  BranchInst *BI = BranchInst::Create(this, New);
  BI->setDebugLoc(Loc);

  return New;
}

BasicBlock::iterator BasicBlock::erase(BasicBlock::iterator FromIt,
                                       BasicBlock::iterator ToIt) {
  for (Instruction &I : make_early_inc_range(make_range(FromIt, ToIt)))
    I.eraseFromParent();
  return ToIt;
}

void BasicBlock::replacePhiUsesWith(BasicBlock *Old, BasicBlock *New) {
  // N.B. This might not be a complete BasicBlock, so don't assume
  // that it ends with a non-phi instruction.
  for (Instruction &I : *this) {
    PHINode *PN = dyn_cast<PHINode>(&I);
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
  for (BasicBlock *Succ : successors(TI))
    Succ->replacePhiUsesWith(Old, New);
}

void BasicBlock::replaceSuccessorsPhiUsesWith(BasicBlock *New) {
  this->replaceSuccessorsPhiUsesWith(this, New);
}

bool BasicBlock::isLandingPad() const {
  return isa<LandingPadInst>(getFirstNonPHI());
}

const LandingPadInst *BasicBlock::getLandingPadInst() const {
  return dyn_cast<LandingPadInst>(getFirstNonPHI());
}

std::optional<uint64_t> BasicBlock::getIrrLoopHeaderWeight() const {
  const Instruction *TI = getTerminator();
  if (MDNode *MDIrrLoopHeader =
      TI->getMetadata(LLVMContext::MD_irr_loop)) {
    MDString *MDName = cast<MDString>(MDIrrLoopHeader->getOperand(0));
    if (MDName->getString().equals("loop_header_weight")) {
      auto *CI = mdconst::extract<ConstantInt>(MDIrrLoopHeader->getOperand(1));
      return std::optional<uint64_t>(CI->getValue().getZExtValue());
    }
  }
  return std::nullopt;
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

  NumInstrRenumberings++;
}

void BasicBlock::flushTerminatorDbgValues() {
  // If we erase the terminator in a block, any DPValues will sink and "fall
  // off the end", existing after any terminator that gets inserted. With
  // dbg.value intrinsics we would just insert the terminator at end() and
  // the dbg.values would come before the terminator. With DPValues, we must
  // do this manually.
  // To get out of this unfortunate form, whenever we insert a terminator,
  // check whether there's anything trailing at the end and move those DPValues
  // in front of the terminator.

  // Do nothing if we're not in new debug-info format.
  if (!IsNewDbgInfoFormat)
    return;

  // If there's no terminator, there's nothing to do.
  Instruction *Term = getTerminator();
  if (!Term)
    return;

  // Are there any dangling DPValues?
  DPMarker *TrailingDPValues = getTrailingDPValues();
  if (!TrailingDPValues)
    return;

  // Transfer DPValues from the trailing position onto the terminator.
  createMarker(Term);
  Term->DbgMarker->absorbDebugValues(*TrailingDPValues, false);
  TrailingDPValues->eraseFromParent();
  deleteTrailingDPValues();
}

void BasicBlock::spliceDebugInfoEmptyBlock(BasicBlock::iterator Dest,
                                           BasicBlock *Src,
                                           BasicBlock::iterator First,
                                           BasicBlock::iterator Last) {
  // Imagine the folowing:
  //
  //   bb1:
  //     dbg.value(...
  //     ret i32 0
  //
  // If an optimisation pass attempts to splice the contents of the block from
  // BB1->begin() to BB1->getTerminator(), then the dbg.value will be
  // transferred to the destination.
  // However, in the "new" DPValue format for debug-info, that range is empty:
  // begin() returns an iterator to the terminator, as there will only be a
  // single instruction in the block. We must piece together from the bits set
  // in the iterators whether there was the intention to transfer any debug
  // info.

  // If we're not in "new" debug-info format, do nothing.
  if (!IsNewDbgInfoFormat)
    return;

  assert(First == Last);
  bool InsertAtHead = Dest.getHeadBit();
  bool ReadFromHead = First.getHeadBit();

  // If the source block is completely empty, including no terminator, then
  // transfer any trailing DPValues that are still hanging around. This can
  // occur when a block is optimised away and the terminator has been moved
  // somewhere else.
  if (Src->empty()) {
    assert(Dest != end() &&
           "Transferring trailing DPValues to another trailing position");
    DPMarker *SrcTrailingDPValues = Src->getTrailingDPValues();
    if (!SrcTrailingDPValues)
      return;

    Dest->adoptDbgValues(Src, Src->end(), InsertAtHead);
    // adoptDbgValues should have released the trailing DPValues.
    assert(!Src->getTrailingDPValues());
    return;
  }

  // There are instructions in this block; if the First iterator was
  // with begin() / getFirstInsertionPt() then the caller intended debug-info
  // at the start of the block to be transferred. Return otherwise.
  if (Src->empty() || First != Src->begin() || !ReadFromHead)
    return;

  // Is there actually anything to transfer?
  if (!First->hasDbgValues())
    return;

  createMarker(Dest)->absorbDebugValues(*First->DbgMarker, InsertAtHead);

  return;
}

void BasicBlock::spliceDebugInfo(BasicBlock::iterator Dest, BasicBlock *Src,
                                 BasicBlock::iterator First,
                                 BasicBlock::iterator Last) {
  /* Do a quick normalisation before calling the real splice implementation. We
     might be operating on a degenerate basic block that has no instructions
     in it, a legitimate transient state. In that case, Dest will be end() and
     any DPValues temporarily stored in the TrailingDPValues map in LLVMContext.
     We might illustrate it thus:

                         Dest
                           |
     this-block:    ~~~~~~~~
      Src-block:            ++++B---B---B---B:::C
                                |               |
                               First           Last

     However: does the caller expect the "~" DPValues to end up before or after
     the spliced segment? This is communciated in the "Head" bit of Dest, which
     signals whether the caller called begin() or end() on this block.

     If the head bit is set, then all is well, we leave DPValues trailing just
     like how dbg.value instructions would trail after instructions spliced to
     the beginning of this block.

     If the head bit isn't set, then try to jam the "~" DPValues onto the front
     of the First instruction, then splice like normal, which joins the "~"
     DPValues with the "+" DPValues. However if the "+" DPValues are supposed to
     be left behind in Src, then:
      * detach the "+" DPValues,
      * move the "~" DPValues onto First,
      * splice like normal,
      * replace the "+" DPValues onto the Last position.
     Complicated, but gets the job done. */

  // If we're inserting at end(), and not in front of dangling DPValues, then
  // move the DPValues onto "First". They'll then be moved naturally in the
  // splice process.
  DPMarker *MoreDanglingDPValues = nullptr;
  DPMarker *OurTrailingDPValues = getTrailingDPValues();
  if (Dest == end() && !Dest.getHeadBit() && OurTrailingDPValues) {
    // Are the "+" DPValues not supposed to move? If so, detach them
    // temporarily.
    if (!First.getHeadBit() && First->hasDbgValues()) {
      MoreDanglingDPValues = Src->getMarker(First);
      MoreDanglingDPValues->removeFromParent();
    }

    if (First->hasDbgValues()) {
      // Place them at the front, it would look like this:
      //            Dest
      //              |
      // this-block:
      // Src-block: ~~~~~~~~++++B---B---B---B:::C
      //                        |               |
      //                       First           Last
      First->adoptDbgValues(this, end(), true);
    } else {
      // No current marker, create one and absorb in. (FIXME: we can avoid an
      // allocation in the future).
      DPMarker *CurMarker = Src->createMarker(&*First);
      CurMarker->absorbDebugValues(*OurTrailingDPValues, false);
      OurTrailingDPValues->eraseFromParent();
    }
    deleteTrailingDPValues();
    First.setHeadBit(true);
  }

  // Call the main debug-info-splicing implementation.
  spliceDebugInfoImpl(Dest, Src, First, Last);

  // Do we have some "+" DPValues hanging around that weren't supposed to move,
  // and we detached to make things easier?
  if (!MoreDanglingDPValues)
    return;

  // FIXME: we could avoid an allocation here sometimes. (adoptDbgValues
  // requires an iterator).
  DPMarker *LastMarker = Src->createMarker(Last);
  LastMarker->absorbDebugValues(*MoreDanglingDPValues, true);
  MoreDanglingDPValues->eraseFromParent();
}

void BasicBlock::spliceDebugInfoImpl(BasicBlock::iterator Dest, BasicBlock *Src,
                                     BasicBlock::iterator First,
                                     BasicBlock::iterator Last) {
  // Find out where to _place_ these dbg.values; if InsertAtHead is specified,
  // this will be at the start of Dest's debug value range, otherwise this is
  // just Dest's marker.
  bool InsertAtHead = Dest.getHeadBit();
  bool ReadFromHead = First.getHeadBit();
  // Use this flag to signal the abnormal case, where we don't want to copy the
  // DPValues ahead of the "Last" position.
  bool ReadFromTail = !Last.getTailBit();
  bool LastIsEnd = (Last == Src->end());

  /*
    Here's an illustration of what we're about to do. We have two blocks, this
    and Src, and two segments of list. Each instruction is marked by a capital
    while potential DPValue debug-info is marked out by "-" characters and a few
    other special characters (+:=) where I want to highlight what's going on.

                                                 Dest
                                                   |
     this-block:    A----A----A                ====A----A----A----A---A---A
      Src-block                ++++B---B---B---B:::C
                                   |               |
                                  First           Last

    The splice method is going to take all the instructions from First up to
    (but not including) Last and insert them in _front_ of Dest, forming one
    long list. All the DPValues attached to instructions _between_ First and
    Last need no maintenence. However, we have to do special things with the
    DPValues marked with the +:= characters. We only have three positions:
    should the "+" DPValues be transferred, and if so to where? Do we move the
    ":" DPValues? Would they go in front of the "=" DPValues, or should the "="
    DPValues go before "+" DPValues?

    We're told which way it should be by the bits carried in the iterators. The
    "Head" bit indicates whether the specified position is supposed to be at the
    front of the attached DPValues (true) or not (false). The Tail bit is true
    on the other end of a range: is the range intended to include DPValues up to
    the end (false) or not (true).

    FIXME: the tail bit doesn't need to be distinct from the head bit, we could
    combine them.

    Here are some examples of different configurations:

      Dest.Head = true, First.Head = true, Last.Tail = false

      this-block:    A----A----A++++B---B---B---B:::====A----A----A----A---A---A
                                    |                   |
                                  First                Dest

    Wheras if we didn't want to read from the Src list,

      Dest.Head = true, First.Head = false, Last.Tail = false

      this-block:    A----A----AB---B---B---B:::====A----A----A----A---A---A
                                |                   |
                              First                Dest

    Or if we didn't want to insert at the head of Dest:

      Dest.Head = false, First.Head = false, Last.Tail = false

      this-block:    A----A----A====B---B---B---B:::A----A----A----A---A---A
                                    |               |
                                  First            Dest

    Tests for these various configurations can be found in the unit test file
    BasicBlockDbgInfoTest.cpp.

   */

  // Detach the marker at Dest -- this lets us move the "====" DPValues around.
  DPMarker *DestMarker = nullptr;
  if (Dest != end()) {
    if ((DestMarker = getMarker(Dest)))
      DestMarker->removeFromParent();
  }

  // If we're moving the tail range of DPValues (":::"), absorb them into the
  // front of the DPValues at Dest.
  if (ReadFromTail && Src->getMarker(Last)) {
    DPMarker *FromLast = Src->getMarker(Last);
    if (LastIsEnd) {
      Dest->adoptDbgValues(Src, Last, true);
      // adoptDbgValues will release any trailers.
      assert(!Src->getTrailingDPValues());
    } else {
      // FIXME: can we use adoptDbgValues here to reduce allocations?
      DPMarker *OntoDest = createMarker(Dest);
      OntoDest->absorbDebugValues(*FromLast, true);
    }
  }

  // If we're _not_ reading from the head of First, i.e. the "++++" DPValues,
  // move their markers onto Last. They remain in the Src block. No action
  // needed.
  if (!ReadFromHead && First->hasDbgValues()) {
    if (Last != Src->end()) {
      Last->adoptDbgValues(Src, First, true);
    } else {
      DPMarker *OntoLast = Src->createMarker(Last);
      DPMarker *FromFirst = Src->createMarker(First);
      // Always insert at front of Last.
      OntoLast->absorbDebugValues(*FromFirst, true);
    }
  }

  // Finally, do something with the "====" DPValues we detached.
  if (DestMarker) {
    if (InsertAtHead) {
      // Insert them at the end of the DPValues at Dest. The "::::" DPValues
      // might be in front of them.
      DPMarker *NewDestMarker = createMarker(Dest);
      NewDestMarker->absorbDebugValues(*DestMarker, false);
    } else {
      // Insert them right at the start of the range we moved, ahead of First
      // and the "++++" DPValues.
      DPMarker *FirstMarker = createMarker(First);
      FirstMarker->absorbDebugValues(*DestMarker, true);
    }
    DestMarker->eraseFromParent();
  } else if (Dest == end() && !InsertAtHead) {
    // In the rare circumstance where we insert at end(), and we did not
    // generate the iterator with begin() / getFirstInsertionPt(), it means
    // any trailing debug-info at the end of the block would "normally" have
    // been pushed in front of "First". Move it there now.
    DPMarker *FirstMarker = getMarker(First);
    DPMarker *TrailingDPValues = getTrailingDPValues();
    if (TrailingDPValues) {
      FirstMarker->absorbDebugValues(*TrailingDPValues, true);
      TrailingDPValues->eraseFromParent();
      deleteTrailingDPValues();
    }
  }
}

void BasicBlock::splice(iterator Dest, BasicBlock *Src, iterator First,
                        iterator Last) {
  assert(Src->IsNewDbgInfoFormat == IsNewDbgInfoFormat);

#ifdef EXPENSIVE_CHECKS
  // Check that First is before Last.
  auto FromBBEnd = Src->end();
  for (auto It = First; It != Last; ++It)
    assert(It != FromBBEnd && "FromBeginIt not before FromEndIt!");
#endif // EXPENSIVE_CHECKS

  // Lots of horrible special casing for empty transfers: the dbg.values between
  // two positions could be spliced in dbg.value mode.
  if (First == Last) {
    spliceDebugInfoEmptyBlock(Dest, Src, First, Last);
    return;
  }

  // Handle non-instr debug-info specific juggling.
  if (IsNewDbgInfoFormat)
    spliceDebugInfo(Dest, Src, First, Last);

  // And move the instructions.
  getInstList().splice(Dest, Src->getInstList(), First, Last);

  flushTerminatorDbgValues();
}

void BasicBlock::insertDPValueAfter(DbgRecord *DPV, Instruction *I) {
  assert(IsNewDbgInfoFormat);
  assert(I->getParent() == this);

  iterator NextIt = std::next(I->getIterator());
  DPMarker *NextMarker = createMarker(NextIt);
  NextMarker->insertDPValue(DPV, true);
}

void BasicBlock::insertDPValueBefore(DbgRecord *DPV,
                                     InstListType::iterator Where) {
  // We should never directly insert at the end of the block, new DPValues
  // shouldn't be generated at times when there's no terminator.
  assert(Where != end());
  assert(Where->getParent() == this);
  if (!Where->DbgMarker)
    createMarker(Where);
  bool InsertAtHead = Where.getHeadBit();
  createMarker(&*Where);
  Where->DbgMarker->insertDPValue(DPV, InsertAtHead);
}

DPMarker *BasicBlock::getNextMarker(Instruction *I) {
  return getMarker(std::next(I->getIterator()));
}

DPMarker *BasicBlock::getMarker(InstListType::iterator It) {
  if (It == end()) {
    DPMarker *DPM = getTrailingDPValues();
    return DPM;
  }
  return It->DbgMarker;
}

void BasicBlock::reinsertInstInDPValues(
    Instruction *I, std::optional<DPValue::self_iterator> Pos) {
  // "I" was originally removed from a position where it was
  // immediately in front of Pos. Any DPValues on that position then "fell down"
  // onto Pos. "I" has been re-inserted at the front of that wedge of DPValues,
  // shuffle them around to represent the original positioning. To illustrate:
  //
  //   Instructions:  I1---I---I0
  //       DPValues:    DDD DDD
  //
  // Instruction "I" removed,
  //
  //   Instructions:  I1------I0
  //       DPValues:    DDDDDD
  //                       ^Pos
  //
  // Instruction "I" re-inserted (now):
  //
  //   Instructions:  I1---I------I0
  //       DPValues:        DDDDDD
  //                           ^Pos
  //
  // After this method completes:
  //
  //   Instructions:  I1---I---I0
  //       DPValues:    DDD DDD

  // This happens if there were no DPValues on I0. Are there now DPValues there?
  if (!Pos) {
    DPMarker *NextMarker = getNextMarker(I);
    if (!NextMarker)
      return;
    if (NextMarker->StoredDPValues.empty())
      return;
    // There are DPMarkers there now -- they fell down from "I".
    DPMarker *ThisMarker = createMarker(I);
    ThisMarker->absorbDebugValues(*NextMarker, false);
    return;
  }

  // Is there even a range of DPValues to move?
  DPMarker *DPM = (*Pos)->getMarker();
  auto Range = make_range(DPM->StoredDPValues.begin(), (*Pos));
  if (Range.begin() == Range.end())
    return;

  // Otherwise: splice.
  DPMarker *ThisMarker = createMarker(I);
  assert(ThisMarker->StoredDPValues.empty());
  ThisMarker->absorbDebugValues(Range, *DPM, true);
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

void BasicBlock::setTrailingDPValues(DPMarker *foo) {
  getContext().pImpl->setTrailingDPValues(this, foo);
}

DPMarker *BasicBlock::getTrailingDPValues() {
  return getContext().pImpl->getTrailingDPValues(this);
}

void BasicBlock::deleteTrailingDPValues() {
  getContext().pImpl->deleteTrailingDPValues(this);
}

