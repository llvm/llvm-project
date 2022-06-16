//===- VPlan.cpp - Vectorizer Plan ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This is the LLVM vectorization plan. It represents a candidate for
/// vectorization, allowing to plan and optimize how to vectorize a given loop
/// before generating LLVM-IR.
/// The vectorizer uses vectorization plans to estimate the costs of potential
/// candidates and if profitable to execute the desired plan, generating vector
/// LLVM-IR code.
///
//===----------------------------------------------------------------------===//

#include "VPlan.h"
#include "VPlanDominatorTree.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/GenericDomTreeConstruction.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include <cassert>
#include <string>
#include <vector>

using namespace llvm;
extern cl::opt<bool> EnableVPlanNativePath;

#define DEBUG_TYPE "vplan"

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
raw_ostream &llvm::operator<<(raw_ostream &OS, const VPValue &V) {
  const VPInstruction *Instr = dyn_cast<VPInstruction>(&V);
  VPSlotTracker SlotTracker(
      (Instr && Instr->getParent()) ? Instr->getParent()->getPlan() : nullptr);
  V.print(OS, SlotTracker);
  return OS;
}
#endif

Value *VPLane::getAsRuntimeExpr(IRBuilderBase &Builder,
                                const ElementCount &VF) const {
  switch (LaneKind) {
  case VPLane::Kind::ScalableLast:
    // Lane = RuntimeVF - VF.getKnownMinValue() + Lane
    return Builder.CreateSub(getRuntimeVF(Builder, Builder.getInt32Ty(), VF),
                             Builder.getInt32(VF.getKnownMinValue() - Lane));
  case VPLane::Kind::First:
    return Builder.getInt32(Lane);
  }
  llvm_unreachable("Unknown lane kind");
}

VPValue::VPValue(const unsigned char SC, Value *UV, VPDef *Def)
    : SubclassID(SC), UnderlyingVal(UV), Def(Def) {
  if (Def)
    Def->addDefinedValue(this);
}

VPValue::~VPValue() {
  assert(Users.empty() && "trying to delete a VPValue with remaining users");
  if (Def)
    Def->removeDefinedValue(this);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPValue::print(raw_ostream &OS, VPSlotTracker &SlotTracker) const {
  if (const VPRecipeBase *R = dyn_cast_or_null<VPRecipeBase>(Def))
    R->print(OS, "", SlotTracker);
  else
    printAsOperand(OS, SlotTracker);
}

void VPValue::dump() const {
  const VPRecipeBase *Instr = dyn_cast_or_null<VPRecipeBase>(this->Def);
  VPSlotTracker SlotTracker(
      (Instr && Instr->getParent()) ? Instr->getParent()->getPlan() : nullptr);
  print(dbgs(), SlotTracker);
  dbgs() << "\n";
}

void VPDef::dump() const {
  const VPRecipeBase *Instr = dyn_cast_or_null<VPRecipeBase>(this);
  VPSlotTracker SlotTracker(
      (Instr && Instr->getParent()) ? Instr->getParent()->getPlan() : nullptr);
  print(dbgs(), "", SlotTracker);
  dbgs() << "\n";
}
#endif

// Get the top-most entry block of \p Start. This is the entry block of the
// containing VPlan. This function is templated to support both const and non-const blocks
template <typename T> static T *getPlanEntry(T *Start) {
  T *Next = Start;
  T *Current = Start;
  while ((Next = Next->getParent()))
    Current = Next;

  SmallSetVector<T *, 8> WorkList;
  WorkList.insert(Current);

  for (unsigned i = 0; i < WorkList.size(); i++) {
    T *Current = WorkList[i];
    if (Current->getNumPredecessors() == 0)
      return Current;
    auto &Predecessors = Current->getPredecessors();
    WorkList.insert(Predecessors.begin(), Predecessors.end());
  }

  llvm_unreachable("VPlan without any entry node without predecessors");
}

VPlan *VPBlockBase::getPlan() { return getPlanEntry(this)->Plan; }

const VPlan *VPBlockBase::getPlan() const { return getPlanEntry(this)->Plan; }

/// \return the VPBasicBlock that is the entry of Block, possibly indirectly.
const VPBasicBlock *VPBlockBase::getEntryBasicBlock() const {
  const VPBlockBase *Block = this;
  while (const VPRegionBlock *Region = dyn_cast<VPRegionBlock>(Block))
    Block = Region->getEntry();
  return cast<VPBasicBlock>(Block);
}

VPBasicBlock *VPBlockBase::getEntryBasicBlock() {
  VPBlockBase *Block = this;
  while (VPRegionBlock *Region = dyn_cast<VPRegionBlock>(Block))
    Block = Region->getEntry();
  return cast<VPBasicBlock>(Block);
}

void VPBlockBase::setPlan(VPlan *ParentPlan) {
  assert(ParentPlan->getEntry() == this &&
         "Can only set plan on its entry block.");
  Plan = ParentPlan;
}

/// \return the VPBasicBlock that is the exit of Block, possibly indirectly.
const VPBasicBlock *VPBlockBase::getExitingBasicBlock() const {
  const VPBlockBase *Block = this;
  while (const VPRegionBlock *Region = dyn_cast<VPRegionBlock>(Block))
    Block = Region->getExiting();
  return cast<VPBasicBlock>(Block);
}

VPBasicBlock *VPBlockBase::getExitingBasicBlock() {
  VPBlockBase *Block = this;
  while (VPRegionBlock *Region = dyn_cast<VPRegionBlock>(Block))
    Block = Region->getExiting();
  return cast<VPBasicBlock>(Block);
}

VPBlockBase *VPBlockBase::getEnclosingBlockWithSuccessors() {
  if (!Successors.empty() || !Parent)
    return this;
  assert(Parent->getExiting() == this &&
         "Block w/o successors not the exiting block of its parent.");
  return Parent->getEnclosingBlockWithSuccessors();
}

VPBlockBase *VPBlockBase::getEnclosingBlockWithPredecessors() {
  if (!Predecessors.empty() || !Parent)
    return this;
  assert(Parent->getEntry() == this &&
         "Block w/o predecessors not the entry of its parent.");
  return Parent->getEnclosingBlockWithPredecessors();
}

void VPBlockBase::deleteCFG(VPBlockBase *Entry) {
  SmallVector<VPBlockBase *, 8> Blocks(depth_first(Entry));

  for (VPBlockBase *Block : Blocks)
    delete Block;
}

VPBasicBlock::iterator VPBasicBlock::getFirstNonPhi() {
  iterator It = begin();
  while (It != end() && It->isPhi())
    It++;
  return It;
}

Value *VPTransformState::get(VPValue *Def, const VPIteration &Instance) {
  if (!Def->getDef())
    return Def->getLiveInIRValue();

  if (hasScalarValue(Def, Instance)) {
    return Data
        .PerPartScalars[Def][Instance.Part][Instance.Lane.mapToCacheIndex(VF)];
  }

  assert(hasVectorValue(Def, Instance.Part));
  auto *VecPart = Data.PerPartOutput[Def][Instance.Part];
  if (!VecPart->getType()->isVectorTy()) {
    assert(Instance.Lane.isFirstLane() && "cannot get lane > 0 for scalar");
    return VecPart;
  }
  // TODO: Cache created scalar values.
  Value *Lane = Instance.Lane.getAsRuntimeExpr(Builder, VF);
  auto *Extract = Builder.CreateExtractElement(VecPart, Lane);
  // set(Def, Extract, Instance);
  return Extract;
}
BasicBlock *VPTransformState::CFGState::getPreheaderBBFor(VPRecipeBase *R) {
  VPRegionBlock *LoopRegion = R->getParent()->getEnclosingLoopRegion();
  return VPBB2IRBB[LoopRegion->getPreheaderVPBB()];
}

BasicBlock *
VPBasicBlock::createEmptyBasicBlock(VPTransformState::CFGState &CFG) {
  // BB stands for IR BasicBlocks. VPBB stands for VPlan VPBasicBlocks.
  // Pred stands for Predessor. Prev stands for Previous - last visited/created.
  BasicBlock *PrevBB = CFG.PrevBB;
  BasicBlock *NewBB = BasicBlock::Create(PrevBB->getContext(), getName(),
                                         PrevBB->getParent(), CFG.ExitBB);
  LLVM_DEBUG(dbgs() << "LV: created " << NewBB->getName() << '\n');

  // Hook up the new basic block to its predecessors.
  for (VPBlockBase *PredVPBlock : getHierarchicalPredecessors()) {
    VPBasicBlock *PredVPBB = PredVPBlock->getExitingBasicBlock();
    auto &PredVPSuccessors = PredVPBB->getHierarchicalSuccessors();
    BasicBlock *PredBB = CFG.VPBB2IRBB[PredVPBB];

    assert(PredBB && "Predecessor basic-block not found building successor.");
    auto *PredBBTerminator = PredBB->getTerminator();
    LLVM_DEBUG(dbgs() << "LV: draw edge from" << PredBB->getName() << '\n');

    auto *TermBr = dyn_cast<BranchInst>(PredBBTerminator);
    if (isa<UnreachableInst>(PredBBTerminator)) {
      assert(PredVPSuccessors.size() == 1 &&
             "Predecessor ending w/o branch must have single successor.");
      DebugLoc DL = PredBBTerminator->getDebugLoc();
      PredBBTerminator->eraseFromParent();
      auto *Br = BranchInst::Create(NewBB, PredBB);
      Br->setDebugLoc(DL);
    } else if (TermBr && !TermBr->isConditional()) {
      TermBr->setSuccessor(0, NewBB);
    } else {
      // Set each forward successor here when it is created, excluding
      // backedges. A backward successor is set when the branch is created.
      unsigned idx = PredVPSuccessors.front() == this ? 0 : 1;
      assert(!TermBr->getSuccessor(idx) &&
             "Trying to reset an existing successor block.");
      TermBr->setSuccessor(idx, NewBB);
    }
  }
  return NewBB;
}

void VPBasicBlock::execute(VPTransformState *State) {
  bool Replica = State->Instance && !State->Instance->isFirstIteration();
  VPBasicBlock *PrevVPBB = State->CFG.PrevVPBB;
  VPBlockBase *SingleHPred = nullptr;
  BasicBlock *NewBB = State->CFG.PrevBB; // Reuse it if possible.

  auto IsLoopRegion = [](VPBlockBase *BB) {
    auto *R = dyn_cast<VPRegionBlock>(BB);
    return R && !R->isReplicator();
  };

  // 1. Create an IR basic block, or reuse the last one or ExitBB if possible.
  if (getPlan()->getVectorLoopRegion()->getSingleSuccessor() == this) {
    // ExitBB can be re-used for the exit block of the Plan.
    NewBB = State->CFG.ExitBB;
    State->CFG.PrevBB = NewBB;

    // Update the branch instruction in the predecessor to branch to ExitBB.
    VPBlockBase *PredVPB = getSingleHierarchicalPredecessor();
    VPBasicBlock *ExitingVPBB = PredVPB->getExitingBasicBlock();
    assert(PredVPB->getSingleSuccessor() == this &&
           "predecessor must have the current block as only successor");
    BasicBlock *ExitingBB = State->CFG.VPBB2IRBB[ExitingVPBB];
    // The Exit block of a loop is always set to be successor 0 of the Exiting
    // block.
    cast<BranchInst>(ExitingBB->getTerminator())->setSuccessor(0, NewBB);
  } else if (PrevVPBB && /* A */
             !((SingleHPred = getSingleHierarchicalPredecessor()) &&
               SingleHPred->getExitingBasicBlock() == PrevVPBB &&
               PrevVPBB->getSingleHierarchicalSuccessor() &&
               (SingleHPred->getParent() == getEnclosingLoopRegion() &&
                !IsLoopRegion(SingleHPred))) &&         /* B */
             !(Replica && getPredecessors().empty())) { /* C */
    // The last IR basic block is reused, as an optimization, in three cases:
    // A. the first VPBB reuses the loop pre-header BB - when PrevVPBB is null;
    // B. when the current VPBB has a single (hierarchical) predecessor which
    //    is PrevVPBB and the latter has a single (hierarchical) successor which
    //    both are in the same non-replicator region; and
    // C. when the current VPBB is an entry of a region replica - where PrevVPBB
    //    is the exiting VPBB of this region from a previous instance, or the
    //    predecessor of this region.

    NewBB = createEmptyBasicBlock(State->CFG);
    State->Builder.SetInsertPoint(NewBB);
    // Temporarily terminate with unreachable until CFG is rewired.
    UnreachableInst *Terminator = State->Builder.CreateUnreachable();
    // Register NewBB in its loop. In innermost loops its the same for all
    // BB's.
    if (State->CurrentVectorLoop)
      State->CurrentVectorLoop->addBasicBlockToLoop(NewBB, *State->LI);
    State->Builder.SetInsertPoint(Terminator);
    State->CFG.PrevBB = NewBB;
  }

  // 2. Fill the IR basic block with IR instructions.
  LLVM_DEBUG(dbgs() << "LV: vectorizing VPBB:" << getName()
                    << " in BB:" << NewBB->getName() << '\n');

  State->CFG.VPBB2IRBB[this] = NewBB;
  State->CFG.PrevVPBB = this;

  for (VPRecipeBase &Recipe : Recipes)
    Recipe.execute(*State);

  LLVM_DEBUG(dbgs() << "LV: filled BB:" << *NewBB);
}

void VPBasicBlock::dropAllReferences(VPValue *NewValue) {
  for (VPRecipeBase &R : Recipes) {
    for (auto *Def : R.definedValues())
      Def->replaceAllUsesWith(NewValue);

    for (unsigned I = 0, E = R.getNumOperands(); I != E; I++)
      R.setOperand(I, NewValue);
  }
}

VPBasicBlock *VPBasicBlock::splitAt(iterator SplitAt) {
  assert((SplitAt == end() || SplitAt->getParent() == this) &&
         "can only split at a position in the same block");

  SmallVector<VPBlockBase *, 2> Succs(successors());
  // First, disconnect the current block from its successors.
  for (VPBlockBase *Succ : Succs)
    VPBlockUtils::disconnectBlocks(this, Succ);

  // Create new empty block after the block to split.
  auto *SplitBlock = new VPBasicBlock(getName() + ".split");
  VPBlockUtils::insertBlockAfter(SplitBlock, this);

  // Add successors for block to split to new block.
  for (VPBlockBase *Succ : Succs)
    VPBlockUtils::connectBlocks(SplitBlock, Succ);

  // Finally, move the recipes starting at SplitAt to new block.
  for (VPRecipeBase &ToMove :
       make_early_inc_range(make_range(SplitAt, this->end())))
    ToMove.moveBefore(*SplitBlock, SplitBlock->end());

  return SplitBlock;
}

VPRegionBlock *VPBasicBlock::getEnclosingLoopRegion() {
  VPRegionBlock *P = getParent();
  if (P && P->isReplicator()) {
    P = P->getParent();
    assert(!cast<VPRegionBlock>(P)->isReplicator() &&
           "unexpected nested replicate regions");
  }
  return P;
}

static bool hasConditionalTerminator(const VPBasicBlock *VPBB) {
  if (VPBB->empty()) {
    assert(
        VPBB->getNumSuccessors() < 2 &&
        "block with multiple successors doesn't have a recipe as terminator");
    return false;
  }

  const VPRecipeBase *R = &VPBB->back();
  auto *VPI = dyn_cast<VPInstruction>(R);
  bool IsCondBranch =
      isa<VPBranchOnMaskRecipe>(R) ||
      (VPI && (VPI->getOpcode() == VPInstruction::BranchOnCond ||
               VPI->getOpcode() == VPInstruction::BranchOnCount));
  (void)IsCondBranch;

  if (VPBB->getNumSuccessors() >= 2 || VPBB->isExiting()) {
    assert(IsCondBranch && "block with multiple successors not terminated by "
                           "conditional branch recipe");

    return true;
  }

  assert(
      !IsCondBranch &&
      "block with 0 or 1 successors terminated by conditional branch recipe");
  return false;
}

VPRecipeBase *VPBasicBlock::getTerminator() {
  if (hasConditionalTerminator(this))
    return &back();
  return nullptr;
}

const VPRecipeBase *VPBasicBlock::getTerminator() const {
  if (hasConditionalTerminator(this))
    return &back();
  return nullptr;
}

bool VPBasicBlock::isExiting() const {
  return getParent()->getExitingBasicBlock() == this;
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPBlockBase::printSuccessors(raw_ostream &O, const Twine &Indent) const {
  if (getSuccessors().empty()) {
    O << Indent << "No successors\n";
  } else {
    O << Indent << "Successor(s): ";
    ListSeparator LS;
    for (auto *Succ : getSuccessors())
      O << LS << Succ->getName();
    O << '\n';
  }
}

void VPBasicBlock::print(raw_ostream &O, const Twine &Indent,
                         VPSlotTracker &SlotTracker) const {
  O << Indent << getName() << ":\n";

  auto RecipeIndent = Indent + "  ";
  for (const VPRecipeBase &Recipe : *this) {
    Recipe.print(O, RecipeIndent, SlotTracker);
    O << '\n';
  }

  printSuccessors(O, Indent);
}
#endif

void VPRegionBlock::dropAllReferences(VPValue *NewValue) {
  for (VPBlockBase *Block : depth_first(Entry))
    // Drop all references in VPBasicBlocks and replace all uses with
    // DummyValue.
    Block->dropAllReferences(NewValue);
}

void VPRegionBlock::execute(VPTransformState *State) {
  ReversePostOrderTraversal<VPBlockBase *> RPOT(Entry);

  if (!isReplicator()) {
    // Create and register the new vector loop.
    Loop *PrevLoop = State->CurrentVectorLoop;
    State->CurrentVectorLoop = State->LI->AllocateLoop();
    BasicBlock *VectorPH = State->CFG.VPBB2IRBB[getPreheaderVPBB()];
    Loop *ParentLoop = State->LI->getLoopFor(VectorPH);

    // Insert the new loop into the loop nest and register the new basic blocks
    // before calling any utilities such as SCEV that require valid LoopInfo.
    if (ParentLoop)
      ParentLoop->addChildLoop(State->CurrentVectorLoop);
    else
      State->LI->addTopLevelLoop(State->CurrentVectorLoop);

    // Visit the VPBlocks connected to "this", starting from it.
    for (VPBlockBase *Block : RPOT) {
      LLVM_DEBUG(dbgs() << "LV: VPBlock in RPO " << Block->getName() << '\n');
      Block->execute(State);
    }

    State->CurrentVectorLoop = PrevLoop;
    return;
  }

  assert(!State->Instance && "Replicating a Region with non-null instance.");

  // Enter replicating mode.
  State->Instance = VPIteration(0, 0);

  for (unsigned Part = 0, UF = State->UF; Part < UF; ++Part) {
    State->Instance->Part = Part;
    assert(!State->VF.isScalable() && "VF is assumed to be non scalable.");
    for (unsigned Lane = 0, VF = State->VF.getKnownMinValue(); Lane < VF;
         ++Lane) {
      State->Instance->Lane = VPLane(Lane, VPLane::Kind::First);
      // Visit the VPBlocks connected to \p this, starting from it.
      for (VPBlockBase *Block : RPOT) {
        LLVM_DEBUG(dbgs() << "LV: VPBlock in RPO " << Block->getName() << '\n');
        Block->execute(State);
      }
    }
  }

  // Exit replicating mode.
  State->Instance.reset();
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPRegionBlock::print(raw_ostream &O, const Twine &Indent,
                          VPSlotTracker &SlotTracker) const {
  O << Indent << (isReplicator() ? "<xVFxUF> " : "<x1> ") << getName() << ": {";
  auto NewIndent = Indent + "  ";
  for (auto *BlockBase : depth_first(Entry)) {
    O << '\n';
    BlockBase->print(O, NewIndent, SlotTracker);
  }
  O << Indent << "}\n";

  printSuccessors(O, Indent);
}
#endif

bool VPRecipeBase::mayWriteToMemory() const {
  switch (getVPDefID()) {
  case VPWidenMemoryInstructionSC: {
    return cast<VPWidenMemoryInstructionRecipe>(this)->isStore();
  }
  case VPReplicateSC:
  case VPWidenCallSC:
    return cast<Instruction>(getVPSingleValue()->getUnderlyingValue())
        ->mayWriteToMemory();
  case VPBranchOnMaskSC:
    return false;
  case VPWidenIntOrFpInductionSC:
  case VPWidenCanonicalIVSC:
  case VPWidenPHISC:
  case VPBlendSC:
  case VPWidenSC:
  case VPWidenGEPSC:
  case VPReductionSC:
  case VPWidenSelectSC: {
    const Instruction *I =
        dyn_cast_or_null<Instruction>(getVPSingleValue()->getUnderlyingValue());
    (void)I;
    assert((!I || !I->mayWriteToMemory()) &&
           "underlying instruction may write to memory");
    return false;
  }
  default:
    return true;
  }
}

bool VPRecipeBase::mayReadFromMemory() const {
  switch (getVPDefID()) {
  case VPWidenMemoryInstructionSC: {
    return !cast<VPWidenMemoryInstructionRecipe>(this)->isStore();
  }
  case VPReplicateSC:
  case VPWidenCallSC:
    return cast<Instruction>(getVPSingleValue()->getUnderlyingValue())
        ->mayReadFromMemory();
  case VPBranchOnMaskSC:
    return false;
  case VPWidenIntOrFpInductionSC:
  case VPWidenCanonicalIVSC:
  case VPWidenPHISC:
  case VPBlendSC:
  case VPWidenSC:
  case VPWidenGEPSC:
  case VPReductionSC:
  case VPWidenSelectSC: {
    const Instruction *I =
        dyn_cast_or_null<Instruction>(getVPSingleValue()->getUnderlyingValue());
    (void)I;
    assert((!I || !I->mayReadFromMemory()) &&
           "underlying instruction may read from memory");
    return false;
  }
  default:
    return true;
  }
}

bool VPRecipeBase::mayHaveSideEffects() const {
  switch (getVPDefID()) {
  case VPBranchOnMaskSC:
    return false;
  case VPWidenIntOrFpInductionSC:
  case VPWidenPointerInductionSC:
  case VPWidenCanonicalIVSC:
  case VPWidenPHISC:
  case VPBlendSC:
  case VPWidenSC:
  case VPWidenGEPSC:
  case VPReductionSC:
  case VPWidenSelectSC:
  case VPScalarIVStepsSC: {
    const Instruction *I =
        dyn_cast_or_null<Instruction>(getVPSingleValue()->getUnderlyingValue());
    (void)I;
    assert((!I || !I->mayHaveSideEffects()) &&
           "underlying instruction has side-effects");
    return false;
  }
  case VPReplicateSC: {
    auto *R = cast<VPReplicateRecipe>(this);
    return R->getUnderlyingInstr()->mayHaveSideEffects();
  }
  default:
    return true;
  }
}

void VPLiveOut::fixPhi(VPlan &Plan, VPTransformState &State) {
  auto Lane = VPLane::getLastLaneForVF(State.VF);
  VPValue *ExitValue = getOperand(0);
  if (Plan.isUniformAfterVectorization(ExitValue))
    Lane = VPLane::getFirstLane();
  Phi->addIncoming(State.get(ExitValue, VPIteration(State.UF - 1, Lane)),
                   State.Builder.GetInsertBlock());
}

void VPRecipeBase::insertBefore(VPRecipeBase *InsertPos) {
  assert(!Parent && "Recipe already in some VPBasicBlock");
  assert(InsertPos->getParent() &&
         "Insertion position not in any VPBasicBlock");
  Parent = InsertPos->getParent();
  Parent->getRecipeList().insert(InsertPos->getIterator(), this);
}

void VPRecipeBase::insertBefore(VPBasicBlock &BB,
                                iplist<VPRecipeBase>::iterator I) {
  assert(!Parent && "Recipe already in some VPBasicBlock");
  assert(I == BB.end() || I->getParent() == &BB);
  Parent = &BB;
  BB.getRecipeList().insert(I, this);
}

void VPRecipeBase::insertAfter(VPRecipeBase *InsertPos) {
  assert(!Parent && "Recipe already in some VPBasicBlock");
  assert(InsertPos->getParent() &&
         "Insertion position not in any VPBasicBlock");
  Parent = InsertPos->getParent();
  Parent->getRecipeList().insertAfter(InsertPos->getIterator(), this);
}

void VPRecipeBase::removeFromParent() {
  assert(getParent() && "Recipe not in any VPBasicBlock");
  getParent()->getRecipeList().remove(getIterator());
  Parent = nullptr;
}

iplist<VPRecipeBase>::iterator VPRecipeBase::eraseFromParent() {
  assert(getParent() && "Recipe not in any VPBasicBlock");
  return getParent()->getRecipeList().erase(getIterator());
}

void VPRecipeBase::moveAfter(VPRecipeBase *InsertPos) {
  removeFromParent();
  insertAfter(InsertPos);
}

void VPRecipeBase::moveBefore(VPBasicBlock &BB,
                              iplist<VPRecipeBase>::iterator I) {
  removeFromParent();
  insertBefore(BB, I);
}

void VPInstruction::generateInstruction(VPTransformState &State,
                                        unsigned Part) {
  IRBuilderBase &Builder = State.Builder;
  Builder.SetCurrentDebugLocation(DL);

  if (Instruction::isBinaryOp(getOpcode())) {
    Value *A = State.get(getOperand(0), Part);
    Value *B = State.get(getOperand(1), Part);
    Value *V = Builder.CreateBinOp((Instruction::BinaryOps)getOpcode(), A, B);
    State.set(this, V, Part);
    return;
  }

  switch (getOpcode()) {
  case VPInstruction::Not: {
    Value *A = State.get(getOperand(0), Part);
    Value *V = Builder.CreateNot(A);
    State.set(this, V, Part);
    break;
  }
  case VPInstruction::ICmpULE: {
    Value *IV = State.get(getOperand(0), Part);
    Value *TC = State.get(getOperand(1), Part);
    Value *V = Builder.CreateICmpULE(IV, TC);
    State.set(this, V, Part);
    break;
  }
  case Instruction::Select: {
    Value *Cond = State.get(getOperand(0), Part);
    Value *Op1 = State.get(getOperand(1), Part);
    Value *Op2 = State.get(getOperand(2), Part);
    Value *V = Builder.CreateSelect(Cond, Op1, Op2);
    State.set(this, V, Part);
    break;
  }
  case VPInstruction::ActiveLaneMask: {
    // Get first lane of vector induction variable.
    Value *VIVElem0 = State.get(getOperand(0), VPIteration(Part, 0));
    // Get the original loop tripcount.
    Value *ScalarTC = State.get(getOperand(1), Part);

    auto *Int1Ty = Type::getInt1Ty(Builder.getContext());
    auto *PredTy = VectorType::get(Int1Ty, State.VF);
    Instruction *Call = Builder.CreateIntrinsic(
        Intrinsic::get_active_lane_mask, {PredTy, ScalarTC->getType()},
        {VIVElem0, ScalarTC}, nullptr, "active.lane.mask");
    State.set(this, Call, Part);
    break;
  }
  case VPInstruction::FirstOrderRecurrenceSplice: {
    // Generate code to combine the previous and current values in vector v3.
    //
    //   vector.ph:
    //     v_init = vector(..., ..., ..., a[-1])
    //     br vector.body
    //
    //   vector.body
    //     i = phi [0, vector.ph], [i+4, vector.body]
    //     v1 = phi [v_init, vector.ph], [v2, vector.body]
    //     v2 = a[i, i+1, i+2, i+3];
    //     v3 = vector(v1(3), v2(0, 1, 2))

    // For the first part, use the recurrence phi (v1), otherwise v2.
    auto *V1 = State.get(getOperand(0), 0);
    Value *PartMinus1 = Part == 0 ? V1 : State.get(getOperand(1), Part - 1);
    if (!PartMinus1->getType()->isVectorTy()) {
      State.set(this, PartMinus1, Part);
    } else {
      Value *V2 = State.get(getOperand(1), Part);
      State.set(this, Builder.CreateVectorSplice(PartMinus1, V2, -1), Part);
    }
    break;
  }

  case VPInstruction::CanonicalIVIncrement:
  case VPInstruction::CanonicalIVIncrementNUW: {
    Value *Next = nullptr;
    if (Part == 0) {
      bool IsNUW = getOpcode() == VPInstruction::CanonicalIVIncrementNUW;
      auto *Phi = State.get(getOperand(0), 0);
      // The loop step is equal to the vectorization factor (num of SIMD
      // elements) times the unroll factor (num of SIMD instructions).
      Value *Step =
          createStepForVF(Builder, Phi->getType(), State.VF, State.UF);
      Next = Builder.CreateAdd(Phi, Step, "index.next", IsNUW, false);
    } else {
      Next = State.get(this, 0);
    }

    State.set(this, Next, Part);
    break;
  }
  case VPInstruction::BranchOnCond: {
    if (Part != 0)
      break;

    Value *Cond = State.get(getOperand(0), VPIteration(Part, 0));
    VPRegionBlock *ParentRegion = getParent()->getParent();
    VPBasicBlock *Header = ParentRegion->getEntryBasicBlock();

    // Replace the temporary unreachable terminator with a new conditional
    // branch, hooking it up to backward destination for exiting blocks now and
    // to forward destination(s) later when they are created.
    BranchInst *CondBr =
        Builder.CreateCondBr(Cond, Builder.GetInsertBlock(), nullptr);

    if (getParent()->isExiting())
      CondBr->setSuccessor(1, State.CFG.VPBB2IRBB[Header]);

    CondBr->setSuccessor(0, nullptr);
    Builder.GetInsertBlock()->getTerminator()->eraseFromParent();
    break;
  }
  case VPInstruction::BranchOnCount: {
    if (Part != 0)
      break;
    // First create the compare.
    Value *IV = State.get(getOperand(0), Part);
    Value *TC = State.get(getOperand(1), Part);
    Value *Cond = Builder.CreateICmpEQ(IV, TC);

    // Now create the branch.
    auto *Plan = getParent()->getPlan();
    VPRegionBlock *TopRegion = Plan->getVectorLoopRegion();
    VPBasicBlock *Header = TopRegion->getEntry()->getEntryBasicBlock();

    // Replace the temporary unreachable terminator with a new conditional
    // branch, hooking it up to backward destination (the header) now and to the
    // forward destination (the exit/middle block) later when it is created.
    // Note that CreateCondBr expects a valid BB as first argument, so we need
    // to set it to nullptr later.
    BranchInst *CondBr = Builder.CreateCondBr(Cond, Builder.GetInsertBlock(),
                                              State.CFG.VPBB2IRBB[Header]);
    CondBr->setSuccessor(0, nullptr);
    Builder.GetInsertBlock()->getTerminator()->eraseFromParent();
    break;
  }
  default:
    llvm_unreachable("Unsupported opcode for instruction");
  }
}

void VPInstruction::execute(VPTransformState &State) {
  assert(!State.Instance && "VPInstruction executing an Instance");
  IRBuilderBase::FastMathFlagGuard FMFGuard(State.Builder);
  State.Builder.setFastMathFlags(FMF);
  for (unsigned Part = 0; Part < State.UF; ++Part)
    generateInstruction(State, Part);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPInstruction::dump() const {
  VPSlotTracker SlotTracker(getParent()->getPlan());
  print(dbgs(), "", SlotTracker);
}

void VPInstruction::print(raw_ostream &O, const Twine &Indent,
                          VPSlotTracker &SlotTracker) const {
  O << Indent << "EMIT ";

  if (hasResult()) {
    printAsOperand(O, SlotTracker);
    O << " = ";
  }

  switch (getOpcode()) {
  case VPInstruction::Not:
    O << "not";
    break;
  case VPInstruction::ICmpULE:
    O << "icmp ule";
    break;
  case VPInstruction::SLPLoad:
    O << "combined load";
    break;
  case VPInstruction::SLPStore:
    O << "combined store";
    break;
  case VPInstruction::ActiveLaneMask:
    O << "active lane mask";
    break;
  case VPInstruction::FirstOrderRecurrenceSplice:
    O << "first-order splice";
    break;
  case VPInstruction::CanonicalIVIncrement:
    O << "VF * UF + ";
    break;
  case VPInstruction::CanonicalIVIncrementNUW:
    O << "VF * UF +(nuw) ";
    break;
  case VPInstruction::BranchOnCond:
    O << "branch-on-cond";
    break;
  case VPInstruction::BranchOnCount:
    O << "branch-on-count ";
    break;
  default:
    O << Instruction::getOpcodeName(getOpcode());
  }

  O << FMF;

  for (const VPValue *Operand : operands()) {
    O << " ";
    Operand->printAsOperand(O, SlotTracker);
  }

  if (DL) {
    O << ", !dbg ";
    DL.print(O);
  }
}
#endif

void VPInstruction::setFastMathFlags(FastMathFlags FMFNew) {
  // Make sure the VPInstruction is a floating-point operation.
  assert((Opcode == Instruction::FAdd || Opcode == Instruction::FMul ||
          Opcode == Instruction::FNeg || Opcode == Instruction::FSub ||
          Opcode == Instruction::FDiv || Opcode == Instruction::FRem ||
          Opcode == Instruction::FCmp) &&
         "this op can't take fast-math flags");
  FMF = FMFNew;
}

void VPlan::prepareToExecute(Value *TripCountV, Value *VectorTripCountV,
                             Value *CanonicalIVStartValue,
                             VPTransformState &State) {

  VPBasicBlock *ExitingVPBB = getVectorLoopRegion()->getExitingBasicBlock();
  auto *Term = dyn_cast<VPInstruction>(&ExitingVPBB->back());
  // Try to simplify BranchOnCount to 'BranchOnCond true' if TC <= VF * UF when
  // preparing to execute the plan for the main vector loop.
  if (!CanonicalIVStartValue && Term &&
      Term->getOpcode() == VPInstruction::BranchOnCount &&
      isa<ConstantInt>(TripCountV)) {
    ConstantInt *C = cast<ConstantInt>(TripCountV);
    uint64_t TCVal = C->getZExtValue();
    if (TCVal && TCVal <= State.VF.getKnownMinValue() * State.UF) {
      auto *BOC =
          new VPInstruction(VPInstruction::BranchOnCond,
                            {getOrAddExternalDef(State.Builder.getTrue())});
      Term->eraseFromParent();
      ExitingVPBB->appendRecipe(BOC);
      // TODO: Further simplifications are possible
      //      1. Replace inductions with constants.
      //      2. Replace vector loop region with VPBasicBlock.
    }
  }

  // Check if the trip count is needed, and if so build it.
  if (TripCount && TripCount->getNumUsers()) {
    for (unsigned Part = 0, UF = State.UF; Part < UF; ++Part)
      State.set(TripCount, TripCountV, Part);
  }

  // Check if the backedge taken count is needed, and if so build it.
  if (BackedgeTakenCount && BackedgeTakenCount->getNumUsers()) {
    IRBuilder<> Builder(State.CFG.PrevBB->getTerminator());
    auto *TCMO = Builder.CreateSub(TripCountV,
                                   ConstantInt::get(TripCountV->getType(), 1),
                                   "trip.count.minus.1");
    auto VF = State.VF;
    Value *VTCMO =
        VF.isScalar() ? TCMO : Builder.CreateVectorSplat(VF, TCMO, "broadcast");
    for (unsigned Part = 0, UF = State.UF; Part < UF; ++Part)
      State.set(BackedgeTakenCount, VTCMO, Part);
  }

  for (unsigned Part = 0, UF = State.UF; Part < UF; ++Part)
    State.set(&VectorTripCount, VectorTripCountV, Part);

  // When vectorizing the epilogue loop, the canonical induction start value
  // needs to be changed from zero to the value after the main vector loop.
  if (CanonicalIVStartValue) {
    VPValue *VPV = getOrAddExternalDef(CanonicalIVStartValue);
    auto *IV = getCanonicalIV();
    assert(all_of(IV->users(),
                  [](const VPUser *U) {
                    if (isa<VPScalarIVStepsRecipe>(U))
                      return true;
                    auto *VPI = cast<VPInstruction>(U);
                    return VPI->getOpcode() ==
                               VPInstruction::CanonicalIVIncrement ||
                           VPI->getOpcode() ==
                               VPInstruction::CanonicalIVIncrementNUW;
                  }) &&
           "the canonical IV should only be used by its increments or "
           "ScalarIVSteps when "
           "resetting the start value");
    IV->setOperand(0, VPV);
  }
}

/// Generate the code inside the preheader and body of the vectorized loop.
/// Assumes a single pre-header basic-block was created for this. Introduce
/// additional basic-blocks as needed, and fill them all.
void VPlan::execute(VPTransformState *State) {
  // Set the reverse mapping from VPValues to Values for code generation.
  for (auto &Entry : Value2VPValue)
    State->VPValue2Value[Entry.second] = Entry.first;

  // Initialize CFG state.
  State->CFG.PrevVPBB = nullptr;
  State->CFG.ExitBB = State->CFG.PrevBB->getSingleSuccessor();
  BasicBlock *VectorPreHeader = State->CFG.PrevBB;
  State->Builder.SetInsertPoint(VectorPreHeader->getTerminator());

  // Generate code in the loop pre-header and body.
  for (VPBlockBase *Block : depth_first(Entry))
    Block->execute(State);

  VPBasicBlock *LatchVPBB = getVectorLoopRegion()->getExitingBasicBlock();
  BasicBlock *VectorLatchBB = State->CFG.VPBB2IRBB[LatchVPBB];

  // Fix the latch value of canonical, reduction and first-order recurrences
  // phis in the vector loop.
  VPBasicBlock *Header = getVectorLoopRegion()->getEntryBasicBlock();
  for (VPRecipeBase &R : Header->phis()) {
    // Skip phi-like recipes that generate their backedege values themselves.
    if (isa<VPWidenPHIRecipe>(&R))
      continue;

    if (isa<VPWidenPointerInductionRecipe>(&R) ||
        isa<VPWidenIntOrFpInductionRecipe>(&R)) {
      PHINode *Phi = nullptr;
      if (isa<VPWidenIntOrFpInductionRecipe>(&R)) {
        Phi = cast<PHINode>(State->get(R.getVPSingleValue(), 0));
      } else {
        auto *WidenPhi = cast<VPWidenPointerInductionRecipe>(&R);
        // TODO: Split off the case that all users of a pointer phi are scalar
        // from the VPWidenPointerInductionRecipe.
        if (WidenPhi->onlyScalarsGenerated(State->VF))
          continue;

        auto *GEP = cast<GetElementPtrInst>(State->get(WidenPhi, 0));
        Phi = cast<PHINode>(GEP->getPointerOperand());
      }

      Phi->setIncomingBlock(1, VectorLatchBB);

      // Move the last step to the end of the latch block. This ensures
      // consistent placement of all induction updates.
      Instruction *Inc = cast<Instruction>(Phi->getIncomingValue(1));
      Inc->moveBefore(VectorLatchBB->getTerminator()->getPrevNode());
      continue;
    }

    auto *PhiR = cast<VPHeaderPHIRecipe>(&R);
    // For  canonical IV, first-order recurrences and in-order reduction phis,
    // only a single part is generated, which provides the last part from the
    // previous iteration. For non-ordered reductions all UF parts are
    // generated.
    bool SinglePartNeeded = isa<VPCanonicalIVPHIRecipe>(PhiR) ||
                            isa<VPFirstOrderRecurrencePHIRecipe>(PhiR) ||
                            cast<VPReductionPHIRecipe>(PhiR)->isOrdered();
    unsigned LastPartForNewPhi = SinglePartNeeded ? 1 : State->UF;

    for (unsigned Part = 0; Part < LastPartForNewPhi; ++Part) {
      Value *Phi = State->get(PhiR, Part);
      Value *Val = State->get(PhiR->getBackedgeValue(),
                              SinglePartNeeded ? State->UF - 1 : Part);
      cast<PHINode>(Phi)->addIncoming(Val, VectorLatchBB);
    }
  }

  // We do not attempt to preserve DT for outer loop vectorization currently.
  if (!EnableVPlanNativePath) {
    BasicBlock *VectorHeaderBB = State->CFG.VPBB2IRBB[Header];
    State->DT->addNewBlock(VectorHeaderBB, VectorPreHeader);
    updateDominatorTree(State->DT, VectorHeaderBB, VectorLatchBB,
                        State->CFG.ExitBB);
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD
void VPlan::print(raw_ostream &O) const {
  VPSlotTracker SlotTracker(this);

  O << "VPlan '" << Name << "' {";

  if (VectorTripCount.getNumUsers() > 0) {
    O << "\nLive-in ";
    VectorTripCount.printAsOperand(O, SlotTracker);
    O << " = vector-trip-count\n";
  }

  if (BackedgeTakenCount && BackedgeTakenCount->getNumUsers()) {
    O << "\nLive-in ";
    BackedgeTakenCount->printAsOperand(O, SlotTracker);
    O << " = backedge-taken count\n";
  }

  for (const VPBlockBase *Block : depth_first(getEntry())) {
    O << '\n';
    Block->print(O, "", SlotTracker);
  }

  if (!LiveOuts.empty())
    O << "\n";
  for (auto &KV : LiveOuts) {
    O << "Live-out ";
    KV.second->getPhi()->printAsOperand(O);
    O << " = ";
    KV.second->getOperand(0)->printAsOperand(O, SlotTracker);
    O << "\n";
  }

  O << "}\n";
}

LLVM_DUMP_METHOD
void VPlan::printDOT(raw_ostream &O) const {
  VPlanPrinter Printer(O, *this);
  Printer.dump();
}

LLVM_DUMP_METHOD
void VPlan::dump() const { print(dbgs()); }
#endif

void VPlan::addLiveOut(PHINode *PN, VPValue *V) {
  assert(LiveOuts.count(PN) == 0 && "an exit value for PN already exists");
  LiveOuts.insert({PN, new VPLiveOut(PN, V)});
}

void VPlan::updateDominatorTree(DominatorTree *DT, BasicBlock *LoopHeaderBB,
                                BasicBlock *LoopLatchBB,
                                BasicBlock *LoopExitBB) {
  // The vector body may be more than a single basic-block by this point.
  // Update the dominator tree information inside the vector body by propagating
  // it from header to latch, expecting only triangular control-flow, if any.
  BasicBlock *PostDomSucc = nullptr;
  for (auto *BB = LoopHeaderBB; BB != LoopLatchBB; BB = PostDomSucc) {
    // Get the list of successors of this block.
    std::vector<BasicBlock *> Succs(succ_begin(BB), succ_end(BB));
    assert(Succs.size() <= 2 &&
           "Basic block in vector loop has more than 2 successors.");
    PostDomSucc = Succs[0];
    if (Succs.size() == 1) {
      assert(PostDomSucc->getSinglePredecessor() &&
             "PostDom successor has more than one predecessor.");
      DT->addNewBlock(PostDomSucc, BB);
      continue;
    }
    BasicBlock *InterimSucc = Succs[1];
    if (PostDomSucc->getSingleSuccessor() == InterimSucc) {
      PostDomSucc = Succs[1];
      InterimSucc = Succs[0];
    }
    assert(InterimSucc->getSingleSuccessor() == PostDomSucc &&
           "One successor of a basic block does not lead to the other.");
    assert(InterimSucc->getSinglePredecessor() &&
           "Interim successor has more than one predecessor.");
    assert(PostDomSucc->hasNPredecessors(2) &&
           "PostDom successor has more than two predecessors.");
    DT->addNewBlock(InterimSucc, BB);
    DT->addNewBlock(PostDomSucc, BB);
  }
  // Latch block is a new dominator for the loop exit.
  DT->changeImmediateDominator(LoopExitBB, LoopLatchBB);
  assert(DT->verify(DominatorTree::VerificationLevel::Fast));
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
Twine VPlanPrinter::getUID(const VPBlockBase *Block) {
  return (isa<VPRegionBlock>(Block) ? "cluster_N" : "N") +
         Twine(getOrCreateBID(Block));
}

Twine VPlanPrinter::getOrCreateName(const VPBlockBase *Block) {
  const std::string &Name = Block->getName();
  if (!Name.empty())
    return Name;
  return "VPB" + Twine(getOrCreateBID(Block));
}

void VPlanPrinter::dump() {
  Depth = 1;
  bumpIndent(0);
  OS << "digraph VPlan {\n";
  OS << "graph [labelloc=t, fontsize=30; label=\"Vectorization Plan";
  if (!Plan.getName().empty())
    OS << "\\n" << DOT::EscapeString(Plan.getName());
  if (Plan.BackedgeTakenCount) {
    OS << ", where:\\n";
    Plan.BackedgeTakenCount->print(OS, SlotTracker);
    OS << " := BackedgeTakenCount";
  }
  OS << "\"]\n";
  OS << "node [shape=rect, fontname=Courier, fontsize=30]\n";
  OS << "edge [fontname=Courier, fontsize=30]\n";
  OS << "compound=true\n";

  for (const VPBlockBase *Block : depth_first(Plan.getEntry()))
    dumpBlock(Block);

  OS << "}\n";
}

void VPlanPrinter::dumpBlock(const VPBlockBase *Block) {
  if (const VPBasicBlock *BasicBlock = dyn_cast<VPBasicBlock>(Block))
    dumpBasicBlock(BasicBlock);
  else if (const VPRegionBlock *Region = dyn_cast<VPRegionBlock>(Block))
    dumpRegion(Region);
  else
    llvm_unreachable("Unsupported kind of VPBlock.");
}

void VPlanPrinter::drawEdge(const VPBlockBase *From, const VPBlockBase *To,
                            bool Hidden, const Twine &Label) {
  // Due to "dot" we print an edge between two regions as an edge between the
  // exiting basic block and the entry basic of the respective regions.
  const VPBlockBase *Tail = From->getExitingBasicBlock();
  const VPBlockBase *Head = To->getEntryBasicBlock();
  OS << Indent << getUID(Tail) << " -> " << getUID(Head);
  OS << " [ label=\"" << Label << '\"';
  if (Tail != From)
    OS << " ltail=" << getUID(From);
  if (Head != To)
    OS << " lhead=" << getUID(To);
  if (Hidden)
    OS << "; splines=none";
  OS << "]\n";
}

void VPlanPrinter::dumpEdges(const VPBlockBase *Block) {
  auto &Successors = Block->getSuccessors();
  if (Successors.size() == 1)
    drawEdge(Block, Successors.front(), false, "");
  else if (Successors.size() == 2) {
    drawEdge(Block, Successors.front(), false, "T");
    drawEdge(Block, Successors.back(), false, "F");
  } else {
    unsigned SuccessorNumber = 0;
    for (auto *Successor : Successors)
      drawEdge(Block, Successor, false, Twine(SuccessorNumber++));
  }
}

void VPlanPrinter::dumpBasicBlock(const VPBasicBlock *BasicBlock) {
  // Implement dot-formatted dump by performing plain-text dump into the
  // temporary storage followed by some post-processing.
  OS << Indent << getUID(BasicBlock) << " [label =\n";
  bumpIndent(1);
  std::string Str;
  raw_string_ostream SS(Str);
  // Use no indentation as we need to wrap the lines into quotes ourselves.
  BasicBlock->print(SS, "", SlotTracker);

  // We need to process each line of the output separately, so split
  // single-string plain-text dump.
  SmallVector<StringRef, 0> Lines;
  StringRef(Str).rtrim('\n').split(Lines, "\n");

  auto EmitLine = [&](StringRef Line, StringRef Suffix) {
    OS << Indent << '"' << DOT::EscapeString(Line.str()) << "\\l\"" << Suffix;
  };

  // Don't need the "+" after the last line.
  for (auto Line : make_range(Lines.begin(), Lines.end() - 1))
    EmitLine(Line, " +\n");
  EmitLine(Lines.back(), "\n");

  bumpIndent(-1);
  OS << Indent << "]\n";

  dumpEdges(BasicBlock);
}

void VPlanPrinter::dumpRegion(const VPRegionBlock *Region) {
  OS << Indent << "subgraph " << getUID(Region) << " {\n";
  bumpIndent(1);
  OS << Indent << "fontname=Courier\n"
     << Indent << "label=\""
     << DOT::EscapeString(Region->isReplicator() ? "<xVFxUF> " : "<x1> ")
     << DOT::EscapeString(Region->getName()) << "\"\n";
  // Dump the blocks of the region.
  assert(Region->getEntry() && "Region contains no inner blocks.");
  for (const VPBlockBase *Block : depth_first(Region->getEntry()))
    dumpBlock(Block);
  bumpIndent(-1);
  OS << Indent << "}\n";
  dumpEdges(Region);
}

void VPlanIngredient::print(raw_ostream &O) const {
  if (auto *Inst = dyn_cast<Instruction>(V)) {
    if (!Inst->getType()->isVoidTy()) {
      Inst->printAsOperand(O, false);
      O << " = ";
    }
    O << Inst->getOpcodeName() << " ";
    unsigned E = Inst->getNumOperands();
    if (E > 0) {
      Inst->getOperand(0)->printAsOperand(O, false);
      for (unsigned I = 1; I < E; ++I)
        Inst->getOperand(I)->printAsOperand(O << ", ", false);
    }
  } else // !Inst
    V->printAsOperand(O, false);
}

void VPWidenCallRecipe::print(raw_ostream &O, const Twine &Indent,
                              VPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN-CALL ";

  auto *CI = cast<CallInst>(getUnderlyingInstr());
  if (CI->getType()->isVoidTy())
    O << "void ";
  else {
    printAsOperand(O, SlotTracker);
    O << " = ";
  }

  O << "call @" << CI->getCalledFunction()->getName() << "(";
  printOperands(O, SlotTracker);
  O << ")";
}

void VPWidenSelectRecipe::print(raw_ostream &O, const Twine &Indent,
                                VPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN-SELECT ";
  printAsOperand(O, SlotTracker);
  O << " = select ";
  getOperand(0)->printAsOperand(O, SlotTracker);
  O << ", ";
  getOperand(1)->printAsOperand(O, SlotTracker);
  O << ", ";
  getOperand(2)->printAsOperand(O, SlotTracker);
  O << (InvariantCond ? " (condition is loop invariant)" : "");
}

void VPWidenRecipe::print(raw_ostream &O, const Twine &Indent,
                          VPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN ";
  printAsOperand(O, SlotTracker);
  O << " = " << getUnderlyingInstr()->getOpcodeName() << " ";
  printOperands(O, SlotTracker);
}

void VPWidenIntOrFpInductionRecipe::print(raw_ostream &O, const Twine &Indent,
                                          VPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN-INDUCTION";
  if (getTruncInst()) {
    O << "\\l\"";
    O << " +\n" << Indent << "\"  " << VPlanIngredient(IV) << "\\l\"";
    O << " +\n" << Indent << "\"  ";
    getVPValue(0)->printAsOperand(O, SlotTracker);
  } else
    O << " " << VPlanIngredient(IV);

  O << ", ";
  getStepValue()->printAsOperand(O, SlotTracker);
}

void VPWidenPointerInductionRecipe::print(raw_ostream &O, const Twine &Indent,
                                          VPSlotTracker &SlotTracker) const {
  O << Indent << "EMIT ";
  printAsOperand(O, SlotTracker);
  O << " = WIDEN-POINTER-INDUCTION ";
  getStartValue()->printAsOperand(O, SlotTracker);
  O << ", " << *IndDesc.getStep();
}

#endif

bool VPWidenIntOrFpInductionRecipe::isCanonical() const {
  auto *StartC = dyn_cast<ConstantInt>(getStartValue()->getLiveInIRValue());
  auto *StepC = dyn_cast<SCEVConstant>(getInductionDescriptor().getStep());
  return StartC && StartC->isZero() && StepC && StepC->isOne();
}

VPCanonicalIVPHIRecipe *VPScalarIVStepsRecipe::getCanonicalIV() const {
  return cast<VPCanonicalIVPHIRecipe>(getOperand(0));
}

bool VPScalarIVStepsRecipe::isCanonical() const {
  auto *CanIV = getCanonicalIV();
  // The start value of the steps-recipe must match the start value of the
  // canonical induction and it must step by 1.
  if (CanIV->getStartValue() != getStartValue())
    return false;
  auto *StepVPV = getStepValue();
  if (StepVPV->getDef())
    return false;
  auto *StepC = dyn_cast_or_null<ConstantInt>(StepVPV->getLiveInIRValue());
  return StepC && StepC->isOne();
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPScalarIVStepsRecipe::print(raw_ostream &O, const Twine &Indent,
                                  VPSlotTracker &SlotTracker) const {
  O << Indent;
  printAsOperand(O, SlotTracker);
  O << Indent << "= SCALAR-STEPS ";
  printOperands(O, SlotTracker);
}

void VPWidenGEPRecipe::print(raw_ostream &O, const Twine &Indent,
                             VPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN-GEP ";
  O << (IsPtrLoopInvariant ? "Inv" : "Var");
  size_t IndicesNumber = IsIndexLoopInvariant.size();
  for (size_t I = 0; I < IndicesNumber; ++I)
    O << "[" << (IsIndexLoopInvariant[I] ? "Inv" : "Var") << "]";

  O << " ";
  printAsOperand(O, SlotTracker);
  O << " = getelementptr ";
  printOperands(O, SlotTracker);
}

void VPWidenPHIRecipe::print(raw_ostream &O, const Twine &Indent,
                             VPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN-PHI ";

  auto *OriginalPhi = cast<PHINode>(getUnderlyingValue());
  // Unless all incoming values are modeled in VPlan  print the original PHI
  // directly.
  // TODO: Remove once all VPWidenPHIRecipe instances keep all relevant incoming
  // values as VPValues.
  if (getNumOperands() != OriginalPhi->getNumOperands()) {
    O << VPlanIngredient(OriginalPhi);
    return;
  }

  printAsOperand(O, SlotTracker);
  O << " = phi ";
  printOperands(O, SlotTracker);
}

void VPBlendRecipe::print(raw_ostream &O, const Twine &Indent,
                          VPSlotTracker &SlotTracker) const {
  O << Indent << "BLEND ";
  Phi->printAsOperand(O, false);
  O << " =";
  if (getNumIncomingValues() == 1) {
    // Not a User of any mask: not really blending, this is a
    // single-predecessor phi.
    O << " ";
    getIncomingValue(0)->printAsOperand(O, SlotTracker);
  } else {
    for (unsigned I = 0, E = getNumIncomingValues(); I < E; ++I) {
      O << " ";
      getIncomingValue(I)->printAsOperand(O, SlotTracker);
      O << "/";
      getMask(I)->printAsOperand(O, SlotTracker);
    }
  }
}

void VPReductionRecipe::print(raw_ostream &O, const Twine &Indent,
                              VPSlotTracker &SlotTracker) const {
  O << Indent << "REDUCE ";
  printAsOperand(O, SlotTracker);
  O << " = ";
  getChainOp()->printAsOperand(O, SlotTracker);
  O << " +";
  if (isa<FPMathOperator>(getUnderlyingInstr()))
    O << getUnderlyingInstr()->getFastMathFlags();
  O << " reduce." << Instruction::getOpcodeName(RdxDesc->getOpcode()) << " (";
  getVecOp()->printAsOperand(O, SlotTracker);
  if (getCondOp()) {
    O << ", ";
    getCondOp()->printAsOperand(O, SlotTracker);
  }
  O << ")";
  if (RdxDesc->IntermediateStore)
    O << " (with final reduction value stored in invariant address sank "
         "outside of loop)";
}

void VPReplicateRecipe::print(raw_ostream &O, const Twine &Indent,
                              VPSlotTracker &SlotTracker) const {
  O << Indent << (IsUniform ? "CLONE " : "REPLICATE ");

  if (!getUnderlyingInstr()->getType()->isVoidTy()) {
    printAsOperand(O, SlotTracker);
    O << " = ";
  }
  if (auto *CB = dyn_cast<CallBase>(getUnderlyingInstr())) {
    O << "call @" << CB->getCalledFunction()->getName() << "(";
    interleaveComma(make_range(op_begin(), op_begin() + (getNumOperands() - 1)),
                    O, [&O, &SlotTracker](VPValue *Op) {
                      Op->printAsOperand(O, SlotTracker);
                    });
    O << ")";
  } else {
    O << Instruction::getOpcodeName(getUnderlyingInstr()->getOpcode()) << " ";
    printOperands(O, SlotTracker);
  }

  if (AlsoPack)
    O << " (S->V)";
}

void VPPredInstPHIRecipe::print(raw_ostream &O, const Twine &Indent,
                                VPSlotTracker &SlotTracker) const {
  O << Indent << "PHI-PREDICATED-INSTRUCTION ";
  printAsOperand(O, SlotTracker);
  O << " = ";
  printOperands(O, SlotTracker);
}

void VPWidenMemoryInstructionRecipe::print(raw_ostream &O, const Twine &Indent,
                                           VPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN ";

  if (!isStore()) {
    getVPSingleValue()->printAsOperand(O, SlotTracker);
    O << " = ";
  }
  O << Instruction::getOpcodeName(Ingredient.getOpcode()) << " ";

  printOperands(O, SlotTracker);
}
#endif

void VPCanonicalIVPHIRecipe::execute(VPTransformState &State) {
  Value *Start = getStartValue()->getLiveInIRValue();
  PHINode *EntryPart = PHINode::Create(
      Start->getType(), 2, "index", &*State.CFG.PrevBB->getFirstInsertionPt());

  BasicBlock *VectorPH = State.CFG.getPreheaderBBFor(this);
  EntryPart->addIncoming(Start, VectorPH);
  EntryPart->setDebugLoc(DL);
  for (unsigned Part = 0, UF = State.UF; Part < UF; ++Part)
    State.set(this, EntryPart, Part);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPCanonicalIVPHIRecipe::print(raw_ostream &O, const Twine &Indent,
                                   VPSlotTracker &SlotTracker) const {
  O << Indent << "EMIT ";
  printAsOperand(O, SlotTracker);
  O << " = CANONICAL-INDUCTION";
}
#endif

bool VPWidenPointerInductionRecipe::onlyScalarsGenerated(ElementCount VF) {
  bool IsUniform = vputils::onlyFirstLaneUsed(this);
  return all_of(users(),
                [&](const VPUser *U) { return U->usesScalars(this); }) &&
         (IsUniform || !VF.isScalable());
}

void VPExpandSCEVRecipe::execute(VPTransformState &State) {
  assert(!State.Instance && "cannot be used in per-lane");
  const DataLayout &DL = State.CFG.PrevBB->getModule()->getDataLayout();
  SCEVExpander Exp(SE, DL, "induction");

  Value *Res = Exp.expandCodeFor(Expr, Expr->getType(),
                                 &*State.Builder.GetInsertPoint());

  for (unsigned Part = 0, UF = State.UF; Part < UF; ++Part)
    State.set(this, Res, Part);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPExpandSCEVRecipe::print(raw_ostream &O, const Twine &Indent,
                               VPSlotTracker &SlotTracker) const {
  O << Indent << "EMIT ";
  getVPSingleValue()->printAsOperand(O, SlotTracker);
  O << " = EXPAND SCEV " << *Expr;
}
#endif

void VPWidenCanonicalIVRecipe::execute(VPTransformState &State) {
  Value *CanonicalIV = State.get(getOperand(0), 0);
  Type *STy = CanonicalIV->getType();
  IRBuilder<> Builder(State.CFG.PrevBB->getTerminator());
  ElementCount VF = State.VF;
  Value *VStart = VF.isScalar()
                      ? CanonicalIV
                      : Builder.CreateVectorSplat(VF, CanonicalIV, "broadcast");
  for (unsigned Part = 0, UF = State.UF; Part < UF; ++Part) {
    Value *VStep = createStepForVF(Builder, STy, VF, Part);
    if (VF.isVector()) {
      VStep = Builder.CreateVectorSplat(VF, VStep);
      VStep = Builder.CreateAdd(VStep, Builder.CreateStepVector(VStep->getType()));
    }
    Value *CanonicalVectorIV = Builder.CreateAdd(VStart, VStep, "vec.iv");
    State.set(this, CanonicalVectorIV, Part);
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPWidenCanonicalIVRecipe::print(raw_ostream &O, const Twine &Indent,
                                     VPSlotTracker &SlotTracker) const {
  O << Indent << "EMIT ";
  printAsOperand(O, SlotTracker);
  O << " = WIDEN-CANONICAL-INDUCTION ";
  printOperands(O, SlotTracker);
}
#endif

void VPFirstOrderRecurrencePHIRecipe::execute(VPTransformState &State) {
  auto &Builder = State.Builder;
  // Create a vector from the initial value.
  auto *VectorInit = getStartValue()->getLiveInIRValue();

  Type *VecTy = State.VF.isScalar()
                    ? VectorInit->getType()
                    : VectorType::get(VectorInit->getType(), State.VF);

  BasicBlock *VectorPH = State.CFG.getPreheaderBBFor(this);
  if (State.VF.isVector()) {
    auto *IdxTy = Builder.getInt32Ty();
    auto *One = ConstantInt::get(IdxTy, 1);
    IRBuilder<>::InsertPointGuard Guard(Builder);
    Builder.SetInsertPoint(VectorPH->getTerminator());
    auto *RuntimeVF = getRuntimeVF(Builder, IdxTy, State.VF);
    auto *LastIdx = Builder.CreateSub(RuntimeVF, One);
    VectorInit = Builder.CreateInsertElement(
        PoisonValue::get(VecTy), VectorInit, LastIdx, "vector.recur.init");
  }

  // Create a phi node for the new recurrence.
  PHINode *EntryPart = PHINode::Create(
      VecTy, 2, "vector.recur", &*State.CFG.PrevBB->getFirstInsertionPt());
  EntryPart->addIncoming(VectorInit, VectorPH);
  State.set(this, EntryPart, 0);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPFirstOrderRecurrencePHIRecipe::print(raw_ostream &O, const Twine &Indent,
                                            VPSlotTracker &SlotTracker) const {
  O << Indent << "FIRST-ORDER-RECURRENCE-PHI ";
  printAsOperand(O, SlotTracker);
  O << " = phi ";
  printOperands(O, SlotTracker);
}
#endif

void VPReductionPHIRecipe::execute(VPTransformState &State) {
  PHINode *PN = cast<PHINode>(getUnderlyingValue());
  auto &Builder = State.Builder;

  // In order to support recurrences we need to be able to vectorize Phi nodes.
  // Phi nodes have cycles, so we need to vectorize them in two stages. This is
  // stage #1: We create a new vector PHI node with no incoming edges. We'll use
  // this value when we vectorize all of the instructions that use the PHI.
  bool ScalarPHI = State.VF.isScalar() || IsInLoop;
  Type *VecTy =
      ScalarPHI ? PN->getType() : VectorType::get(PN->getType(), State.VF);

  BasicBlock *HeaderBB = State.CFG.PrevBB;
  assert(State.CurrentVectorLoop->getHeader() == HeaderBB &&
         "recipe must be in the vector loop header");
  unsigned LastPartForNewPhi = isOrdered() ? 1 : State.UF;
  for (unsigned Part = 0; Part < LastPartForNewPhi; ++Part) {
    Value *EntryPart =
        PHINode::Create(VecTy, 2, "vec.phi", &*HeaderBB->getFirstInsertionPt());
    State.set(this, EntryPart, Part);
  }

  BasicBlock *VectorPH = State.CFG.getPreheaderBBFor(this);

  // Reductions do not have to start at zero. They can start with
  // any loop invariant values.
  VPValue *StartVPV = getStartValue();
  Value *StartV = StartVPV->getLiveInIRValue();

  Value *Iden = nullptr;
  RecurKind RK = RdxDesc.getRecurrenceKind();
  if (RecurrenceDescriptor::isMinMaxRecurrenceKind(RK) ||
      RecurrenceDescriptor::isSelectCmpRecurrenceKind(RK)) {
    // MinMax reduction have the start value as their identify.
    if (ScalarPHI) {
      Iden = StartV;
    } else {
      IRBuilderBase::InsertPointGuard IPBuilder(Builder);
      Builder.SetInsertPoint(VectorPH->getTerminator());
      StartV = Iden =
          Builder.CreateVectorSplat(State.VF, StartV, "minmax.ident");
    }
  } else {
    Iden = RdxDesc.getRecurrenceIdentity(RK, VecTy->getScalarType(),
                                         RdxDesc.getFastMathFlags());

    if (!ScalarPHI) {
      Iden = Builder.CreateVectorSplat(State.VF, Iden);
      IRBuilderBase::InsertPointGuard IPBuilder(Builder);
      Builder.SetInsertPoint(VectorPH->getTerminator());
      Constant *Zero = Builder.getInt32(0);
      StartV = Builder.CreateInsertElement(Iden, StartV, Zero);
    }
  }

  for (unsigned Part = 0; Part < LastPartForNewPhi; ++Part) {
    Value *EntryPart = State.get(this, Part);
    // Make sure to add the reduction start value only to the
    // first unroll part.
    Value *StartVal = (Part == 0) ? StartV : Iden;
    cast<PHINode>(EntryPart)->addIncoming(StartVal, VectorPH);
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPReductionPHIRecipe::print(raw_ostream &O, const Twine &Indent,
                                 VPSlotTracker &SlotTracker) const {
  O << Indent << "WIDEN-REDUCTION-PHI ";

  printAsOperand(O, SlotTracker);
  O << " = phi ";
  printOperands(O, SlotTracker);
}
#endif

void VPWidenPHIRecipe::execute(VPTransformState &State) {
  assert(EnableVPlanNativePath &&
         "Non-native vplans are not expected to have VPWidenPHIRecipes.");

  // Currently we enter here in the VPlan-native path for non-induction
  // PHIs where all control flow is uniform. We simply widen these PHIs.
  // Create a vector phi with no operands - the vector phi operands will be
  // set at the end of vector code generation.
  VPBasicBlock *Parent = getParent();
  VPRegionBlock *LoopRegion = Parent->getEnclosingLoopRegion();
  unsigned StartIdx = 0;
  // For phis in header blocks of loop regions, use the index of the value
  // coming from the preheader.
  if (LoopRegion->getEntryBasicBlock() == Parent) {
    for (unsigned I = 0; I < getNumOperands(); ++I) {
      if (getIncomingBlock(I) ==
          LoopRegion->getSinglePredecessor()->getExitingBasicBlock())
        StartIdx = I;
    }
  }
  Value *Op0 = State.get(getOperand(StartIdx), 0);
  Type *VecTy = Op0->getType();
  Value *VecPhi = State.Builder.CreatePHI(VecTy, 2, "vec.phi");
  State.set(this, VecPhi, 0);
}

template void DomTreeBuilder::Calculate<VPDominatorTree>(VPDominatorTree &DT);

void VPValue::replaceAllUsesWith(VPValue *New) {
  for (unsigned J = 0; J < getNumUsers();) {
    VPUser *User = Users[J];
    unsigned NumUsers = getNumUsers();
    for (unsigned I = 0, E = User->getNumOperands(); I < E; ++I)
      if (User->getOperand(I) == this)
        User->setOperand(I, New);
    // If a user got removed after updating the current user, the next user to
    // update will be moved to the current position, so we only need to
    // increment the index if the number of users did not change.
    if (NumUsers == getNumUsers())
      J++;
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPValue::printAsOperand(raw_ostream &OS, VPSlotTracker &Tracker) const {
  if (const Value *UV = getUnderlyingValue()) {
    OS << "ir<";
    UV->printAsOperand(OS, false);
    OS << ">";
    return;
  }

  unsigned Slot = Tracker.getSlot(this);
  if (Slot == unsigned(-1))
    OS << "<badref>";
  else
    OS << "vp<%" << Tracker.getSlot(this) << ">";
}

void VPUser::printOperands(raw_ostream &O, VPSlotTracker &SlotTracker) const {
  interleaveComma(operands(), O, [&O, &SlotTracker](VPValue *Op) {
    Op->printAsOperand(O, SlotTracker);
  });
}
#endif

void VPInterleavedAccessInfo::visitRegion(VPRegionBlock *Region,
                                          Old2NewTy &Old2New,
                                          InterleavedAccessInfo &IAI) {
  ReversePostOrderTraversal<VPBlockBase *> RPOT(Region->getEntry());
  for (VPBlockBase *Base : RPOT) {
    visitBlock(Base, Old2New, IAI);
  }
}

void VPInterleavedAccessInfo::visitBlock(VPBlockBase *Block, Old2NewTy &Old2New,
                                         InterleavedAccessInfo &IAI) {
  if (VPBasicBlock *VPBB = dyn_cast<VPBasicBlock>(Block)) {
    for (VPRecipeBase &VPI : *VPBB) {
      if (isa<VPHeaderPHIRecipe>(&VPI))
        continue;
      assert(isa<VPInstruction>(&VPI) && "Can only handle VPInstructions");
      auto *VPInst = cast<VPInstruction>(&VPI);

      auto *Inst = dyn_cast_or_null<Instruction>(VPInst->getUnderlyingValue());
      if (!Inst)
        continue;
      auto *IG = IAI.getInterleaveGroup(Inst);
      if (!IG)
        continue;

      auto NewIGIter = Old2New.find(IG);
      if (NewIGIter == Old2New.end())
        Old2New[IG] = new InterleaveGroup<VPInstruction>(
            IG->getFactor(), IG->isReverse(), IG->getAlign());

      if (Inst == IG->getInsertPos())
        Old2New[IG]->setInsertPos(VPInst);

      InterleaveGroupMap[VPInst] = Old2New[IG];
      InterleaveGroupMap[VPInst]->insertMember(
          VPInst, IG->getIndex(Inst),
          Align(IG->isReverse() ? (-1) * int(IG->getFactor())
                                : IG->getFactor()));
    }
  } else if (VPRegionBlock *Region = dyn_cast<VPRegionBlock>(Block))
    visitRegion(Region, Old2New, IAI);
  else
    llvm_unreachable("Unsupported kind of VPBlock.");
}

VPInterleavedAccessInfo::VPInterleavedAccessInfo(VPlan &Plan,
                                                 InterleavedAccessInfo &IAI) {
  Old2NewTy Old2New;
  visitRegion(Plan.getVectorLoopRegion(), Old2New, IAI);
}

void VPSlotTracker::assignSlot(const VPValue *V) {
  assert(Slots.find(V) == Slots.end() && "VPValue already has a slot!");
  Slots[V] = NextSlot++;
}

void VPSlotTracker::assignSlots(const VPlan &Plan) {

  for (const auto &P : Plan.VPExternalDefs)
    assignSlot(P.second);

  assignSlot(&Plan.VectorTripCount);
  if (Plan.BackedgeTakenCount)
    assignSlot(Plan.BackedgeTakenCount);

  ReversePostOrderTraversal<
      VPBlockRecursiveTraversalWrapper<const VPBlockBase *>>
      RPOT(VPBlockRecursiveTraversalWrapper<const VPBlockBase *>(
          Plan.getEntry()));
  for (const VPBasicBlock *VPBB :
       VPBlockUtils::blocksOnly<const VPBasicBlock>(RPOT))
    for (const VPRecipeBase &Recipe : *VPBB)
      for (VPValue *Def : Recipe.definedValues())
        assignSlot(Def);
}

bool vputils::onlyFirstLaneUsed(VPValue *Def) {
  return all_of(Def->users(),
                [Def](VPUser *U) { return U->onlyFirstLaneUsed(Def); });
}

VPValue *vputils::getOrCreateVPValueForSCEVExpr(VPlan &Plan, const SCEV *Expr,
                                                ScalarEvolution &SE) {
  if (auto *E = dyn_cast<SCEVConstant>(Expr))
    return Plan.getOrAddExternalDef(E->getValue());
  if (auto *E = dyn_cast<SCEVUnknown>(Expr))
    return Plan.getOrAddExternalDef(E->getValue());

  VPBasicBlock *Preheader = Plan.getEntry()->getEntryBasicBlock();
  VPValue *Step = new VPExpandSCEVRecipe(Expr, SE);
  Preheader->appendRecipe(cast<VPRecipeBase>(Step->getDef()));
  return Step;
}
