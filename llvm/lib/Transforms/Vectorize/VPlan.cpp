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
#include "LoopVectorizationPlanner.h"
#include "VPlanCFG.h"
#include "VPlanDominatorTree.h"
#include "VPlanHelpers.h"
#include "VPlanPatternMatch.h"
#include "VPlanTransforms.h"
#include "VPlanUtils.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/DomTreeUpdater.h"
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
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/LoopVersioning.h"
#include "llvm/Transforms/Vectorize/LoopVectorizationLegality.h"
#include <cassert>
#include <string>

using namespace llvm;
using namespace llvm::VPlanPatternMatch;

namespace llvm {
extern cl::opt<bool> EnableVPlanNativePath;
}

/// @{
/// Metadata attribute names
const char LLVMLoopVectorizeFollowupAll[] = "llvm.loop.vectorize.followup_all";
const char LLVMLoopVectorizeFollowupVectorized[] =
    "llvm.loop.vectorize.followup_vectorized";
const char LLVMLoopVectorizeFollowupEpilogue[] =
    "llvm.loop.vectorize.followup_epilogue";
/// @}

extern cl::opt<unsigned> ForceTargetInstructionCost;

static cl::opt<bool> PrintVPlansInDotFormat(
    "vplan-print-in-dot-format", cl::Hidden,
    cl::desc("Use dot format instead of plain text when dumping VPlans"));

#define DEBUG_TYPE "loop-vectorize"

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
raw_ostream &llvm::operator<<(raw_ostream &OS, const VPRecipeBase &R) {
  const VPBasicBlock *Parent = R.getParent();
  VPSlotTracker SlotTracker(Parent ? Parent->getPlan() : nullptr);
  R.print(OS, "", SlotTracker);
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

VPRecipeBase *VPValue::getDefiningRecipe() {
  return cast_or_null<VPRecipeBase>(Def);
}

const VPRecipeBase *VPValue::getDefiningRecipe() const {
  return cast_or_null<VPRecipeBase>(Def);
}

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
    if (!Current->hasPredecessors())
      return Current;
    auto &Predecessors = Current->getPredecessors();
    WorkList.insert_range(Predecessors);
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
  assert(ParentPlan->getEntry() == this && "Can only set plan on its entry.");
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

bool VPBlockUtils::isHeader(const VPBlockBase *VPB,
                            const VPDominatorTree &VPDT) {
  auto *VPBB = dyn_cast<VPBasicBlock>(VPB);
  if (!VPBB)
    return false;

  // If VPBB is in a region R, VPBB is a loop header if R is a loop region with
  // VPBB as its entry, i.e., free of predecessors.
  if (auto *R = VPBB->getParent())
    return !R->isReplicator() && !VPBB->hasPredecessors();

  // A header dominates its second predecessor (the latch), with the other
  // predecessor being the preheader
  return VPB->getPredecessors().size() == 2 &&
         VPDT.dominates(VPB, VPB->getPredecessors()[1]);
}

bool VPBlockUtils::isLatch(const VPBlockBase *VPB,
                           const VPDominatorTree &VPDT) {
  // A latch has a header as its second successor, with its other successor
  // leaving the loop. A preheader OTOH has a header as its first (and only)
  // successor.
  return VPB->getNumSuccessors() == 2 &&
         VPBlockUtils::isHeader(VPB->getSuccessors()[1], VPDT);
}

VPBasicBlock::iterator VPBasicBlock::getFirstNonPhi() {
  iterator It = begin();
  while (It != end() && It->isPhi())
    It++;
  return It;
}

VPTransformState::VPTransformState(const TargetTransformInfo *TTI,
                                   ElementCount VF, LoopInfo *LI,
                                   DominatorTree *DT, AssumptionCache *AC,
                                   IRBuilderBase &Builder, VPlan *Plan,
                                   Loop *CurrentParentLoop, Type *CanonicalIVTy)
    : TTI(TTI), VF(VF), CFG(DT), LI(LI), AC(AC), Builder(Builder), Plan(Plan),
      CurrentParentLoop(CurrentParentLoop), TypeAnalysis(*Plan), VPDT(*Plan) {}

Value *VPTransformState::get(const VPValue *Def, const VPLane &Lane) {
  if (Def->isLiveIn())
    return Def->getLiveInIRValue();

  if (hasScalarValue(Def, Lane))
    return Data.VPV2Scalars[Def][Lane.mapToCacheIndex(VF)];

  if (!Lane.isFirstLane() && vputils::isSingleScalar(Def) &&
      hasScalarValue(Def, VPLane::getFirstLane())) {
    return Data.VPV2Scalars[Def][0];
  }

  // Look through BuildVector to avoid redundant extracts.
  // TODO: Remove once replicate regions are unrolled explicitly.
  if (Lane.getKind() == VPLane::Kind::First && match(Def, m_BuildVector())) {
    auto *BuildVector = cast<VPInstruction>(Def);
    return get(BuildVector->getOperand(Lane.getKnownLane()), true);
  }

  assert(hasVectorValue(Def));
  auto *VecPart = Data.VPV2Vector[Def];
  if (!VecPart->getType()->isVectorTy()) {
    assert(Lane.isFirstLane() && "cannot get lane > 0 for scalar");
    return VecPart;
  }
  // TODO: Cache created scalar values.
  Value *LaneV = Lane.getAsRuntimeExpr(Builder, VF);
  auto *Extract = Builder.CreateExtractElement(VecPart, LaneV);
  // set(Def, Extract, Instance);
  return Extract;
}

Value *VPTransformState::get(const VPValue *Def, bool NeedsScalar) {
  if (NeedsScalar) {
    assert((VF.isScalar() || Def->isLiveIn() || hasVectorValue(Def) ||
            !vputils::onlyFirstLaneUsed(Def) ||
            (hasScalarValue(Def, VPLane(0)) &&
             Data.VPV2Scalars[Def].size() == 1)) &&
           "Trying to access a single scalar per part but has multiple scalars "
           "per part.");
    return get(Def, VPLane(0));
  }

  // If Values have been set for this Def return the one relevant for \p Part.
  if (hasVectorValue(Def))
    return Data.VPV2Vector[Def];

  auto GetBroadcastInstrs = [this](Value *V) {
    if (VF.isScalar())
      return V;
    // Broadcast the scalar into all locations in the vector.
    Value *Shuf = Builder.CreateVectorSplat(VF, V, "broadcast");
    return Shuf;
  };

  if (!hasScalarValue(Def, {0})) {
    assert(Def->isLiveIn() && "expected a live-in");
    Value *IRV = Def->getLiveInIRValue();
    Value *B = GetBroadcastInstrs(IRV);
    set(Def, B);
    return B;
  }

  Value *ScalarValue = get(Def, VPLane(0));
  // If we aren't vectorizing, we can just copy the scalar map values over
  // to the vector map.
  if (VF.isScalar()) {
    set(Def, ScalarValue);
    return ScalarValue;
  }

  bool IsSingleScalar = vputils::isSingleScalar(Def);

  VPLane LastLane(IsSingleScalar ? 0 : VF.getFixedValue() - 1);
  // Check if there is a scalar value for the selected lane.
  if (!hasScalarValue(Def, LastLane)) {
    // At the moment, VPWidenIntOrFpInductionRecipes, VPScalarIVStepsRecipes and
    // VPExpandSCEVRecipes can also be a single scalar.
    assert((isa<VPWidenIntOrFpInductionRecipe, VPScalarIVStepsRecipe,
                VPExpandSCEVRecipe>(Def->getDefiningRecipe())) &&
           "unexpected recipe found to be invariant");
    IsSingleScalar = true;
    LastLane = 0;
  }

  // We need to construct the vector value for a single-scalar value by
  // broadcasting the scalar to all lanes.
  // TODO: Replace by introducing Broadcast VPInstructions.
  assert(IsSingleScalar && "must be a single-scalar at this point");
  // Set the insert point after the last scalarized instruction or after the
  // last PHI, if LastInst is a PHI. This ensures the insertelement sequence
  // will directly follow the scalar definitions.
  auto OldIP = Builder.saveIP();
  auto *LastInst = cast<Instruction>(get(Def, LastLane));
  auto NewIP = isa<PHINode>(LastInst)
                   ? LastInst->getParent()->getFirstNonPHIIt()
                   : std::next(BasicBlock::iterator(LastInst));
  Builder.SetInsertPoint(&*NewIP);
  Value *VectorValue = GetBroadcastInstrs(ScalarValue);
  set(Def, VectorValue);
  Builder.restoreIP(OldIP);
  return VectorValue;
}

void VPTransformState::setDebugLocFrom(DebugLoc DL) {
  const DILocation *DIL = DL;
  // When a FSDiscriminator is enabled, we don't need to add the multiply
  // factors to the discriminators.
  if (DIL &&
      Builder.GetInsertBlock()
          ->getParent()
          ->shouldEmitDebugInfoForProfiling() &&
      !EnableFSDiscriminator) {
    // FIXME: For scalable vectors, assume vscale=1.
    unsigned UF = Plan->getUF();
    auto NewDIL =
        DIL->cloneByMultiplyingDuplicationFactor(UF * VF.getKnownMinValue());
    if (NewDIL)
      Builder.SetCurrentDebugLocation(*NewDIL);
    else
      LLVM_DEBUG(dbgs() << "Failed to create new discriminator: "
                        << DIL->getFilename() << " Line: " << DIL->getLine());
  } else
    Builder.SetCurrentDebugLocation(DL);
}

Value *VPTransformState::packScalarIntoVectorizedValue(const VPValue *Def,
                                                       Value *WideValue,
                                                       const VPLane &Lane) {
  Value *ScalarInst = get(Def, Lane);
  Value *LaneExpr = Lane.getAsRuntimeExpr(Builder, VF);
  if (auto *StructTy = dyn_cast<StructType>(WideValue->getType())) {
    // We must handle each element of a vectorized struct type.
    for (unsigned I = 0, E = StructTy->getNumElements(); I != E; I++) {
      Value *ScalarValue = Builder.CreateExtractValue(ScalarInst, I);
      Value *VectorValue = Builder.CreateExtractValue(WideValue, I);
      VectorValue =
          Builder.CreateInsertElement(VectorValue, ScalarValue, LaneExpr);
      WideValue = Builder.CreateInsertValue(WideValue, VectorValue, I);
    }
  } else {
    WideValue = Builder.CreateInsertElement(WideValue, ScalarInst, LaneExpr);
  }
  return WideValue;
}

BasicBlock *VPBasicBlock::createEmptyBasicBlock(VPTransformState &State) {
  auto &CFG = State.CFG;
  // BB stands for IR BasicBlocks. VPBB stands for VPlan VPBasicBlocks.
  // Pred stands for Predessor. Prev stands for Previous - last visited/created.
  BasicBlock *PrevBB = CFG.PrevBB;
  BasicBlock *NewBB = BasicBlock::Create(PrevBB->getContext(), getName(),
                                         PrevBB->getParent(), CFG.ExitBB);
  LLVM_DEBUG(dbgs() << "LV: created " << NewBB->getName() << '\n');

  return NewBB;
}

void VPBasicBlock::connectToPredecessors(VPTransformState &State) {
  auto &CFG = State.CFG;
  BasicBlock *NewBB = CFG.VPBB2IRBB[this];

  // Register NewBB in its loop. In innermost loops its the same for all
  // BB's.
  Loop *ParentLoop = State.CurrentParentLoop;
  // If this block has a sole successor that is an exit block or is an exit
  // block itself then it needs adding to the same parent loop as the exit
  // block.
  VPBlockBase *SuccOrExitVPB = getSingleSuccessor();
  SuccOrExitVPB = SuccOrExitVPB ? SuccOrExitVPB : this;
  if (State.Plan->isExitBlock(SuccOrExitVPB)) {
    ParentLoop = State.LI->getLoopFor(
        cast<VPIRBasicBlock>(SuccOrExitVPB)->getIRBasicBlock());
  }

  if (ParentLoop && !State.LI->getLoopFor(NewBB))
    ParentLoop->addBasicBlockToLoop(NewBB, *State.LI);

  SmallVector<VPBlockBase *> Preds;
  if (VPBlockUtils::isHeader(this, State.VPDT)) {
    // There's no block for the latch yet, connect to the preheader only.
    Preds = {getPredecessors()[0]};
  } else {
    Preds = to_vector(getPredecessors());
  }

  // Hook up the new basic block to its predecessors.
  for (VPBlockBase *PredVPBlock : Preds) {
    VPBasicBlock *PredVPBB = PredVPBlock->getExitingBasicBlock();
    auto &PredVPSuccessors = PredVPBB->getHierarchicalSuccessors();
    assert(CFG.VPBB2IRBB.contains(PredVPBB) &&
           "Predecessor basic-block not found building successor.");
    BasicBlock *PredBB = CFG.VPBB2IRBB[PredVPBB];
    auto *PredBBTerminator = PredBB->getTerminator();
    LLVM_DEBUG(dbgs() << "LV: draw edge from " << PredBB->getName() << '\n');

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
      // Branches to VPIRBasicBlocks must have the same successors in VPlan as
      // in the original IR, except when the predecessor is the entry block.
      // This enables including SCEV and memory runtime check blocks in VPlan.
      // TODO: Remove exception by modeling the terminator of entry block using
      // BranchOnCond.
      unsigned idx = PredVPSuccessors.front() == this ? 0 : 1;
      assert((TermBr && (!TermBr->getSuccessor(idx) ||
                         (isa<VPIRBasicBlock>(this) &&
                          (TermBr->getSuccessor(idx) == NewBB ||
                           PredVPBlock == getPlan()->getEntry())))) &&
             "Trying to reset an existing successor block.");
      TermBr->setSuccessor(idx, NewBB);
    }
    CFG.DTU.applyUpdates({{DominatorTree::Insert, PredBB, NewBB}});
  }
}

void VPIRBasicBlock::execute(VPTransformState *State) {
  assert(getHierarchicalSuccessors().size() <= 2 &&
         "VPIRBasicBlock can have at most two successors at the moment!");
  // Move completely disconnected blocks to their final position.
  if (IRBB->hasNPredecessors(0) && succ_begin(IRBB) == succ_end(IRBB))
    IRBB->moveAfter(State->CFG.PrevBB);
  State->Builder.SetInsertPoint(IRBB->getTerminator());
  State->CFG.PrevBB = IRBB;
  State->CFG.VPBB2IRBB[this] = IRBB;
  executeRecipes(State, IRBB);
  // Create a branch instruction to terminate IRBB if one was not created yet
  // and is needed.
  if (getSingleSuccessor() && isa<UnreachableInst>(IRBB->getTerminator())) {
    auto *Br = State->Builder.CreateBr(IRBB);
    Br->setOperand(0, nullptr);
    IRBB->getTerminator()->eraseFromParent();
  } else {
    assert(
        (getNumSuccessors() == 0 || isa<BranchInst>(IRBB->getTerminator())) &&
        "other blocks must be terminated by a branch");
  }

  connectToPredecessors(*State);
}

VPIRBasicBlock *VPIRBasicBlock::clone() {
  auto *NewBlock = getPlan()->createEmptyVPIRBasicBlock(IRBB);
  for (VPRecipeBase &R : Recipes)
    NewBlock->appendRecipe(R.clone());
  return NewBlock;
}

void VPBasicBlock::execute(VPTransformState *State) {
  bool Replica = bool(State->Lane);
  BasicBlock *NewBB = State->CFG.PrevBB; // Reuse it if possible.

  if (VPBlockUtils::isHeader(this, State->VPDT)) {
    // Create and register the new vector loop.
    Loop *PrevParentLoop = State->CurrentParentLoop;
    State->CurrentParentLoop = State->LI->AllocateLoop();

    // Insert the new loop into the loop nest and register the new basic blocks
    // before calling any utilities such as SCEV that require valid LoopInfo.
    if (PrevParentLoop)
      PrevParentLoop->addChildLoop(State->CurrentParentLoop);
    else
      State->LI->addTopLevelLoop(State->CurrentParentLoop);
  }

  auto IsReplicateRegion = [](VPBlockBase *BB) {
    auto *R = dyn_cast_or_null<VPRegionBlock>(BB);
    assert((!R || R->isReplicator()) &&
           "only replicate region blocks should remain");
    return R;
  };
  // 1. Create an IR basic block.
  if ((Replica && this == getParent()->getEntry()) ||
      IsReplicateRegion(getSingleHierarchicalPredecessor())) {
    // Reuse the previous basic block if the current VPBB is either
    //  * the entry to a replicate region, or
    //  * the exit of a replicate region.
    State->CFG.VPBB2IRBB[this] = NewBB;
  } else {
    NewBB = createEmptyBasicBlock(*State);

    State->Builder.SetInsertPoint(NewBB);
    // Temporarily terminate with unreachable until CFG is rewired.
    UnreachableInst *Terminator = State->Builder.CreateUnreachable();
    State->Builder.SetInsertPoint(Terminator);

    State->CFG.PrevBB = NewBB;
    State->CFG.VPBB2IRBB[this] = NewBB;
    connectToPredecessors(*State);
  }

  // 2. Fill the IR basic block with IR instructions.
  executeRecipes(State, NewBB);

  // If this block is a latch, update CurrentParentLoop.
  if (VPBlockUtils::isLatch(this, State->VPDT))
    State->CurrentParentLoop = State->CurrentParentLoop->getParentLoop();
}

VPBasicBlock *VPBasicBlock::clone() {
  auto *NewBlock = getPlan()->createVPBasicBlock(getName());
  for (VPRecipeBase &R : *this)
    NewBlock->appendRecipe(R.clone());
  return NewBlock;
}

void VPBasicBlock::executeRecipes(VPTransformState *State, BasicBlock *BB) {
  LLVM_DEBUG(dbgs() << "LV: vectorizing VPBB: " << getName()
                    << " in BB: " << BB->getName() << '\n');

  State->CFG.PrevVPBB = this;

  for (VPRecipeBase &Recipe : Recipes) {
    State->setDebugLocFrom(Recipe.getDebugLoc());
    Recipe.execute(*State);
  }

  LLVM_DEBUG(dbgs() << "LV: filled BB: " << *BB);
}

VPBasicBlock *VPBasicBlock::splitAt(iterator SplitAt) {
  assert((SplitAt == end() || SplitAt->getParent() == this) &&
         "can only split at a position in the same block");

  // Create new empty block after the block to split.
  auto *SplitBlock = getPlan()->createVPBasicBlock(getName() + ".split");
  VPBlockUtils::insertBlockAfter(SplitBlock, this);

  // Finally, move the recipes starting at SplitAt to new block.
  for (VPRecipeBase &ToMove :
       make_early_inc_range(make_range(SplitAt, this->end())))
    ToMove.moveBefore(*SplitBlock, SplitBlock->end());

  return SplitBlock;
}

/// Return the enclosing loop region for region \p P. The templated version is
/// used to support both const and non-const block arguments.
template <typename T> static T *getEnclosingLoopRegionForRegion(T *P) {
  if (P && P->isReplicator()) {
    P = P->getParent();
    // Multiple loop regions can be nested, but replicate regions can only be
    // nested inside a loop region or must be outside any other region.
    assert((!P || !P->isReplicator()) && "unexpected nested replicate regions");
  }
  return P;
}

VPRegionBlock *VPBasicBlock::getEnclosingLoopRegion() {
  return getEnclosingLoopRegionForRegion(getParent());
}

const VPRegionBlock *VPBasicBlock::getEnclosingLoopRegion() const {
  return getEnclosingLoopRegionForRegion(getParent());
}

static bool hasConditionalTerminator(const VPBasicBlock *VPBB) {
  if (VPBB->empty()) {
    assert(
        VPBB->getNumSuccessors() < 2 &&
        "block with multiple successors doesn't have a recipe as terminator");
    return false;
  }

  const VPRecipeBase *R = &VPBB->back();
  bool IsSwitch = isa<VPInstruction>(R) &&
                  cast<VPInstruction>(R)->getOpcode() == Instruction::Switch;
  bool IsCondBranch = isa<VPBranchOnMaskRecipe>(R) ||
                      match(R, m_BranchOnCond(m_VPValue())) ||
                      match(R, m_BranchOnCount(m_VPValue(), m_VPValue()));
  (void)IsCondBranch;
  (void)IsSwitch;
  if (VPBB->getNumSuccessors() == 2 ||
      (VPBB->isExiting() && !VPBB->getParent()->isReplicator())) {
    assert((IsCondBranch || IsSwitch) &&
           "block with multiple successors not terminated by "
           "conditional branch nor switch recipe");

    return true;
  }

  if (VPBB->getNumSuccessors() > 2) {
    assert(IsSwitch && "block with more than 2 successors not terminated by "
                       "a switch recipe");
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
  return getParent() && getParent()->getExitingBasicBlock() == this;
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPBlockBase::print(raw_ostream &O) const {
  VPSlotTracker SlotTracker(getPlan());
  print(O, "", SlotTracker);
}

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

static std::pair<VPBlockBase *, VPBlockBase *> cloneFrom(VPBlockBase *Entry);

// Clone the CFG for all nodes reachable from \p Entry, this includes cloning
// the blocks and their recipes. Operands of cloned recipes will NOT be updated.
// Remapping of operands must be done separately. Returns a pair with the new
// entry and exiting blocks of the cloned region. If \p Entry isn't part of a
// region, return nullptr for the exiting block.
static std::pair<VPBlockBase *, VPBlockBase *> cloneFrom(VPBlockBase *Entry) {
  DenseMap<VPBlockBase *, VPBlockBase *> Old2NewVPBlocks;
  VPBlockBase *Exiting = nullptr;
  bool InRegion = Entry->getParent();
  // First, clone blocks reachable from Entry.
  for (VPBlockBase *BB : vp_depth_first_shallow(Entry)) {
    VPBlockBase *NewBB = BB->clone();
    Old2NewVPBlocks[BB] = NewBB;
    if (InRegion && BB->getNumSuccessors() == 0) {
      assert(!Exiting && "Multiple exiting blocks?");
      Exiting = BB;
    }
  }
  assert((!InRegion || Exiting) && "regions must have a single exiting block");

  // Second, update the predecessors & successors of the cloned blocks.
  for (VPBlockBase *BB : vp_depth_first_shallow(Entry)) {
    VPBlockBase *NewBB = Old2NewVPBlocks[BB];
    SmallVector<VPBlockBase *> NewPreds;
    for (VPBlockBase *Pred : BB->getPredecessors()) {
      NewPreds.push_back(Old2NewVPBlocks[Pred]);
    }
    NewBB->setPredecessors(NewPreds);
    SmallVector<VPBlockBase *> NewSuccs;
    for (VPBlockBase *Succ : BB->successors()) {
      NewSuccs.push_back(Old2NewVPBlocks[Succ]);
    }
    NewBB->setSuccessors(NewSuccs);
  }

#if !defined(NDEBUG)
  // Verify that the order of predecessors and successors matches in the cloned
  // version.
  for (const auto &[OldBB, NewBB] :
       zip(vp_depth_first_shallow(Entry),
           vp_depth_first_shallow(Old2NewVPBlocks[Entry]))) {
    for (const auto &[OldPred, NewPred] :
         zip(OldBB->getPredecessors(), NewBB->getPredecessors()))
      assert(NewPred == Old2NewVPBlocks[OldPred] && "Different predecessors");

    for (const auto &[OldSucc, NewSucc] :
         zip(OldBB->successors(), NewBB->successors()))
      assert(NewSucc == Old2NewVPBlocks[OldSucc] && "Different successors");
  }
#endif

  return std::make_pair(Old2NewVPBlocks[Entry],
                        Exiting ? Old2NewVPBlocks[Exiting] : nullptr);
}

VPRegionBlock *VPRegionBlock::clone() {
  const auto &[NewEntry, NewExiting] = cloneFrom(getEntry());
  auto *NewRegion = getPlan()->createVPRegionBlock(NewEntry, NewExiting,
                                                   getName(), isReplicator());
  for (VPBlockBase *Block : vp_depth_first_shallow(NewEntry))
    Block->setParent(NewRegion);
  return NewRegion;
}

void VPRegionBlock::execute(VPTransformState *State) {
  assert(isReplicator() &&
         "Loop regions should have been lowered to plain CFG");
  assert(!State->Lane && "Replicating a Region with non-null instance.");
  assert(!State->VF.isScalable() && "VF is assumed to be non scalable.");

  ReversePostOrderTraversal<VPBlockShallowTraversalWrapper<VPBlockBase *>> RPOT(
      Entry);
  State->Lane = VPLane(0);
  for (unsigned Lane = 0, VF = State->VF.getFixedValue(); Lane < VF; ++Lane) {
    State->Lane = VPLane(Lane, VPLane::Kind::First);
    // Visit the VPBlocks connected to \p this, starting from it.
    for (VPBlockBase *Block : RPOT) {
      LLVM_DEBUG(dbgs() << "LV: VPBlock in RPO " << Block->getName() << '\n');
      Block->execute(State);
    }
  }

  // Exit replicating mode.
  State->Lane.reset();
}

InstructionCost VPBasicBlock::cost(ElementCount VF, VPCostContext &Ctx) {
  InstructionCost Cost = 0;
  for (VPRecipeBase &R : Recipes)
    Cost += R.cost(VF, Ctx);
  return Cost;
}

const VPBasicBlock *VPBasicBlock::getCFGPredecessor(unsigned Idx) const {
  const VPBlockBase *Pred = nullptr;
  if (hasPredecessors()) {
    Pred = getPredecessors()[Idx];
  } else {
    auto *Region = getParent();
    assert(Region && !Region->isReplicator() && Region->getEntry() == this &&
           "must be in the entry block of a non-replicate region");
    assert(Idx < 2 && Region->getNumPredecessors() == 1 &&
           "loop region has a single predecessor (preheader), its entry block "
           "has 2 incoming blocks");

    // Idx ==  0 selects the predecessor of the region, Idx == 1 selects the
    // region itself whose exiting block feeds the phi across the backedge.
    Pred = Idx == 0 ? Region->getSinglePredecessor() : Region;
  }
  return Pred->getExitingBasicBlock();
}

InstructionCost VPRegionBlock::cost(ElementCount VF, VPCostContext &Ctx) {
  if (!isReplicator()) {
    InstructionCost Cost = 0;
    for (VPBlockBase *Block : vp_depth_first_shallow(getEntry()))
      Cost += Block->cost(VF, Ctx);
    InstructionCost BackedgeCost =
        ForceTargetInstructionCost.getNumOccurrences()
            ? InstructionCost(ForceTargetInstructionCost.getNumOccurrences())
            : Ctx.TTI.getCFInstrCost(Instruction::Br, Ctx.CostKind);
    LLVM_DEBUG(dbgs() << "Cost of " << BackedgeCost << " for VF " << VF
                      << ": vector loop backedge\n");
    Cost += BackedgeCost;
    return Cost;
  }

  // Compute the cost of a replicate region. Replicating isn't supported for
  // scalable vectors, return an invalid cost for them.
  // TODO: Discard scalable VPlans with replicate recipes earlier after
  // construction.
  if (VF.isScalable())
    return InstructionCost::getInvalid();

  // First compute the cost of the conditionally executed recipes, followed by
  // account for the branching cost, except if the mask is a header mask or
  // uniform condition.
  using namespace llvm::VPlanPatternMatch;
  VPBasicBlock *Then = cast<VPBasicBlock>(getEntry()->getSuccessors()[0]);
  InstructionCost ThenCost = Then->cost(VF, Ctx);

  // For the scalar case, we may not always execute the original predicated
  // block, Thus, scale the block's cost by the probability of executing it.
  if (VF.isScalar())
    return ThenCost / getPredBlockCostDivisor(Ctx.CostKind);

  return ThenCost;
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPRegionBlock::print(raw_ostream &O, const Twine &Indent,
                          VPSlotTracker &SlotTracker) const {
  O << Indent << (isReplicator() ? "<xVFxUF> " : "<x1> ") << getName() << ": {";
  auto NewIndent = Indent + "  ";
  for (auto *BlockBase : vp_depth_first_shallow(Entry)) {
    O << '\n';
    BlockBase->print(O, NewIndent, SlotTracker);
  }
  O << Indent << "}\n";

  printSuccessors(O, Indent);
}
#endif

void VPRegionBlock::dissolveToCFGLoop() {
  auto *Header = cast<VPBasicBlock>(getEntry());
  if (auto *CanIV = dyn_cast<VPCanonicalIVPHIRecipe>(&Header->front())) {
    assert(this == getPlan()->getVectorLoopRegion() &&
           "Canonical IV must be in the entry of the top-level loop region");
    auto *ScalarR = VPBuilder(CanIV).createScalarPhi(
        {CanIV->getStartValue(), CanIV->getBackedgeValue()},
        CanIV->getDebugLoc(), "index");
    CanIV->replaceAllUsesWith(ScalarR);
    CanIV->eraseFromParent();
  }

  VPBlockBase *Preheader = getSinglePredecessor();
  auto *ExitingLatch = cast<VPBasicBlock>(getExiting());
  VPBlockBase *Middle = getSingleSuccessor();
  VPBlockUtils::disconnectBlocks(Preheader, this);
  VPBlockUtils::disconnectBlocks(this, Middle);

  for (VPBlockBase *VPB : vp_depth_first_shallow(Entry))
    VPB->setParent(getParent());

  VPBlockUtils::connectBlocks(Preheader, Header);
  VPBlockUtils::connectBlocks(ExitingLatch, Middle);
  VPBlockUtils::connectBlocks(ExitingLatch, Header);
}

VPlan::VPlan(Loop *L) {
  setEntry(createVPIRBasicBlock(L->getLoopPreheader()));
  ScalarHeader = createVPIRBasicBlock(L->getHeader());

  SmallVector<BasicBlock *> IRExitBlocks;
  L->getUniqueExitBlocks(IRExitBlocks);
  for (BasicBlock *EB : IRExitBlocks)
    ExitBlocks.push_back(createVPIRBasicBlock(EB));
}

VPlan::~VPlan() {
  VPValue DummyValue;

  for (auto *VPB : CreatedBlocks) {
    if (auto *VPBB = dyn_cast<VPBasicBlock>(VPB)) {
      // Replace all operands of recipes and all VPValues defined in VPBB with
      // DummyValue so the block can be deleted.
      for (VPRecipeBase &R : *VPBB) {
        for (auto *Def : R.definedValues())
          Def->replaceAllUsesWith(&DummyValue);

        for (unsigned I = 0, E = R.getNumOperands(); I != E; I++)
          R.setOperand(I, &DummyValue);
      }
    }
    delete VPB;
  }
  for (VPValue *VPV : getLiveIns())
    delete VPV;
  if (BackedgeTakenCount)
    delete BackedgeTakenCount;
}

VPIRBasicBlock *VPlan::getExitBlock(BasicBlock *IRBB) const {
  auto Iter = find_if(getExitBlocks(), [IRBB](const VPIRBasicBlock *VPIRBB) {
    return VPIRBB->getIRBasicBlock() == IRBB;
  });
  assert(Iter != getExitBlocks().end() && "no exit block found");
  return *Iter;
}

bool VPlan::isExitBlock(VPBlockBase *VPBB) {
  return is_contained(ExitBlocks, VPBB);
}

/// Generate the code inside the preheader and body of the vectorized loop.
/// Assumes a single pre-header basic-block was created for this. Introduce
/// additional basic-blocks as needed, and fill them all.
void VPlan::execute(VPTransformState *State) {
  // Initialize CFG state.
  State->CFG.PrevVPBB = nullptr;
  State->CFG.ExitBB = State->CFG.PrevBB->getSingleSuccessor();

  // Update VPDominatorTree since VPBasicBlock may be removed after State was
  // constructed.
  State->VPDT.recalculate(*this);

  // Disconnect VectorPreHeader from ExitBB in both the CFG and DT.
  BasicBlock *VectorPreHeader = State->CFG.PrevBB;
  cast<BranchInst>(VectorPreHeader->getTerminator())->setSuccessor(0, nullptr);
  State->CFG.DTU.applyUpdates(
      {{DominatorTree::Delete, VectorPreHeader, State->CFG.ExitBB}});

  LLVM_DEBUG(dbgs() << "Executing best plan with VF=" << State->VF
                    << ", UF=" << getUF() << '\n');
  setName("Final VPlan");
  LLVM_DEBUG(dump());

  // Disconnect scalar preheader and scalar header, as the dominator tree edge
  // will be updated as part of VPlan execution. This allows keeping the DTU
  // logic generic during VPlan execution.
  BasicBlock *ScalarPh = State->CFG.ExitBB;
  State->CFG.DTU.applyUpdates(
      {{DominatorTree::Delete, ScalarPh, ScalarPh->getSingleSuccessor()}});

  ReversePostOrderTraversal<VPBlockShallowTraversalWrapper<VPBlockBase *>> RPOT(
      Entry);
  // Generate code for the VPlan, in parts of the vector skeleton, loop body and
  // successor blocks including the middle, exit and scalar preheader blocks.
  for (VPBlockBase *Block : RPOT)
    Block->execute(State);

  State->CFG.DTU.flush();

  VPBasicBlock *Header = vputils::getFirstLoopHeader(*this, State->VPDT);
  if (!Header)
    return;

  auto *LatchVPBB = cast<VPBasicBlock>(Header->getPredecessors()[1]);
  BasicBlock *VectorLatchBB = State->CFG.VPBB2IRBB[LatchVPBB];

  // Fix the latch value of canonical, reduction and first-order recurrences
  // phis in the vector loop.
  for (VPRecipeBase &R : Header->phis()) {
    // Skip phi-like recipes that generate their backedege values themselves.
    if (isa<VPWidenPHIRecipe>(&R))
      continue;

    auto *PhiR = cast<VPSingleDefRecipe>(&R);
    // VPInstructions currently model scalar Phis only.
    bool NeedsScalar = isa<VPInstruction>(PhiR) ||
                       (isa<VPReductionPHIRecipe>(PhiR) &&
                        cast<VPReductionPHIRecipe>(PhiR)->isInLoop());

    Value *Phi = State->get(PhiR, NeedsScalar);
    // VPHeaderPHIRecipe supports getBackedgeValue() but VPInstruction does
    // not.
    Value *Val = State->get(PhiR->getOperand(1), NeedsScalar);
    cast<PHINode>(Phi)->addIncoming(Val, VectorLatchBB);
  }
}

InstructionCost VPlan::cost(ElementCount VF, VPCostContext &Ctx) {
  // For now only return the cost of the vector loop region, ignoring any other
  // blocks, like the preheader or middle blocks, expect for checking them for
  // recipes with invalid costs.
  InstructionCost Cost = getVectorLoopRegion()->cost(VF, Ctx);

  // If the cost of the loop region is invalid or any recipe in the skeleton
  // outside loop regions are invalid return an invalid cost.
  if (!Cost.isValid() || any_of(VPBlockUtils::blocksOnly<VPBasicBlock>(
                                    vp_depth_first_shallow(getEntry())),
                                [&VF, &Ctx](VPBasicBlock *VPBB) {
                                  return !VPBB->cost(VF, Ctx).isValid();
                                }))
    return InstructionCost::getInvalid();

  return Cost;
}

VPRegionBlock *VPlan::getVectorLoopRegion() {
  // TODO: Cache if possible.
  for (VPBlockBase *B : vp_depth_first_shallow(getEntry()))
    if (auto *R = dyn_cast<VPRegionBlock>(B))
      return R->isReplicator() ? nullptr : R;
  return nullptr;
}

const VPRegionBlock *VPlan::getVectorLoopRegion() const {
  for (const VPBlockBase *B : vp_depth_first_shallow(getEntry()))
    if (auto *R = dyn_cast<VPRegionBlock>(B))
      return R->isReplicator() ? nullptr : R;
  return nullptr;
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPlan::printLiveIns(raw_ostream &O) const {
  VPSlotTracker SlotTracker(this);

  if (VF.getNumUsers() > 0) {
    O << "\nLive-in ";
    VF.printAsOperand(O, SlotTracker);
    O << " = VF";
  }

  if (VFxUF.getNumUsers() > 0) {
    O << "\nLive-in ";
    VFxUF.printAsOperand(O, SlotTracker);
    O << " = VF * UF";
  }

  if (VectorTripCount.getNumUsers() > 0) {
    O << "\nLive-in ";
    VectorTripCount.printAsOperand(O, SlotTracker);
    O << " = vector-trip-count";
  }

  if (BackedgeTakenCount && BackedgeTakenCount->getNumUsers()) {
    O << "\nLive-in ";
    BackedgeTakenCount->printAsOperand(O, SlotTracker);
    O << " = backedge-taken count";
  }

  O << "\n";
  if (TripCount) {
    if (TripCount->isLiveIn())
      O << "Live-in ";
    TripCount->printAsOperand(O, SlotTracker);
    O << " = original trip-count";
    O << "\n";
  }
}

LLVM_DUMP_METHOD
void VPlan::print(raw_ostream &O) const {
  VPSlotTracker SlotTracker(this);

  O << "VPlan '" << getName() << "' {";

  printLiveIns(O);

  ReversePostOrderTraversal<VPBlockShallowTraversalWrapper<const VPBlockBase *>>
      RPOT(getEntry());
  for (const VPBlockBase *Block : RPOT) {
    O << '\n';
    Block->print(O, "", SlotTracker);
  }

  O << "}\n";
}

std::string VPlan::getName() const {
  std::string Out;
  raw_string_ostream RSO(Out);
  RSO << Name << " for ";
  if (!VFs.empty()) {
    RSO << "VF={" << VFs[0];
    for (ElementCount VF : drop_begin(VFs))
      RSO << "," << VF;
    RSO << "},";
  }

  if (UFs.empty()) {
    RSO << "UF>=1";
  } else {
    RSO << "UF={" << UFs[0];
    for (unsigned UF : drop_begin(UFs))
      RSO << "," << UF;
    RSO << "}";
  }

  return Out;
}

LLVM_DUMP_METHOD
void VPlan::printDOT(raw_ostream &O) const {
  VPlanPrinter Printer(O, *this);
  Printer.dump();
}

LLVM_DUMP_METHOD
void VPlan::dump() const { print(dbgs()); }
#endif

static void remapOperands(VPBlockBase *Entry, VPBlockBase *NewEntry,
                          DenseMap<VPValue *, VPValue *> &Old2NewVPValues) {
  // Update the operands of all cloned recipes starting at NewEntry. This
  // traverses all reachable blocks. This is done in two steps, to handle cycles
  // in PHI recipes.
  ReversePostOrderTraversal<VPBlockDeepTraversalWrapper<VPBlockBase *>>
      OldDeepRPOT(Entry);
  ReversePostOrderTraversal<VPBlockDeepTraversalWrapper<VPBlockBase *>>
      NewDeepRPOT(NewEntry);
  // First, collect all mappings from old to new VPValues defined by cloned
  // recipes.
  for (const auto &[OldBB, NewBB] :
       zip(VPBlockUtils::blocksOnly<VPBasicBlock>(OldDeepRPOT),
           VPBlockUtils::blocksOnly<VPBasicBlock>(NewDeepRPOT))) {
    assert(OldBB->getRecipeList().size() == NewBB->getRecipeList().size() &&
           "blocks must have the same number of recipes");
    for (const auto &[OldR, NewR] : zip(*OldBB, *NewBB)) {
      assert(OldR.getNumOperands() == NewR.getNumOperands() &&
             "recipes must have the same number of operands");
      assert(OldR.getNumDefinedValues() == NewR.getNumDefinedValues() &&
             "recipes must define the same number of operands");
      for (const auto &[OldV, NewV] :
           zip(OldR.definedValues(), NewR.definedValues()))
        Old2NewVPValues[OldV] = NewV;
    }
  }

  // Update all operands to use cloned VPValues.
  for (VPBasicBlock *NewBB :
       VPBlockUtils::blocksOnly<VPBasicBlock>(NewDeepRPOT)) {
    for (VPRecipeBase &NewR : *NewBB)
      for (unsigned I = 0, E = NewR.getNumOperands(); I != E; ++I) {
        VPValue *NewOp = Old2NewVPValues.lookup(NewR.getOperand(I));
        NewR.setOperand(I, NewOp);
      }
  }
}

VPlan *VPlan::duplicate() {
  unsigned NumBlocksBeforeCloning = CreatedBlocks.size();
  // Clone blocks.
  const auto &[NewEntry, __] = cloneFrom(Entry);

  BasicBlock *ScalarHeaderIRBB = getScalarHeader()->getIRBasicBlock();
  VPIRBasicBlock *NewScalarHeader = nullptr;
  if (getScalarHeader()->hasPredecessors()) {
    NewScalarHeader = cast<VPIRBasicBlock>(*find_if(
        vp_depth_first_shallow(NewEntry), [ScalarHeaderIRBB](VPBlockBase *VPB) {
          auto *VPIRBB = dyn_cast<VPIRBasicBlock>(VPB);
          return VPIRBB && VPIRBB->getIRBasicBlock() == ScalarHeaderIRBB;
        }));
  } else {
    NewScalarHeader = createVPIRBasicBlock(ScalarHeaderIRBB);
  }
  // Create VPlan, clone live-ins and remap operands in the cloned blocks.
  auto *NewPlan = new VPlan(cast<VPBasicBlock>(NewEntry), NewScalarHeader);
  DenseMap<VPValue *, VPValue *> Old2NewVPValues;
  for (VPValue *OldLiveIn : getLiveIns()) {
    Old2NewVPValues[OldLiveIn] =
        NewPlan->getOrAddLiveIn(OldLiveIn->getLiveInIRValue());
  }
  Old2NewVPValues[&VectorTripCount] = &NewPlan->VectorTripCount;
  Old2NewVPValues[&VF] = &NewPlan->VF;
  Old2NewVPValues[&VFxUF] = &NewPlan->VFxUF;
  if (BackedgeTakenCount) {
    NewPlan->BackedgeTakenCount = new VPValue();
    Old2NewVPValues[BackedgeTakenCount] = NewPlan->BackedgeTakenCount;
  }
  if (TripCount && TripCount->isLiveIn())
    Old2NewVPValues[TripCount] =
        NewPlan->getOrAddLiveIn(TripCount->getLiveInIRValue());
  // else NewTripCount will be created and inserted into Old2NewVPValues when
  // TripCount is cloned. In any case NewPlan->TripCount is updated below.

  remapOperands(Entry, NewEntry, Old2NewVPValues);

  // Initialize remaining fields of cloned VPlan.
  NewPlan->VFs = VFs;
  NewPlan->UFs = UFs;
  // TODO: Adjust names.
  NewPlan->Name = Name;
  if (TripCount) {
    assert(Old2NewVPValues.contains(TripCount) &&
           "TripCount must have been added to Old2NewVPValues");
    NewPlan->TripCount = Old2NewVPValues[TripCount];
  }

  // Transfer all cloned blocks (the second half of all current blocks) from
  // current to new VPlan.
  unsigned NumBlocksAfterCloning = CreatedBlocks.size();
  for (unsigned I :
       seq<unsigned>(NumBlocksBeforeCloning, NumBlocksAfterCloning))
    NewPlan->CreatedBlocks.push_back(this->CreatedBlocks[I]);
  CreatedBlocks.truncate(NumBlocksBeforeCloning);

  // Update ExitBlocks of the new plan.
  for (VPBlockBase *VPB : NewPlan->CreatedBlocks) {
    if (VPB->getNumSuccessors() == 0 && isa<VPIRBasicBlock>(VPB) &&
        VPB != NewScalarHeader)
      NewPlan->ExitBlocks.push_back(cast<VPIRBasicBlock>(VPB));
  }

  return NewPlan;
}

VPIRBasicBlock *VPlan::createEmptyVPIRBasicBlock(BasicBlock *IRBB) {
  auto *VPIRBB = new VPIRBasicBlock(IRBB);
  CreatedBlocks.push_back(VPIRBB);
  return VPIRBB;
}

VPIRBasicBlock *VPlan::createVPIRBasicBlock(BasicBlock *IRBB) {
  auto *VPIRBB = createEmptyVPIRBasicBlock(IRBB);
  for (Instruction &I :
       make_range(IRBB->begin(), IRBB->getTerminator()->getIterator()))
    VPIRBB->appendRecipe(VPIRInstruction::create(I));
  return VPIRBB;
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

  {
    // Print live-ins.
  std::string Str;
  raw_string_ostream SS(Str);
  Plan.printLiveIns(SS);
  SmallVector<StringRef, 0> Lines;
  StringRef(Str).rtrim('\n').split(Lines, "\n");
  for (auto Line : Lines)
    OS << DOT::EscapeString(Line.str()) << "\\n";
  }

  OS << "\"]\n";
  OS << "node [shape=rect, fontname=Courier, fontsize=30]\n";
  OS << "edge [fontname=Courier, fontsize=30]\n";
  OS << "compound=true\n";

  for (const VPBlockBase *Block : vp_depth_first_shallow(Plan.getEntry()))
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
  for (const VPBlockBase *Block : vp_depth_first_shallow(Region->getEntry()))
    dumpBlock(Block);
  bumpIndent(-1);
  OS << Indent << "}\n";
  dumpEdges(Region);
}

#endif

/// Returns true if there is a vector loop region and \p VPV is defined in a
/// loop region.
static bool isDefinedInsideLoopRegions(const VPValue *VPV) {
  const VPRecipeBase *DefR = VPV->getDefiningRecipe();
  return DefR && (!DefR->getParent()->getPlan()->getVectorLoopRegion() ||
                  DefR->getParent()->getEnclosingLoopRegion());
}

bool VPValue::isDefinedOutsideLoopRegions() const {
  return !isDefinedInsideLoopRegions(this);
}
void VPValue::replaceAllUsesWith(VPValue *New) {
  replaceUsesWithIf(New, [](VPUser &, unsigned) { return true; });
}

void VPValue::replaceUsesWithIf(
    VPValue *New,
    llvm::function_ref<bool(VPUser &U, unsigned Idx)> ShouldReplace) {
  // Note that this early exit is required for correctness; the implementation
  // below relies on the number of users for this VPValue to decrease, which
  // isn't the case if this == New.
  if (this == New)
    return;

  for (unsigned J = 0; J < getNumUsers();) {
    VPUser *User = Users[J];
    bool RemovedUser = false;
    for (unsigned I = 0, E = User->getNumOperands(); I < E; ++I) {
      if (User->getOperand(I) != this || !ShouldReplace(*User, I))
        continue;

      RemovedUser = true;
      User->setOperand(I, New);
    }
    // If a user got removed after updating the current user, the next user to
    // update will be moved to the current position, so we only need to
    // increment the index if the number of users did not change.
    if (!RemovedUser)
      J++;
  }
}

void VPUser::replaceUsesOfWith(VPValue *From, VPValue *To) {
  for (unsigned Idx = 0; Idx != getNumOperands(); ++Idx) {
    if (getOperand(Idx) == From)
      setOperand(Idx, To);
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPValue::printAsOperand(raw_ostream &OS, VPSlotTracker &Tracker) const {
  OS << Tracker.getOrCreateName(this);
}

void VPUser::printOperands(raw_ostream &O, VPSlotTracker &SlotTracker) const {
  interleaveComma(operands(), O, [&O, &SlotTracker](VPValue *Op) {
    Op->printAsOperand(O, SlotTracker);
  });
}
#endif

void VPSlotTracker::assignName(const VPValue *V) {
  assert(!VPValue2Name.contains(V) && "VPValue already has a name!");
  auto *UV = V->getUnderlyingValue();
  auto *VPI = dyn_cast_or_null<VPInstruction>(V->getDefiningRecipe());
  if (!UV && !(VPI && !VPI->getName().empty())) {
    VPValue2Name[V] = (Twine("vp<%") + Twine(NextSlot) + ">").str();
    NextSlot++;
    return;
  }

  // Use the name of the underlying Value, wrapped in "ir<>", and versioned by
  // appending ".Number" to the name if there are multiple uses.
  std::string Name;
  if (UV)
    Name = getName(UV);
  else
    Name = VPI->getName();

  assert(!Name.empty() && "Name cannot be empty.");
  StringRef Prefix = UV ? "ir<" : "vp<%";
  std::string BaseName = (Twine(Prefix) + Name + Twine(">")).str();

  // First assign the base name for V.
  const auto &[A, _] = VPValue2Name.try_emplace(V, BaseName);
  // Integer or FP constants with different types will result in he same string
  // due to stripping types.
  if (V->isLiveIn() && isa<ConstantInt, ConstantFP>(UV))
    return;

  // If it is already used by C > 0 other VPValues, increase the version counter
  // C and use it for V.
  const auto &[C, UseInserted] = BaseName2Version.try_emplace(BaseName, 0);
  if (!UseInserted) {
    C->second++;
    A->second = (BaseName + Twine(".") + Twine(C->second)).str();
  }
}

void VPSlotTracker::assignNames(const VPlan &Plan) {
  if (Plan.VF.getNumUsers() > 0)
    assignName(&Plan.VF);
  if (Plan.VFxUF.getNumUsers() > 0)
    assignName(&Plan.VFxUF);
  assignName(&Plan.VectorTripCount);
  if (Plan.BackedgeTakenCount)
    assignName(Plan.BackedgeTakenCount);
  for (VPValue *LI : Plan.getLiveIns())
    assignName(LI);

  ReversePostOrderTraversal<VPBlockDeepTraversalWrapper<const VPBlockBase *>>
      RPOT(VPBlockDeepTraversalWrapper<const VPBlockBase *>(Plan.getEntry()));
  for (const VPBasicBlock *VPBB :
       VPBlockUtils::blocksOnly<const VPBasicBlock>(RPOT))
    assignNames(VPBB);
}

void VPSlotTracker::assignNames(const VPBasicBlock *VPBB) {
  for (const VPRecipeBase &Recipe : *VPBB)
    for (VPValue *Def : Recipe.definedValues())
      assignName(Def);
}

std::string VPSlotTracker::getName(const Value *V) {
  std::string Name;
  raw_string_ostream S(Name);
  if (V->hasName() || !isa<Instruction>(V)) {
    V->printAsOperand(S, false);
    return Name;
  }

  if (!MST) {
    // Lazily create the ModuleSlotTracker when we first hit an unnamed
    // instruction.
    auto *I = cast<Instruction>(V);
    // This check is required to support unit tests with incomplete IR.
    if (I->getParent()) {
      MST = std::make_unique<ModuleSlotTracker>(I->getModule());
      MST->incorporateFunction(*I->getFunction());
    } else {
      MST = std::make_unique<ModuleSlotTracker>(nullptr);
    }
  }
  V->printAsOperand(S, false, *MST);
  return Name;
}

std::string VPSlotTracker::getOrCreateName(const VPValue *V) const {
  std::string Name = VPValue2Name.lookup(V);
  if (!Name.empty())
    return Name;

  // If no name was assigned, no VPlan was provided when creating the slot
  // tracker or it is not reachable from the provided VPlan. This can happen,
  // e.g. when trying to print a recipe that has not been inserted into a VPlan
  // in a debugger.
  // TODO: Update VPSlotTracker constructor to assign names to recipes &
  // VPValues not associated with a VPlan, instead of constructing names ad-hoc
  // here.
  const VPRecipeBase *DefR = V->getDefiningRecipe();
  (void)DefR;
  assert((!DefR || !DefR->getParent() || !DefR->getParent()->getPlan()) &&
         "VPValue defined by a recipe in a VPlan?");

  // Use the underlying value's name, if there is one.
  if (auto *UV = V->getUnderlyingValue()) {
    std::string Name;
    raw_string_ostream S(Name);
    UV->printAsOperand(S, false);
    return (Twine("ir<") + Name + ">").str();
  }

  return "<badref>";
}

bool LoopVectorizationPlanner::getDecisionAndClampRange(
    const std::function<bool(ElementCount)> &Predicate, VFRange &Range) {
  assert(!Range.isEmpty() && "Trying to test an empty VF range.");
  bool PredicateAtRangeStart = Predicate(Range.Start);

  for (ElementCount TmpVF : VFRange(Range.Start * 2, Range.End))
    if (Predicate(TmpVF) != PredicateAtRangeStart) {
      Range.End = TmpVF;
      break;
    }

  return PredicateAtRangeStart;
}

/// Build VPlans for the full range of feasible VF's = {\p MinVF, 2 * \p MinVF,
/// 4 * \p MinVF, ..., \p MaxVF} by repeatedly building a VPlan for a sub-range
/// of VF's starting at a given VF and extending it as much as possible. Each
/// vectorization decision can potentially shorten this sub-range during
/// buildVPlan().
void LoopVectorizationPlanner::buildVPlans(ElementCount MinVF,
                                           ElementCount MaxVF) {
  auto MaxVFTimes2 = MaxVF * 2;
  for (ElementCount VF = MinVF; ElementCount::isKnownLT(VF, MaxVFTimes2);) {
    VFRange SubRange = {VF, MaxVFTimes2};
    if (auto Plan = tryToBuildVPlan(SubRange)) {
      VPlanTransforms::optimize(*Plan);
      // Update the name of the latch of the top-level vector loop region region
      // after optimizations which includes block folding.
      Plan->getVectorLoopRegion()->getExiting()->setName("vector.latch");
      VPlans.push_back(std::move(Plan));
    }
    VF = SubRange.End;
  }
}

VPlan &LoopVectorizationPlanner::getPlanFor(ElementCount VF) const {
  assert(count_if(VPlans,
                  [VF](const VPlanPtr &Plan) { return Plan->hasVF(VF); }) ==
             1 &&
         "Multiple VPlans for VF.");

  for (const VPlanPtr &Plan : VPlans) {
    if (Plan->hasVF(VF))
      return *Plan.get();
  }
  llvm_unreachable("No plan found!");
}

static void addRuntimeUnrollDisableMetaData(Loop *L) {
  SmallVector<Metadata *, 4> MDs;
  // Reserve first location for self reference to the LoopID metadata node.
  MDs.push_back(nullptr);
  bool IsUnrollMetadata = false;
  MDNode *LoopID = L->getLoopID();
  if (LoopID) {
    // First find existing loop unrolling disable metadata.
    for (unsigned I = 1, IE = LoopID->getNumOperands(); I < IE; ++I) {
      auto *MD = dyn_cast<MDNode>(LoopID->getOperand(I));
      if (MD) {
        const auto *S = dyn_cast<MDString>(MD->getOperand(0));
        if (!S)
          continue;
        if (S->getString().starts_with("llvm.loop.unroll.runtime.disable"))
          continue;
        IsUnrollMetadata =
            S->getString().starts_with("llvm.loop.unroll.disable");
      }
      MDs.push_back(LoopID->getOperand(I));
    }
  }

  if (!IsUnrollMetadata) {
    // Add runtime unroll disable metadata.
    LLVMContext &Context = L->getHeader()->getContext();
    SmallVector<Metadata *, 1> DisableOperands;
    DisableOperands.push_back(
        MDString::get(Context, "llvm.loop.unroll.runtime.disable"));
    MDNode *DisableNode = MDNode::get(Context, DisableOperands);
    MDs.push_back(DisableNode);
    MDNode *NewLoopID = MDNode::get(Context, MDs);
    // Set operand 0 to refer to the loop id itself.
    NewLoopID->replaceOperandWith(0, NewLoopID);
    L->setLoopID(NewLoopID);
  }
}

void LoopVectorizationPlanner::updateLoopMetadataAndProfileInfo(
    Loop *VectorLoop, VPBasicBlock *HeaderVPBB, bool VectorizingEpilogue,
    unsigned EstimatedVFxUF, bool DisableRuntimeUnroll) {
  MDNode *LID = OrigLoop->getLoopID();
  // Update the metadata of the scalar loop. Skip the update when vectorizing
  // the epilogue loop, to ensure it is only updated once.
  if (!VectorizingEpilogue) {
    std::optional<MDNode *> RemainderLoopID = makeFollowupLoopID(
        LID, {LLVMLoopVectorizeFollowupAll, LLVMLoopVectorizeFollowupEpilogue});
    if (RemainderLoopID) {
      OrigLoop->setLoopID(*RemainderLoopID);
    } else {
      if (DisableRuntimeUnroll)
        addRuntimeUnrollDisableMetaData(OrigLoop);

      LoopVectorizeHints Hints(OrigLoop, true, *ORE);
      Hints.setAlreadyVectorized();
    }
  }

  if (!VectorLoop)
    return;

  if (std::optional<MDNode *> VectorizedLoopID =
          makeFollowupLoopID(LID, {LLVMLoopVectorizeFollowupAll,
                                   LLVMLoopVectorizeFollowupVectorized})) {
    VectorLoop->setLoopID(*VectorizedLoopID);
  } else {
    // Keep all loop hints from the original loop on the vector loop (we'll
    // replace the vectorizer-specific hints below).
    if (LID)
      VectorLoop->setLoopID(LID);

    if (!VectorizingEpilogue) {
      LoopVectorizeHints Hints(VectorLoop, true, *ORE);
      Hints.setAlreadyVectorized();
    }

    // Check if it's EVL-vectorized and mark the corresponding metadata.
    bool IsEVLVectorized =
        llvm::any_of(*HeaderVPBB, [](const VPRecipeBase &Recipe) {
          // Looking for the ExplictVectorLength VPInstruction.
          if (const auto *VI = dyn_cast<VPInstruction>(&Recipe))
            return VI->getOpcode() == VPInstruction::ExplicitVectorLength;
          return false;
        });
    if (IsEVLVectorized) {
      LLVMContext &Context = VectorLoop->getHeader()->getContext();
      MDNode *LoopID = VectorLoop->getLoopID();
      auto *IsEVLVectorizedMD = MDNode::get(
          Context,
          {MDString::get(Context, "llvm.loop.isvectorized.tailfoldingstyle"),
           MDString::get(Context, "evl")});
      MDNode *NewLoopID = makePostTransformationMetadata(Context, LoopID, {},
                                                         {IsEVLVectorizedMD});
      VectorLoop->setLoopID(NewLoopID);
    }
  }
  TargetTransformInfo::UnrollingPreferences UP;
  TTI.getUnrollingPreferences(VectorLoop, *PSE.getSE(), UP, ORE);
  if (!UP.UnrollVectorizedLoop || VectorizingEpilogue)
    addRuntimeUnrollDisableMetaData(VectorLoop);

  // Set/update profile weights for the vector and remainder loops as original
  // loop iterations are now distributed among them. Note that original loop
  // becomes the scalar remainder loop after vectorization.
  //
  // For cases like foldTailByMasking() and requiresScalarEpiloque() we may
  // end up getting slightly roughened result but that should be OK since
  // profile is not inherently precise anyway. Note also possible bypass of
  // vector code caused by legality checks is ignored, assigning all the weight
  // to the vector loop, optimistically.
  //
  // For scalable vectorization we can't know at compile time how many
  // iterations of the loop are handled in one vector iteration, so instead
  // use the value of vscale used for tuning.
  setProfileInfoAfterUnrolling(OrigLoop, VectorLoop, OrigLoop, EstimatedVFxUF);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void LoopVectorizationPlanner::printPlans(raw_ostream &O) {
  if (VPlans.empty()) {
    O << "LV: No VPlans built.\n";
    return;
  }
  for (const auto &Plan : VPlans)
    if (PrintVPlansInDotFormat)
      Plan->printDOT(O);
    else
      Plan->print(O);
}
#endif

TargetTransformInfo::OperandValueInfo
VPCostContext::getOperandInfo(VPValue *V) const {
  if (!V->isLiveIn())
    return {};

  return TTI::getOperandInfo(V->getLiveInIRValue());
}
