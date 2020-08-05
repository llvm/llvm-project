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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/InstrTypes.h"
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
#include <cassert>
#include <iterator>
#include <string>
#include <vector>

using namespace llvm;
extern cl::opt<bool> EnableVPlanNativePath;

#define DEBUG_TYPE "vplan"

raw_ostream &llvm::operator<<(raw_ostream &OS, const VPValue &V) {
  const VPInstruction *Instr = dyn_cast<VPInstruction>(&V);
  VPSlotTracker SlotTracker(
      (Instr && Instr->getParent()) ? Instr->getParent()->getPlan() : nullptr);
  V.print(OS, SlotTracker);
  return OS;
}

void VPValue::print(raw_ostream &OS, VPSlotTracker &SlotTracker) const {
  if (const VPInstruction *Instr = dyn_cast<VPInstruction>(this))
    Instr->print(OS, SlotTracker);
  else
    printAsOperand(OS, SlotTracker);
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
const VPBasicBlock *VPBlockBase::getExitBasicBlock() const {
  const VPBlockBase *Block = this;
  while (const VPRegionBlock *Region = dyn_cast<VPRegionBlock>(Block))
    Block = Region->getExit();
  return cast<VPBasicBlock>(Block);
}

VPBasicBlock *VPBlockBase::getExitBasicBlock() {
  VPBlockBase *Block = this;
  while (VPRegionBlock *Region = dyn_cast<VPRegionBlock>(Block))
    Block = Region->getExit();
  return cast<VPBasicBlock>(Block);
}

VPBlockBase *VPBlockBase::getEnclosingBlockWithSuccessors() {
  if (!Successors.empty() || !Parent)
    return this;
  assert(Parent->getExit() == this &&
         "Block w/o successors not the exit of its parent.");
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
  SmallVector<VPBlockBase *, 8> Blocks;
  for (VPBlockBase *Block : depth_first(Entry))
    Blocks.push_back(Block);

  for (VPBlockBase *Block : Blocks)
    delete Block;
}

BasicBlock *
VPBasicBlock::createEmptyBasicBlock(VPTransformState::CFGState &CFG) {
  // BB stands for IR BasicBlocks. VPBB stands for VPlan VPBasicBlocks.
  // Pred stands for Predessor. Prev stands for Previous - last visited/created.
  BasicBlock *PrevBB = CFG.PrevBB;
  BasicBlock *NewBB = BasicBlock::Create(PrevBB->getContext(), getName(),
                                         PrevBB->getParent(), CFG.LastBB);
  LLVM_DEBUG(dbgs() << "LV: created " << NewBB->getName() << '\n');

  // Hook up the new basic block to its predecessors.
  for (VPBlockBase *PredVPBlock : getHierarchicalPredecessors()) {
    VPBasicBlock *PredVPBB = PredVPBlock->getExitBasicBlock();
    auto &PredVPSuccessors = PredVPBB->getSuccessors();
    BasicBlock *PredBB = CFG.VPBB2IRBB[PredVPBB];

    // In outer loop vectorization scenario, the predecessor BBlock may not yet
    // be visited(backedge). Mark the VPBasicBlock for fixup at the end of
    // vectorization. We do not encounter this case in inner loop vectorization
    // as we start out by building a loop skeleton with the vector loop header
    // and latch blocks. As a result, we never enter this function for the
    // header block in the non VPlan-native path.
    if (!PredBB) {
      assert(EnableVPlanNativePath &&
             "Unexpected null predecessor in non VPlan-native path");
      CFG.VPBBsToFix.push_back(PredVPBB);
      continue;
    }

    assert(PredBB && "Predecessor basic-block not found building successor.");
    auto *PredBBTerminator = PredBB->getTerminator();
    LLVM_DEBUG(dbgs() << "LV: draw edge from" << PredBB->getName() << '\n');
    if (isa<UnreachableInst>(PredBBTerminator)) {
      assert(PredVPSuccessors.size() == 1 &&
             "Predecessor ending w/o branch must have single successor.");
      PredBBTerminator->eraseFromParent();
      BranchInst::Create(NewBB, PredBB);
    } else {
      assert(PredVPSuccessors.size() == 2 &&
             "Predecessor ending with branch must have two successors.");
      unsigned idx = PredVPSuccessors.front() == this ? 0 : 1;
      assert(!PredBBTerminator->getSuccessor(idx) &&
             "Trying to reset an existing successor block.");
      PredBBTerminator->setSuccessor(idx, NewBB);
    }
  }
  return NewBB;
}

void VPBasicBlock::execute(VPTransformState *State) {
  bool Replica = State->Instance &&
                 !(State->Instance->Part == 0 && State->Instance->Lane == 0);
  VPBasicBlock *PrevVPBB = State->CFG.PrevVPBB;
  VPBlockBase *SingleHPred = nullptr;
  BasicBlock *NewBB = State->CFG.PrevBB; // Reuse it if possible.

  // 1. Create an IR basic block, or reuse the last one if possible.
  // The last IR basic block is reused, as an optimization, in three cases:
  // A. the first VPBB reuses the loop header BB - when PrevVPBB is null;
  // B. when the current VPBB has a single (hierarchical) predecessor which
  //    is PrevVPBB and the latter has a single (hierarchical) successor; and
  // C. when the current VPBB is an entry of a region replica - where PrevVPBB
  //    is the exit of this region from a previous instance, or the predecessor
  //    of this region.
  if (PrevVPBB && /* A */
      !((SingleHPred = getSingleHierarchicalPredecessor()) &&
        SingleHPred->getExitBasicBlock() == PrevVPBB &&
        PrevVPBB->getSingleHierarchicalSuccessor()) && /* B */
      !(Replica && getPredecessors().empty())) {       /* C */
    NewBB = createEmptyBasicBlock(State->CFG);
    State->Builder.SetInsertPoint(NewBB);
    // Temporarily terminate with unreachable until CFG is rewired.
    UnreachableInst *Terminator = State->Builder.CreateUnreachable();
    State->Builder.SetInsertPoint(Terminator);
    // Register NewBB in its loop. In innermost loops its the same for all BB's.
    Loop *L = State->LI->getLoopFor(State->CFG.LastBB);
    L->addBasicBlockToLoop(NewBB, *State->LI);
    State->CFG.PrevBB = NewBB;
  }

  // 2. Fill the IR basic block with IR instructions.
  LLVM_DEBUG(dbgs() << "LV: vectorizing VPBB:" << getName()
                    << " in BB:" << NewBB->getName() << '\n');

  State->CFG.VPBB2IRBB[this] = NewBB;
  State->CFG.PrevVPBB = this;

  for (VPRecipeBase &Recipe : Recipes)
    Recipe.execute(*State);

  VPValue *CBV;
  if (EnableVPlanNativePath && (CBV = getCondBit())) {
    Value *IRCBV = CBV->getUnderlyingValue();
    assert(IRCBV && "Unexpected null underlying value for condition bit");

    // Condition bit value in a VPBasicBlock is used as the branch selector. In
    // the VPlan-native path case, since all branches are uniform we generate a
    // branch instruction using the condition value from vector lane 0 and dummy
    // successors. The successors are fixed later when the successor blocks are
    // visited.
    Value *NewCond = State->Callback.getOrCreateVectorValues(IRCBV, 0);
    NewCond = State->Builder.CreateExtractElement(NewCond,
                                                  State->Builder.getInt32(0));

    // Replace the temporary unreachable terminator with the new conditional
    // branch.
    auto *CurrentTerminator = NewBB->getTerminator();
    assert(isa<UnreachableInst>(CurrentTerminator) &&
           "Expected to replace unreachable terminator with conditional "
           "branch.");
    auto *CondBr = BranchInst::Create(NewBB, nullptr, NewCond);
    CondBr->setSuccessor(0, nullptr);
    ReplaceInstWithInst(CurrentTerminator, CondBr);
  }

  LLVM_DEBUG(dbgs() << "LV: filled BB:" << *NewBB);
}

void VPRegionBlock::execute(VPTransformState *State) {
  ReversePostOrderTraversal<VPBlockBase *> RPOT(Entry);

  if (!isReplicator()) {
    // Visit the VPBlocks connected to "this", starting from it.
    for (VPBlockBase *Block : RPOT) {
      if (EnableVPlanNativePath) {
        // The inner loop vectorization path does not represent loop preheader
        // and exit blocks as part of the VPlan. In the VPlan-native path, skip
        // vectorizing loop preheader block. In future, we may replace this
        // check with the check for loop preheader.
        if (Block->getNumPredecessors() == 0)
          continue;

        // Skip vectorizing loop exit block. In future, we may replace this
        // check with the check for loop exit.
        if (Block->getNumSuccessors() == 0)
          continue;
      }

      LLVM_DEBUG(dbgs() << "LV: VPBlock in RPO " << Block->getName() << '\n');
      Block->execute(State);
    }
    return;
  }

  assert(!State->Instance && "Replicating a Region with non-null instance.");

  // Enter replicating mode.
  State->Instance = {0, 0};

  for (unsigned Part = 0, UF = State->UF; Part < UF; ++Part) {
    State->Instance->Part = Part;
    for (unsigned Lane = 0, VF = State->VF; Lane < VF; ++Lane) {
      State->Instance->Lane = Lane;
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

void VPRecipeBase::insertBefore(VPRecipeBase *InsertPos) {
  assert(!Parent && "Recipe already in some VPBasicBlock");
  assert(InsertPos->getParent() &&
         "Insertion position not in any VPBasicBlock");
  Parent = InsertPos->getParent();
  Parent->getRecipeList().insert(InsertPos->getIterator(), this);
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

void VPInstruction::generateInstruction(VPTransformState &State,
                                        unsigned Part) {
  IRBuilder<> &Builder = State.Builder;

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
    Value *VIVElem0 = State.get(getOperand(0), {Part, 0});
    // Get first lane of backedge-taken-count.
    Value *ScalarBTC = State.get(getOperand(1), {Part, 0});

    auto *Int1Ty = Type::getInt1Ty(Builder.getContext());
    auto *PredTy = FixedVectorType::get(Int1Ty, State.VF);
    Instruction *Call = Builder.CreateIntrinsic(
        Intrinsic::get_active_lane_mask, {PredTy, ScalarBTC->getType()},
        {VIVElem0, ScalarBTC}, nullptr, "active.lane.mask");
    State.set(this, Call, Part);
    break;
  }
  default:
    llvm_unreachable("Unsupported opcode for instruction");
  }
}

void VPInstruction::execute(VPTransformState &State) {
  assert(!State.Instance && "VPInstruction executing an Instance");
  for (unsigned Part = 0; Part < State.UF; ++Part)
    generateInstruction(State, Part);
}

void VPInstruction::print(raw_ostream &O, const Twine &Indent,
                          VPSlotTracker &SlotTracker) const {
  O << "\"EMIT ";
  print(O, SlotTracker);
}

void VPInstruction::print(raw_ostream &O) const {
  VPSlotTracker SlotTracker(getParent()->getPlan());
  print(O, SlotTracker);
}

void VPInstruction::print(raw_ostream &O, VPSlotTracker &SlotTracker) const {
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

  default:
    O << Instruction::getOpcodeName(getOpcode());
  }

  for (const VPValue *Operand : operands()) {
    O << " ";
    Operand->printAsOperand(O, SlotTracker);
  }
}

/// Generate the code inside the body of the vectorized loop. Assumes a single
/// LoopVectorBody basic-block was created for this. Introduce additional
/// basic-blocks as needed, and fill them all.
void VPlan::execute(VPTransformState *State) {
  // -1. Check if the backedge taken count is needed, and if so build it.
  if (BackedgeTakenCount && BackedgeTakenCount->getNumUsers()) {
    Value *TC = State->TripCount;
    IRBuilder<> Builder(State->CFG.PrevBB->getTerminator());
    auto *TCMO = Builder.CreateSub(TC, ConstantInt::get(TC->getType(), 1),
                                   "trip.count.minus.1");
    auto VF = State->VF;
    Value *VTCMO =
        VF == 1 ? TCMO : Builder.CreateVectorSplat(VF, TCMO, "broadcast");
    for (unsigned Part = 0, UF = State->UF; Part < UF; ++Part)
      State->set(BackedgeTakenCount, VTCMO, Part);
  }

  // 0. Set the reverse mapping from VPValues to Values for code generation.
  for (auto &Entry : Value2VPValue)
    State->VPValue2Value[Entry.second] = Entry.first;

  BasicBlock *VectorPreHeaderBB = State->CFG.PrevBB;
  BasicBlock *VectorHeaderBB = VectorPreHeaderBB->getSingleSuccessor();
  assert(VectorHeaderBB && "Loop preheader does not have a single successor.");

  // 1. Make room to generate basic-blocks inside loop body if needed.
  BasicBlock *VectorLatchBB = VectorHeaderBB->splitBasicBlock(
      VectorHeaderBB->getFirstInsertionPt(), "vector.body.latch");
  Loop *L = State->LI->getLoopFor(VectorHeaderBB);
  L->addBasicBlockToLoop(VectorLatchBB, *State->LI);
  // Remove the edge between Header and Latch to allow other connections.
  // Temporarily terminate with unreachable until CFG is rewired.
  // Note: this asserts the generated code's assumption that
  // getFirstInsertionPt() can be dereferenced into an Instruction.
  VectorHeaderBB->getTerminator()->eraseFromParent();
  State->Builder.SetInsertPoint(VectorHeaderBB);
  UnreachableInst *Terminator = State->Builder.CreateUnreachable();
  State->Builder.SetInsertPoint(Terminator);

  // 2. Generate code in loop body.
  State->CFG.PrevVPBB = nullptr;
  State->CFG.PrevBB = VectorHeaderBB;
  State->CFG.LastBB = VectorLatchBB;

  for (VPBlockBase *Block : depth_first(Entry))
    Block->execute(State);

  // Setup branch terminator successors for VPBBs in VPBBsToFix based on
  // VPBB's successors.
  for (auto VPBB : State->CFG.VPBBsToFix) {
    assert(EnableVPlanNativePath &&
           "Unexpected VPBBsToFix in non VPlan-native path");
    BasicBlock *BB = State->CFG.VPBB2IRBB[VPBB];
    assert(BB && "Unexpected null basic block for VPBB");

    unsigned Idx = 0;
    auto *BBTerminator = BB->getTerminator();

    for (VPBlockBase *SuccVPBlock : VPBB->getHierarchicalSuccessors()) {
      VPBasicBlock *SuccVPBB = SuccVPBlock->getEntryBasicBlock();
      BBTerminator->setSuccessor(Idx, State->CFG.VPBB2IRBB[SuccVPBB]);
      ++Idx;
    }
  }

  // 3. Merge the temporary latch created with the last basic-block filled.
  BasicBlock *LastBB = State->CFG.PrevBB;
  // Connect LastBB to VectorLatchBB to facilitate their merge.
  assert((EnableVPlanNativePath ||
          isa<UnreachableInst>(LastBB->getTerminator())) &&
         "Expected InnerLoop VPlan CFG to terminate with unreachable");
  assert((!EnableVPlanNativePath || isa<BranchInst>(LastBB->getTerminator())) &&
         "Expected VPlan CFG to terminate with branch in NativePath");
  LastBB->getTerminator()->eraseFromParent();
  BranchInst::Create(VectorLatchBB, LastBB);

  // Merge LastBB with Latch.
  bool Merged = MergeBlockIntoPredecessor(VectorLatchBB, nullptr, State->LI);
  (void)Merged;
  assert(Merged && "Could not merge last basic block with latch.");
  VectorLatchBB = LastBB;

  // We do not attempt to preserve DT for outer loop vectorization currently.
  if (!EnableVPlanNativePath)
    updateDominatorTree(State->DT, VectorPreHeaderBB, VectorLatchBB,
                        L->getExitBlock());
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD
void VPlan::dump() const { dbgs() << *this << '\n'; }
#endif

void VPlan::updateDominatorTree(DominatorTree *DT, BasicBlock *LoopPreHeaderBB,
                                BasicBlock *LoopLatchBB,
                                BasicBlock *LoopExitBB) {
  BasicBlock *LoopHeaderBB = LoopPreHeaderBB->getSingleSuccessor();
  assert(LoopHeaderBB && "Loop preheader does not have a single successor.");
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

const Twine VPlanPrinter::getUID(const VPBlockBase *Block) {
  return (isa<VPRegionBlock>(Block) ? "cluster_N" : "N") +
         Twine(getOrCreateBID(Block));
}

const Twine VPlanPrinter::getOrCreateName(const VPBlockBase *Block) {
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
  // exit basic block and the entry basic of the respective regions.
  const VPBlockBase *Tail = From->getExitBasicBlock();
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
  OS << Indent << getUID(BasicBlock) << " [label =\n";
  bumpIndent(1);
  OS << Indent << "\"" << DOT::EscapeString(BasicBlock->getName()) << ":\\n\"";
  bumpIndent(1);

  // Dump the block predicate.
  const VPValue *Pred = BasicBlock->getPredicate();
  if (Pred) {
    OS << " +\n" << Indent << " \"BlockPredicate: ";
    if (const VPInstruction *PredI = dyn_cast<VPInstruction>(Pred)) {
      PredI->printAsOperand(OS, SlotTracker);
      OS << " (" << DOT::EscapeString(PredI->getParent()->getName())
         << ")\\l\"";
    } else
      Pred->printAsOperand(OS, SlotTracker);
  }

  for (const VPRecipeBase &Recipe : *BasicBlock) {
    OS << " +\n" << Indent;
    Recipe.print(OS, Indent, SlotTracker);
    OS << "\\l\"";
  }

  // Dump the condition bit.
  const VPValue *CBV = BasicBlock->getCondBit();
  if (CBV) {
    OS << " +\n" << Indent << " \"CondBit: ";
    if (const VPInstruction *CBI = dyn_cast<VPInstruction>(CBV)) {
      CBI->printAsOperand(OS, SlotTracker);
      OS << " (" << DOT::EscapeString(CBI->getParent()->getName()) << ")\\l\"";
    } else {
      CBV->printAsOperand(OS, SlotTracker);
      OS << "\"";
    }
  }

  bumpIndent(-2);
  OS << "\n" << Indent << "]\n";
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

void VPlanPrinter::printAsIngredient(raw_ostream &O, Value *V) {
  std::string IngredientString;
  raw_string_ostream RSO(IngredientString);
  if (auto *Inst = dyn_cast<Instruction>(V)) {
    if (!Inst->getType()->isVoidTy()) {
      Inst->printAsOperand(RSO, false);
      RSO << " = ";
    }
    RSO << Inst->getOpcodeName() << " ";
    unsigned E = Inst->getNumOperands();
    if (E > 0) {
      Inst->getOperand(0)->printAsOperand(RSO, false);
      for (unsigned I = 1; I < E; ++I)
        Inst->getOperand(I)->printAsOperand(RSO << ", ", false);
    }
  } else // !Inst
    V->printAsOperand(RSO, false);
  RSO.flush();
  O << DOT::EscapeString(IngredientString);
}

void VPWidenCallRecipe::print(raw_ostream &O, const Twine &Indent,
                              VPSlotTracker &SlotTracker) const {
  O << "\"WIDEN-CALL " << VPlanIngredient(&Ingredient);
}

void VPWidenSelectRecipe::print(raw_ostream &O, const Twine &Indent,
                                VPSlotTracker &SlotTracker) const {
  O << "\"WIDEN-SELECT" << VPlanIngredient(&Ingredient)
    << (InvariantCond ? " (condition is loop invariant)" : "");
}

void VPWidenRecipe::print(raw_ostream &O, const Twine &Indent,
                          VPSlotTracker &SlotTracker) const {
  O << "\"WIDEN\\l\"";
  O << "\"  " << VPlanIngredient(&Ingredient);
}

void VPWidenIntOrFpInductionRecipe::print(raw_ostream &O, const Twine &Indent,
                                          VPSlotTracker &SlotTracker) const {
  O << "\"WIDEN-INDUCTION";
  if (Trunc) {
    O << "\\l\"";
    O << " +\n" << Indent << "\"  " << VPlanIngredient(IV) << "\\l\"";
    O << " +\n" << Indent << "\"  " << VPlanIngredient(Trunc);
  } else
    O << " " << VPlanIngredient(IV);
}

void VPWidenGEPRecipe::print(raw_ostream &O, const Twine &Indent,
                             VPSlotTracker &SlotTracker) const {
  O << "\"WIDEN-GEP ";
  O << (IsPtrLoopInvariant ? "Inv" : "Var");
  size_t IndicesNumber = IsIndexLoopInvariant.size();
  for (size_t I = 0; I < IndicesNumber; ++I)
    O << "[" << (IsIndexLoopInvariant[I] ? "Inv" : "Var") << "]";
  O << "\\l\"";
  O << " +\n" << Indent << "\"  " << VPlanIngredient(GEP);
}

void VPWidenPHIRecipe::print(raw_ostream &O, const Twine &Indent,
                             VPSlotTracker &SlotTracker) const {
  O << "\"WIDEN-PHI " << VPlanIngredient(Phi);
}

void VPBlendRecipe::print(raw_ostream &O, const Twine &Indent,
                          VPSlotTracker &SlotTracker) const {
  O << "\"BLEND ";
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

void VPReplicateRecipe::print(raw_ostream &O, const Twine &Indent,
                              VPSlotTracker &SlotTracker) const {
  O << "\"" << (IsUniform ? "CLONE " : "REPLICATE ")
    << VPlanIngredient(Ingredient);
  if (AlsoPack)
    O << " (S->V)";
}

void VPPredInstPHIRecipe::print(raw_ostream &O, const Twine &Indent,
                                VPSlotTracker &SlotTracker) const {
  O << "\"PHI-PREDICATED-INSTRUCTION " << VPlanIngredient(PredInst);
}

void VPWidenMemoryInstructionRecipe::print(raw_ostream &O, const Twine &Indent,
                                           VPSlotTracker &SlotTracker) const {
  O << "\"WIDEN " << VPlanIngredient(&Instr);
  O << ", ";
  getAddr()->printAsOperand(O, SlotTracker);
  VPValue *Mask = getMask();
  if (Mask) {
    O << ", ";
    Mask->printAsOperand(O, SlotTracker);
  }
}

void VPWidenCanonicalIVRecipe::execute(VPTransformState &State) {
  Value *CanonicalIV = State.CanonicalIV;
  Type *STy = CanonicalIV->getType();
  IRBuilder<> Builder(State.CFG.PrevBB->getTerminator());
  auto VF = State.VF;
  Value *VStart = VF == 1
                      ? CanonicalIV
                      : Builder.CreateVectorSplat(VF, CanonicalIV, "broadcast");
  for (unsigned Part = 0, UF = State.UF; Part < UF; ++Part) {
    SmallVector<Constant *, 8> Indices;
    for (unsigned Lane = 0; Lane < VF; ++Lane)
      Indices.push_back(ConstantInt::get(STy, Part * VF + Lane));
    // If VF == 1, there is only one iteration in the loop above, thus the
    // element pushed back into Indices is ConstantInt::get(STy, Part)
    Constant *VStep = VF == 1 ? Indices.back() : ConstantVector::get(Indices);
    // Add the consecutive indices to the vector value.
    Value *CanonicalVectorIV = Builder.CreateAdd(VStart, VStep, "vec.iv");
    State.set(getVPValue(), CanonicalVectorIV, Part);
  }
}

void VPWidenCanonicalIVRecipe::print(raw_ostream &O, const Twine &Indent,
                                     VPSlotTracker &SlotTracker) const {
  O << "\"EMIT ";
  getVPValue()->printAsOperand(O, SlotTracker);
  O << " = WIDEN-CANONICAL-INDUCTION";
}

template void DomTreeBuilder::Calculate<VPDominatorTree>(VPDominatorTree &DT);

void VPValue::replaceAllUsesWith(VPValue *New) {
  for (VPUser *User : users())
    for (unsigned I = 0, E = User->getNumOperands(); I < E; ++I)
      if (User->getOperand(I) == this)
        User->setOperand(I, New);
}

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
      assert(isa<VPInstruction>(&VPI) && "Can only handle VPInstructions");
      auto *VPInst = cast<VPInstruction>(&VPI);
      auto *Inst = cast<Instruction>(VPInst->getUnderlyingValue());
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
  visitRegion(cast<VPRegionBlock>(Plan.getEntry()), Old2New, IAI);
}

void VPSlotTracker::assignSlot(const VPValue *V) {
  assert(Slots.find(V) == Slots.end() && "VPValue already has a slot!");
  const Value *UV = V->getUnderlyingValue();
  if (UV)
    return;
  const auto *VPI = dyn_cast<VPInstruction>(V);
  if (VPI && !VPI->hasResult())
    return;

  Slots[V] = NextSlot++;
}

void VPSlotTracker::assignSlots(const VPBlockBase *VPBB) {
  if (auto *Region = dyn_cast<VPRegionBlock>(VPBB))
    assignSlots(Region);
  else
    assignSlots(cast<VPBasicBlock>(VPBB));
}

void VPSlotTracker::assignSlots(const VPRegionBlock *Region) {
  ReversePostOrderTraversal<const VPBlockBase *> RPOT(Region->getEntry());
  for (const VPBlockBase *Block : RPOT)
    assignSlots(Block);
}

void VPSlotTracker::assignSlots(const VPBasicBlock *VPBB) {
  for (const VPRecipeBase &Recipe : *VPBB) {
    if (const auto *VPI = dyn_cast<VPInstruction>(&Recipe))
      assignSlot(VPI);
    else if (const auto *VPIV = dyn_cast<VPWidenCanonicalIVRecipe>(&Recipe))
      assignSlot(VPIV->getVPValue());
  }
}

void VPSlotTracker::assignSlots(const VPlan &Plan) {

  for (const VPValue *V : Plan.VPExternalDefs)
    assignSlot(V);

  for (auto &E : Plan.Value2VPValue)
    if (!isa<VPInstruction>(E.second))
      assignSlot(E.second);

  for (const VPValue *V : Plan.VPCBVs)
    assignSlot(V);

  if (Plan.BackedgeTakenCount)
    assignSlot(Plan.BackedgeTakenCount);

  ReversePostOrderTraversal<const VPBlockBase *> RPOT(Plan.getEntry());
  for (const VPBlockBase *Block : RPOT)
    assignSlots(Block);
}
