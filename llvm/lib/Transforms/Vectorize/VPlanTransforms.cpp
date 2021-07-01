//===-- VPlanTransforms.cpp - Utility VPlan to VPlan transforms -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a set of utility VPlan to VPlan transformations.
///
//===----------------------------------------------------------------------===//

#include "VPlanTransforms.h"
#include "llvm/ADT/PostOrderIterator.h"

using namespace llvm;

void VPlanTransforms::VPInstructionsToVPRecipes(
    Loop *OrigLoop, VPlanPtr &Plan,
    LoopVectorizationLegality::InductionList &Inductions,
    SmallPtrSetImpl<Instruction *> &DeadInstructions, ScalarEvolution &SE) {

  auto *TopRegion = cast<VPRegionBlock>(Plan->getEntry());
  ReversePostOrderTraversal<VPBlockBase *> RPOT(TopRegion->getEntry());

  for (VPBlockBase *Base : RPOT) {
    // Do not widen instructions in pre-header and exit blocks.
    if (Base->getNumPredecessors() == 0 || Base->getNumSuccessors() == 0)
      continue;

    VPBasicBlock *VPBB = Base->getEntryBasicBlock();
    // Introduce each ingredient into VPlan.
    for (auto I = VPBB->begin(), E = VPBB->end(); I != E;) {
      VPRecipeBase *Ingredient = &*I++;
      VPValue *VPV = Ingredient->getVPSingleValue();
      Instruction *Inst = cast<Instruction>(VPV->getUnderlyingValue());
      if (DeadInstructions.count(Inst)) {
        VPValue DummyValue;
        VPV->replaceAllUsesWith(&DummyValue);
        Ingredient->eraseFromParent();
        continue;
      }

      VPRecipeBase *NewRecipe = nullptr;
      if (auto *VPPhi = dyn_cast<VPWidenPHIRecipe>(Ingredient)) {
        auto *Phi = cast<PHINode>(VPPhi->getUnderlyingValue());
        InductionDescriptor II = Inductions.lookup(Phi);
        if (II.getKind() == InductionDescriptor::IK_IntInduction ||
            II.getKind() == InductionDescriptor::IK_FpInduction) {
          VPValue *Start = Plan->getOrAddVPValue(II.getStartValue());
          NewRecipe = new VPWidenIntOrFpInductionRecipe(Phi, Start, nullptr);
        } else {
          Plan->addVPValue(Phi, VPPhi);
          continue;
        }
      } else {
        assert(isa<VPInstruction>(Ingredient) &&
               "only VPInstructions expected here");
        assert(!isa<PHINode>(Inst) && "phis should be handled above");
        // Create VPWidenMemoryInstructionRecipe for loads and stores.
        if (LoadInst *Load = dyn_cast<LoadInst>(Inst)) {
          NewRecipe = new VPWidenMemoryInstructionRecipe(
              *Load, Plan->getOrAddVPValue(getLoadStorePointerOperand(Inst)),
              nullptr /*Mask*/);
        } else if (StoreInst *Store = dyn_cast<StoreInst>(Inst)) {
          NewRecipe = new VPWidenMemoryInstructionRecipe(
              *Store, Plan->getOrAddVPValue(getLoadStorePointerOperand(Inst)),
              Plan->getOrAddVPValue(Store->getValueOperand()),
              nullptr /*Mask*/);
        } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Inst)) {
          NewRecipe = new VPWidenGEPRecipe(
              GEP, Plan->mapToVPValues(GEP->operands()), OrigLoop);
        } else if (CallInst *CI = dyn_cast<CallInst>(Inst)) {
          NewRecipe = new VPWidenCallRecipe(
              *CI, Plan->mapToVPValues(CI->arg_operands()));
        } else if (SelectInst *SI = dyn_cast<SelectInst>(Inst)) {
          bool InvariantCond =
              SE.isLoopInvariant(SE.getSCEV(SI->getOperand(0)), OrigLoop);
          NewRecipe = new VPWidenSelectRecipe(
              *SI, Plan->mapToVPValues(SI->operands()), InvariantCond);
        } else {
          NewRecipe =
              new VPWidenRecipe(*Inst, Plan->mapToVPValues(Inst->operands()));
        }
      }

      NewRecipe->insertBefore(Ingredient);
      if (NewRecipe->getNumDefinedValues() == 1)
        VPV->replaceAllUsesWith(NewRecipe->getVPSingleValue());
      else
        assert(NewRecipe->getNumDefinedValues() == 0 &&
               "Only recpies with zero or one defined values expected");
      Ingredient->eraseFromParent();
      Plan->removeVPValueFor(Inst);
      for (auto *Def : NewRecipe->definedValues()) {
        Plan->addVPValue(Inst, Def);
      }
    }
  }
}

bool VPlanTransforms::sinkScalarOperands(VPlan &Plan) {
  auto Iter = depth_first(
      VPBlockRecursiveTraversalWrapper<VPBlockBase *>(Plan.getEntry()));
  bool Changed = false;
  // First, collect the operands of all predicated replicate recipes as seeds
  // for sinking.
  SetVector<VPValue *> WorkList;
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(Iter)) {
    for (auto &Recipe : *VPBB) {
      auto *RepR = dyn_cast<VPReplicateRecipe>(&Recipe);
      if (!RepR || !RepR->isPredicated())
        continue;
      WorkList.insert(RepR->op_begin(), RepR->op_end());
    }
  }

  // Try to sink each replicate recipe in the worklist.
  while (!WorkList.empty()) {
    auto *C = WorkList.pop_back_val();
    auto *SinkCandidate = dyn_cast_or_null<VPReplicateRecipe>(C->Def);
    if (!SinkCandidate || SinkCandidate->isUniform())
      continue;

    // All users of SinkCandidate must be in the same block in order to perform
    // sinking. Therefore the destination block for sinking must match the block
    // containing the first user.
    auto *FirstUser = dyn_cast<VPRecipeBase>(*SinkCandidate->user_begin());
    if (!FirstUser)
      continue;
    VPBasicBlock *SinkTo = FirstUser->getParent();
    if (SinkCandidate->getParent() == SinkTo ||
        SinkCandidate->mayHaveSideEffects() ||
        SinkCandidate->mayReadOrWriteMemory())
      continue;

    // All recipe users of the sink candidate must be in the same block SinkTo.
    if (any_of(SinkCandidate->users(), [SinkTo](VPUser *U) {
          auto *UI = dyn_cast<VPRecipeBase>(U);
          return !UI || UI->getParent() != SinkTo;
        }))
      continue;

    SinkCandidate->moveBefore(*SinkTo, SinkTo->getFirstNonPhi());
    WorkList.insert(SinkCandidate->op_begin(), SinkCandidate->op_end());
    Changed = true;
  }
  return Changed;
}

/// If \p R is a region with a VPBranchOnMaskRecipe in the entry block, return
/// the mask.
VPValue *getPredicatedMask(VPRegionBlock *R) {
  auto *EntryBB = dyn_cast<VPBasicBlock>(R->getEntry());
  if (!EntryBB || EntryBB->size() != 1 ||
      !isa<VPBranchOnMaskRecipe>(EntryBB->begin()))
    return nullptr;

  return cast<VPBranchOnMaskRecipe>(&*EntryBB->begin())->getOperand(0);
}

/// If \p R is a triangle region, return the 'then' block of the triangle.
static VPBasicBlock *getPredicatedThenBlock(VPRegionBlock *R) {
  auto *EntryBB = cast<VPBasicBlock>(R->getEntry());
  if (EntryBB->getNumSuccessors() != 2)
    return nullptr;

  auto *Succ0 = dyn_cast<VPBasicBlock>(EntryBB->getSuccessors()[0]);
  auto *Succ1 = dyn_cast<VPBasicBlock>(EntryBB->getSuccessors()[1]);
  if (!Succ0 || !Succ1)
    return nullptr;

  if (Succ0->getNumSuccessors() + Succ1->getNumSuccessors() != 1)
    return nullptr;
  if (Succ0->getSingleSuccessor() == Succ1)
    return Succ0;
  if (Succ1->getSingleSuccessor() == Succ0)
    return Succ1;
  return nullptr;
}

bool VPlanTransforms::mergeReplicateRegions(VPlan &Plan) {
  SetVector<VPRegionBlock *> DeletedRegions;
  bool Changed = false;

  // Collect region blocks to process up-front, to avoid iterator invalidation
  // issues while merging regions.
  SmallVector<VPRegionBlock *, 8> CandidateRegions(
      VPBlockUtils::blocksOnly<VPRegionBlock>(depth_first(
          VPBlockRecursiveTraversalWrapper<VPBlockBase *>(Plan.getEntry()))));

  // Check if Base is a predicated triangle, followed by an empty block,
  // followed by another predicate triangle. If that's the case, move the
  // recipes from the first to the second triangle.
  for (VPRegionBlock *Region1 : CandidateRegions) {
    if (DeletedRegions.contains(Region1))
      continue;
    auto *MiddleBasicBlock =
        dyn_cast_or_null<VPBasicBlock>(Region1->getSingleSuccessor());
    if (!MiddleBasicBlock || !MiddleBasicBlock->empty())
      continue;

    auto *Region2 =
        dyn_cast_or_null<VPRegionBlock>(MiddleBasicBlock->getSingleSuccessor());
    if (!Region2)
      continue;

    VPValue *Mask1 = getPredicatedMask(Region1);
    VPValue *Mask2 = getPredicatedMask(Region2);
    if (!Mask1 || Mask1 != Mask2)
      continue;
    VPBasicBlock *Then1 = getPredicatedThenBlock(Region1);
    VPBasicBlock *Then2 = getPredicatedThenBlock(Region2);
    if (!Then1 || !Then2)
      continue;

    assert(Mask1 && Mask2 && "both region must have conditions");

    // Note: No fusion-preventing memory dependencies are expected in either
    // region. Such dependencies should be rejected during earlier dependence
    // checks, which guarantee accesses can be re-ordered for vectorization.
    //
    // If a recipe is used by a first-order recurrence phi, we cannot move it at
    // the moment: a recipe R feeding a first order recurrence phi must allow
    // for a *vector* shuffle to be inserted immediately after it, and therefore
    // if R is *scalarized and predicated* it must appear last in its basic
    // block. In addition, other recipes may need to "sink after" R, so best if
    // R not be moved at all.
    auto IsImmovableRecipe = [](VPRecipeBase &R) {
      assert(R.getNumDefinedValues() <= 1 &&
             "no multi-defs are expected in predicated blocks");
      for (VPUser *U : R.getVPValue()->users()) {
        auto *UI = dyn_cast<VPRecipeBase>(U);
        if (!UI)
          continue;
        auto *PhiR = dyn_cast<VPWidenPHIRecipe>(UI);
        if (PhiR && !PhiR->getRecurrenceDescriptor())
          return true;
      }
      return false;
    };
    if (any_of(*Then1, IsImmovableRecipe))
      continue;

    // Move recipes to the successor region.
    for (VPRecipeBase &ToMove : make_early_inc_range(reverse(*Then1)))
      ToMove.moveBefore(*Then2, Then2->getFirstNonPhi());

    auto *Merge1 = cast<VPBasicBlock>(Then1->getSingleSuccessor());
    auto *Merge2 = cast<VPBasicBlock>(Then2->getSingleSuccessor());

    // Move VPPredInstPHIRecipes from the merge block to the successor region's
    // merge block. Update all users inside the successor region to use the
    // original values.
    for (VPRecipeBase &Phi1ToMove : make_early_inc_range(reverse(*Merge1))) {
      VPValue *PredInst1 =
          cast<VPPredInstPHIRecipe>(&Phi1ToMove)->getOperand(0);
      for (VPUser *U : Phi1ToMove.getVPValue()->users()) {
        auto *UI = dyn_cast<VPRecipeBase>(U);
        if (!UI || UI->getParent() != Then2)
          continue;
        for (unsigned I = 0, E = U->getNumOperands(); I != E; ++I) {
          if (Phi1ToMove.getVPValue() != U->getOperand(I))
            continue;
          U->setOperand(I, PredInst1);
        }
      }

      Phi1ToMove.moveBefore(*Merge2, Merge2->begin());
    }

    // Finally, remove the first region.
    for (VPBlockBase *Pred : make_early_inc_range(Region1->getPredecessors())) {
      VPBlockUtils::disconnectBlocks(Pred, Region1);
      VPBlockUtils::connectBlocks(Pred, MiddleBasicBlock);
    }
    VPBlockUtils::disconnectBlocks(Region1, MiddleBasicBlock);
    DeletedRegions.insert(Region1);
  }

  for (VPRegionBlock *ToDelete : DeletedRegions)
    delete ToDelete;
  return Changed;
}
