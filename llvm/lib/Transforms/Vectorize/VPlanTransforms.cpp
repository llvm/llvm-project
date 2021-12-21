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
    function_ref<const InductionDescriptor *(PHINode *)>
        GetIntOrFpInductionDescriptor,
    SmallPtrSetImpl<Instruction *> &DeadInstructions, ScalarEvolution &SE) {

  auto *TopRegion = cast<VPRegionBlock>(Plan->getEntry());
  ReversePostOrderTraversal<VPBlockBase *> RPOT(TopRegion->getEntry());

  for (VPBlockBase *Base : RPOT) {
    // Do not widen instructions in pre-header and exit blocks.
    if (Base->getNumPredecessors() == 0 || Base->getNumSuccessors() == 0)
      continue;

    VPBasicBlock *VPBB = Base->getEntryBasicBlock();
    // Introduce each ingredient into VPlan.
    for (VPRecipeBase &Ingredient : llvm::make_early_inc_range(*VPBB)) {
      VPValue *VPV = Ingredient.getVPSingleValue();
      Instruction *Inst = cast<Instruction>(VPV->getUnderlyingValue());
      if (DeadInstructions.count(Inst)) {
        VPValue DummyValue;
        VPV->replaceAllUsesWith(&DummyValue);
        Ingredient.eraseFromParent();
        continue;
      }

      VPRecipeBase *NewRecipe = nullptr;
      if (auto *VPPhi = dyn_cast<VPWidenPHIRecipe>(&Ingredient)) {
        auto *Phi = cast<PHINode>(VPPhi->getUnderlyingValue());
        if (const auto *II = GetIntOrFpInductionDescriptor(Phi)) {
          VPValue *Start = Plan->getOrAddVPValue(II->getStartValue());
          NewRecipe = new VPWidenIntOrFpInductionRecipe(Phi, Start, *II);
        } else {
          Plan->addVPValue(Phi, VPPhi);
          continue;
        }
      } else {
        assert(isa<VPInstruction>(&Ingredient) &&
               "only VPInstructions expected here");
        assert(!isa<PHINode>(Inst) && "phis should be handled above");
        // Create VPWidenMemoryInstructionRecipe for loads and stores.
        if (LoadInst *Load = dyn_cast<LoadInst>(Inst)) {
          NewRecipe = new VPWidenMemoryInstructionRecipe(
              *Load, Plan->getOrAddVPValue(getLoadStorePointerOperand(Inst)),
              nullptr /*Mask*/, false /*Consecutive*/, false /*Reverse*/);
        } else if (StoreInst *Store = dyn_cast<StoreInst>(Inst)) {
          NewRecipe = new VPWidenMemoryInstructionRecipe(
              *Store, Plan->getOrAddVPValue(getLoadStorePointerOperand(Inst)),
              Plan->getOrAddVPValue(Store->getValueOperand()), nullptr /*Mask*/,
              false /*Consecutive*/, false /*Reverse*/);
        } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Inst)) {
          NewRecipe = new VPWidenGEPRecipe(
              GEP, Plan->mapToVPValues(GEP->operands()), OrigLoop);
        } else if (CallInst *CI = dyn_cast<CallInst>(Inst)) {
          NewRecipe =
              new VPWidenCallRecipe(*CI, Plan->mapToVPValues(CI->args()));
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

      NewRecipe->insertBefore(&Ingredient);
      if (NewRecipe->getNumDefinedValues() == 1)
        VPV->replaceAllUsesWith(NewRecipe->getVPSingleValue());
      else
        assert(NewRecipe->getNumDefinedValues() == 0 &&
               "Only recpies with zero or one defined values expected");
      Ingredient.eraseFromParent();
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
  SetVector<std::pair<VPBasicBlock *, VPValue *>> WorkList;
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(Iter)) {
    for (auto &Recipe : *VPBB) {
      auto *RepR = dyn_cast<VPReplicateRecipe>(&Recipe);
      if (!RepR || !RepR->isPredicated())
        continue;
      for (VPValue *Op : RepR->operands())
        WorkList.insert(std::make_pair(RepR->getParent(), Op));
    }
  }

  // Try to sink each replicate recipe in the worklist.
  while (!WorkList.empty()) {
    VPBasicBlock *SinkTo;
    VPValue *C;
    std::tie(SinkTo, C) = WorkList.pop_back_val();
    auto *SinkCandidate = dyn_cast_or_null<VPReplicateRecipe>(C->Def);
    if (!SinkCandidate || SinkCandidate->isUniform() ||
        SinkCandidate->getParent() == SinkTo ||
        SinkCandidate->mayHaveSideEffects() ||
        SinkCandidate->mayReadOrWriteMemory())
      continue;

    bool NeedsDuplicating = false;
    // All recipe users of the sink candidate must be in the same block SinkTo
    // or all users outside of SinkTo must be uniform-after-vectorization (
    // i.e., only first lane is used) . In the latter case, we need to duplicate
    // SinkCandidate. At the moment, we identify such UAV's by looking for the
    // address operands of widened memory recipes.
    auto CanSinkWithUser = [SinkTo, &NeedsDuplicating,
                            SinkCandidate](VPUser *U) {
      auto *UI = dyn_cast<VPRecipeBase>(U);
      if (!UI)
        return false;
      if (UI->getParent() == SinkTo)
        return true;
      auto *WidenI = dyn_cast<VPWidenMemoryInstructionRecipe>(UI);
      if (WidenI && WidenI->getAddr() == SinkCandidate) {
        NeedsDuplicating = true;
        return true;
      }
      return false;
    };
    if (!all_of(SinkCandidate->users(), CanSinkWithUser))
      continue;

    if (NeedsDuplicating) {
      Instruction *I = cast<Instruction>(SinkCandidate->getUnderlyingValue());
      auto *Clone =
          new VPReplicateRecipe(I, SinkCandidate->operands(), true, false);
      // TODO: add ".cloned" suffix to name of Clone's VPValue.

      Clone->insertBefore(SinkCandidate);
      SmallVector<VPUser *, 4> Users(SinkCandidate->users());
      for (auto *U : Users) {
        auto *UI = cast<VPRecipeBase>(U);
        if (UI->getParent() == SinkTo)
          continue;

        for (unsigned Idx = 0; Idx != UI->getNumOperands(); Idx++) {
          if (UI->getOperand(Idx) != SinkCandidate)
            continue;
          UI->setOperand(Idx, Clone);
        }
      }
    }
    SinkCandidate->moveBefore(*SinkTo, SinkTo->getFirstNonPhi());
    for (VPValue *Op : SinkCandidate->operands())
      WorkList.insert(std::make_pair(SinkTo, Op));
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
      VPValue *Phi1ToMoveV = Phi1ToMove.getVPSingleValue();
      SmallVector<VPUser *> Users(Phi1ToMoveV->users());
      for (VPUser *U : Users) {
        auto *UI = dyn_cast<VPRecipeBase>(U);
        if (!UI || UI->getParent() != Then2)
          continue;
        for (unsigned I = 0, E = U->getNumOperands(); I != E; ++I) {
          if (Phi1ToMoveV != U->getOperand(I))
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

void VPlanTransforms::removeRedundantInductionCasts(VPlan &Plan) {
  SmallVector<std::pair<VPRecipeBase *, VPValue *>> CastsToRemove;
  for (auto &Phi : Plan.getEntry()->getEntryBasicBlock()->phis()) {
    auto *IV = dyn_cast<VPWidenIntOrFpInductionRecipe>(&Phi);
    if (!IV || IV->getTruncInst())
      continue;

    // Visit all casts connected to IV and in Casts. Collect them.
    // remember them for removal.
    auto &Casts = IV->getInductionDescriptor().getCastInsts();
    VPValue *FindMyCast = IV;
    for (Instruction *IRCast : reverse(Casts)) {
      VPRecipeBase *FoundUserCast = nullptr;
      for (auto *U : FindMyCast->users()) {
        auto *UserCast = cast<VPRecipeBase>(U);
        if (UserCast->getNumDefinedValues() == 1 &&
            UserCast->getVPSingleValue()->getUnderlyingValue() == IRCast) {
          FoundUserCast = UserCast;
          break;
        }
      }
      assert(FoundUserCast && "Missing a cast to remove");
      CastsToRemove.emplace_back(FoundUserCast, IV);
      FindMyCast = FoundUserCast->getVPSingleValue();
    }
  }
  for (auto &E : CastsToRemove) {
    E.first->getVPSingleValue()->replaceAllUsesWith(E.second);
    E.first->eraseFromParent();
  }
}
