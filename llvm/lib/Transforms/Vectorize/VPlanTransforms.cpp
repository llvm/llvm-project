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
#include "VPlanDominatorTree.h"
#include "VPRecipeBuilder.h"
#include "VPlanCFG.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/Intrinsics.h"

using namespace llvm;

void VPlanTransforms::VPInstructionsToVPRecipes(
    VPlanPtr &Plan,
    function_ref<const InductionDescriptor *(PHINode *)>
        GetIntOrFpInductionDescriptor,
    ScalarEvolution &SE, const TargetLibraryInfo &TLI) {

  ReversePostOrderTraversal<VPBlockDeepTraversalWrapper<VPBlockBase *>> RPOT(
      Plan->getEntry());
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(RPOT)) {
    VPRecipeBase *Term = VPBB->getTerminator();
    auto EndIter = Term ? Term->getIterator() : VPBB->end();
    // Introduce each ingredient into VPlan.
    for (VPRecipeBase &Ingredient :
         make_early_inc_range(make_range(VPBB->begin(), EndIter))) {

      VPValue *VPV = Ingredient.getVPSingleValue();
      Instruction *Inst = cast<Instruction>(VPV->getUnderlyingValue());

      VPRecipeBase *NewRecipe = nullptr;
      if (auto *VPPhi = dyn_cast<VPWidenPHIRecipe>(&Ingredient)) {
        auto *Phi = cast<PHINode>(VPPhi->getUnderlyingValue());
        if (const auto *II = GetIntOrFpInductionDescriptor(Phi)) {
          VPValue *Start = Plan->getVPValueOrAddLiveIn(II->getStartValue());
          VPValue *Step =
              vputils::getOrCreateVPValueForSCEVExpr(*Plan, II->getStep(), SE);
          NewRecipe = new VPWidenIntOrFpInductionRecipe(Phi, Start, Step, *II);
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
              *Load, Ingredient.getOperand(0), nullptr /*Mask*/,
              false /*Consecutive*/, false /*Reverse*/);
        } else if (StoreInst *Store = dyn_cast<StoreInst>(Inst)) {
          NewRecipe = new VPWidenMemoryInstructionRecipe(
              *Store, Ingredient.getOperand(1), Ingredient.getOperand(0),
              nullptr /*Mask*/, false /*Consecutive*/, false /*Reverse*/);
        } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Inst)) {
          NewRecipe = new VPWidenGEPRecipe(GEP, Ingredient.operands());
        } else if (CallInst *CI = dyn_cast<CallInst>(Inst)) {
          NewRecipe =
              new VPWidenCallRecipe(*CI, drop_end(Ingredient.operands()),
                                    getVectorIntrinsicIDForCall(CI, &TLI));
        } else if (SelectInst *SI = dyn_cast<SelectInst>(Inst)) {
          NewRecipe = new VPWidenSelectRecipe(*SI, Ingredient.operands());
        } else if (auto *CI = dyn_cast<CastInst>(Inst)) {
          NewRecipe = new VPWidenCastRecipe(
              CI->getOpcode(), Ingredient.getOperand(0), CI->getType(), CI);
        } else {
          NewRecipe = new VPWidenRecipe(*Inst, Ingredient.operands());
        }
      }

      NewRecipe->insertBefore(&Ingredient);
      if (NewRecipe->getNumDefinedValues() == 1)
        VPV->replaceAllUsesWith(NewRecipe->getVPSingleValue());
      else
        assert(NewRecipe->getNumDefinedValues() == 0 &&
               "Only recpies with zero or one defined values expected");
      Ingredient.eraseFromParent();
    }
  }
}

static bool sinkScalarOperands(VPlan &Plan) {
  auto Iter = vp_depth_first_deep(Plan.getEntry());
  bool Changed = false;
  // First, collect the operands of all recipes in replicate blocks as seeds for
  // sinking.
  SetVector<std::pair<VPBasicBlock *, VPRecipeBase *>> WorkList;
  for (VPRegionBlock *VPR : VPBlockUtils::blocksOnly<VPRegionBlock>(Iter)) {
    VPBasicBlock *EntryVPBB = VPR->getEntryBasicBlock();
    if (!VPR->isReplicator() || EntryVPBB->getSuccessors().size() != 2)
      continue;
    VPBasicBlock *VPBB = dyn_cast<VPBasicBlock>(EntryVPBB->getSuccessors()[0]);
    if (!VPBB || VPBB->getSingleSuccessor() != VPR->getExitingBasicBlock())
      continue;
    for (auto &Recipe : *VPBB) {
      for (VPValue *Op : Recipe.operands())
        if (auto *Def = Op->getDefiningRecipe())
          WorkList.insert(std::make_pair(VPBB, Def));
    }
  }

  bool ScalarVFOnly = Plan.hasScalarVFOnly();
  // Try to sink each replicate or scalar IV steps recipe in the worklist.
  for (unsigned I = 0; I != WorkList.size(); ++I) {
    VPBasicBlock *SinkTo;
    VPRecipeBase *SinkCandidate;
    std::tie(SinkTo, SinkCandidate) = WorkList[I];
    if (SinkCandidate->getParent() == SinkTo ||
        SinkCandidate->mayHaveSideEffects() ||
        SinkCandidate->mayReadOrWriteMemory())
      continue;
    if (auto *RepR = dyn_cast<VPReplicateRecipe>(SinkCandidate)) {
      if (!ScalarVFOnly && RepR->isUniform())
        continue;
    } else if (!isa<VPScalarIVStepsRecipe>(SinkCandidate))
      continue;

    bool NeedsDuplicating = false;
    // All recipe users of the sink candidate must be in the same block SinkTo
    // or all users outside of SinkTo must be uniform-after-vectorization (
    // i.e., only first lane is used) . In the latter case, we need to duplicate
    // SinkCandidate.
    auto CanSinkWithUser = [SinkTo, &NeedsDuplicating,
                            SinkCandidate](VPUser *U) {
      auto *UI = dyn_cast<VPRecipeBase>(U);
      if (!UI)
        return false;
      if (UI->getParent() == SinkTo)
        return true;
      NeedsDuplicating =
          UI->onlyFirstLaneUsed(SinkCandidate->getVPSingleValue());
      // We only know how to duplicate VPRecipeRecipes for now.
      return NeedsDuplicating && isa<VPReplicateRecipe>(SinkCandidate);
    };
    if (!all_of(SinkCandidate->getVPSingleValue()->users(), CanSinkWithUser))
      continue;

    if (NeedsDuplicating) {
      if (ScalarVFOnly)
        continue;
      Instruction *I = cast<Instruction>(
          cast<VPReplicateRecipe>(SinkCandidate)->getUnderlyingValue());
      auto *Clone = new VPReplicateRecipe(I, SinkCandidate->operands(), true);
      // TODO: add ".cloned" suffix to name of Clone's VPValue.

      Clone->insertBefore(SinkCandidate);
      for (auto *U : to_vector(SinkCandidate->getVPSingleValue()->users())) {
        auto *UI = cast<VPRecipeBase>(U);
        if (UI->getParent() == SinkTo)
          continue;

        for (unsigned Idx = 0; Idx != UI->getNumOperands(); Idx++) {
          if (UI->getOperand(Idx) != SinkCandidate->getVPSingleValue())
            continue;
          UI->setOperand(Idx, Clone);
        }
      }
    }
    SinkCandidate->moveBefore(*SinkTo, SinkTo->getFirstNonPhi());
    for (VPValue *Op : SinkCandidate->operands())
      if (auto *Def = Op->getDefiningRecipe())
        WorkList.insert(std::make_pair(SinkTo, Def));
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

// Merge replicate regions in their successor region, if a replicate region
// is connected to a successor replicate region with the same predicate by a
// single, empty VPBasicBlock.
static bool mergeReplicateRegionsIntoSuccessors(VPlan &Plan) {
  SetVector<VPRegionBlock *> DeletedRegions;

  // Collect replicate regions followed by an empty block, followed by another
  // replicate region with matching masks to process front. This is to avoid
  // iterator invalidation issues while merging regions.
  SmallVector<VPRegionBlock *, 8> WorkList;
  for (VPRegionBlock *Region1 : VPBlockUtils::blocksOnly<VPRegionBlock>(
           vp_depth_first_deep(Plan.getEntry()))) {
    if (!Region1->isReplicator())
      continue;
    auto *MiddleBasicBlock =
        dyn_cast_or_null<VPBasicBlock>(Region1->getSingleSuccessor());
    if (!MiddleBasicBlock || !MiddleBasicBlock->empty())
      continue;

    auto *Region2 =
        dyn_cast_or_null<VPRegionBlock>(MiddleBasicBlock->getSingleSuccessor());
    if (!Region2 || !Region2->isReplicator())
      continue;

    VPValue *Mask1 = getPredicatedMask(Region1);
    VPValue *Mask2 = getPredicatedMask(Region2);
    if (!Mask1 || Mask1 != Mask2)
      continue;

    assert(Mask1 && Mask2 && "both region must have conditions");
    WorkList.push_back(Region1);
  }

  // Move recipes from Region1 to its successor region, if both are triangles.
  for (VPRegionBlock *Region1 : WorkList) {
    if (DeletedRegions.contains(Region1))
      continue;
    auto *MiddleBasicBlock = cast<VPBasicBlock>(Region1->getSingleSuccessor());
    auto *Region2 = cast<VPRegionBlock>(MiddleBasicBlock->getSingleSuccessor());

    VPBasicBlock *Then1 = getPredicatedThenBlock(Region1);
    VPBasicBlock *Then2 = getPredicatedThenBlock(Region2);
    if (!Then1 || !Then2)
      continue;

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
      for (VPUser *U : to_vector(Phi1ToMoveV->users())) {
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
  return !DeletedRegions.empty();
}

static VPRegionBlock *createReplicateRegion(VPReplicateRecipe *PredRecipe,
                                            VPlan &Plan) {
  Instruction *Instr = PredRecipe->getUnderlyingInstr();
  // Build the triangular if-then region.
  std::string RegionName = (Twine("pred.") + Instr->getOpcodeName()).str();
  assert(Instr->getParent() && "Predicated instruction not in any basic block");
  auto *BlockInMask = PredRecipe->getMask();
  auto *BOMRecipe = new VPBranchOnMaskRecipe(BlockInMask);
  auto *Entry = new VPBasicBlock(Twine(RegionName) + ".entry", BOMRecipe);

  // Replace predicated replicate recipe with a replicate recipe without a
  // mask but in the replicate region.
  auto *RecipeWithoutMask = new VPReplicateRecipe(
      PredRecipe->getUnderlyingInstr(),
      make_range(PredRecipe->op_begin(), std::prev(PredRecipe->op_end())),
      PredRecipe->isUniform());
  auto *Pred = new VPBasicBlock(Twine(RegionName) + ".if", RecipeWithoutMask);

  VPPredInstPHIRecipe *PHIRecipe = nullptr;
  if (PredRecipe->getNumUsers() != 0) {
    PHIRecipe = new VPPredInstPHIRecipe(RecipeWithoutMask);
    PredRecipe->replaceAllUsesWith(PHIRecipe);
    PHIRecipe->setOperand(0, RecipeWithoutMask);
  }
  PredRecipe->eraseFromParent();
  auto *Exiting = new VPBasicBlock(Twine(RegionName) + ".continue", PHIRecipe);
  VPRegionBlock *Region = new VPRegionBlock(Entry, Exiting, RegionName, true);

  // Note: first set Entry as region entry and then connect successors starting
  // from it in order, to propagate the "parent" of each VPBasicBlock.
  VPBlockUtils::insertTwoBlocksAfter(Pred, Exiting, Entry);
  VPBlockUtils::connectBlocks(Pred, Exiting);

  return Region;
}

static void addReplicateRegions(VPlan &Plan) {
  SmallVector<VPReplicateRecipe *> WorkList;
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(
           vp_depth_first_deep(Plan.getEntry()))) {
    for (VPRecipeBase &R : *VPBB)
      if (auto *RepR = dyn_cast<VPReplicateRecipe>(&R)) {
        if (RepR->isPredicated())
          WorkList.push_back(RepR);
      }
  }

  unsigned BBNum = 0;
  for (VPReplicateRecipe *RepR : WorkList) {
    VPBasicBlock *CurrentBlock = RepR->getParent();
    VPBasicBlock *SplitBlock = CurrentBlock->splitAt(RepR->getIterator());

    BasicBlock *OrigBB = RepR->getUnderlyingInstr()->getParent();
    SplitBlock->setName(
        OrigBB->hasName() ? OrigBB->getName() + "." + Twine(BBNum++) : "");
    // Record predicated instructions for above packing optimizations.
    VPBlockBase *Region = createReplicateRegion(RepR, Plan);
    Region->setParent(CurrentBlock->getParent());
    VPBlockUtils::disconnectBlocks(CurrentBlock, SplitBlock);
    VPBlockUtils::connectBlocks(CurrentBlock, Region);
    VPBlockUtils::connectBlocks(Region, SplitBlock);
  }
}

void VPlanTransforms::createAndOptimizeReplicateRegions(VPlan &Plan) {
  // Convert masked VPReplicateRecipes to if-then region blocks.
  addReplicateRegions(Plan);

  bool ShouldSimplify = true;
  while (ShouldSimplify) {
    ShouldSimplify = sinkScalarOperands(Plan);
    ShouldSimplify |= mergeReplicateRegionsIntoSuccessors(Plan);
    ShouldSimplify |= VPlanTransforms::mergeBlocksIntoPredecessors(Plan);
  }
}
bool VPlanTransforms::mergeBlocksIntoPredecessors(VPlan &Plan) {
  SmallVector<VPBasicBlock *> WorkList;
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(
           vp_depth_first_deep(Plan.getEntry()))) {
    auto *PredVPBB =
        dyn_cast_or_null<VPBasicBlock>(VPBB->getSinglePredecessor());
    if (PredVPBB && PredVPBB->getNumSuccessors() == 1)
      WorkList.push_back(VPBB);
  }

  for (VPBasicBlock *VPBB : WorkList) {
    VPBasicBlock *PredVPBB = cast<VPBasicBlock>(VPBB->getSinglePredecessor());
    for (VPRecipeBase &R : make_early_inc_range(*VPBB))
      R.moveBefore(*PredVPBB, PredVPBB->end());
    VPBlockUtils::disconnectBlocks(PredVPBB, VPBB);
    auto *ParentRegion = cast_or_null<VPRegionBlock>(VPBB->getParent());
    if (ParentRegion && ParentRegion->getExiting() == VPBB)
      ParentRegion->setExiting(PredVPBB);
    for (auto *Succ : to_vector(VPBB->successors())) {
      VPBlockUtils::disconnectBlocks(VPBB, Succ);
      VPBlockUtils::connectBlocks(PredVPBB, Succ);
    }
    delete VPBB;
  }
  return !WorkList.empty();
}

void VPlanTransforms::removeRedundantInductionCasts(VPlan &Plan) {
  for (auto &Phi : Plan.getVectorLoopRegion()->getEntryBasicBlock()->phis()) {
    auto *IV = dyn_cast<VPWidenIntOrFpInductionRecipe>(&Phi);
    if (!IV || IV->getTruncInst())
      continue;

    // A sequence of IR Casts has potentially been recorded for IV, which
    // *must be bypassed* when the IV is vectorized, because the vectorized IV
    // will produce the desired casted value. This sequence forms a def-use
    // chain and is provided in reverse order, ending with the cast that uses
    // the IV phi. Search for the recipe of the last cast in the chain and
    // replace it with the original IV. Note that only the final cast is
    // expected to have users outside the cast-chain and the dead casts left
    // over will be cleaned up later.
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
      FindMyCast = FoundUserCast->getVPSingleValue();
    }
    FindMyCast->replaceAllUsesWith(IV);
  }
}

void VPlanTransforms::removeRedundantCanonicalIVs(VPlan &Plan) {
  VPCanonicalIVPHIRecipe *CanonicalIV = Plan.getCanonicalIV();
  VPWidenCanonicalIVRecipe *WidenNewIV = nullptr;
  for (VPUser *U : CanonicalIV->users()) {
    WidenNewIV = dyn_cast<VPWidenCanonicalIVRecipe>(U);
    if (WidenNewIV)
      break;
  }

  if (!WidenNewIV)
    return;

  VPBasicBlock *HeaderVPBB = Plan.getVectorLoopRegion()->getEntryBasicBlock();
  for (VPRecipeBase &Phi : HeaderVPBB->phis()) {
    auto *WidenOriginalIV = dyn_cast<VPWidenIntOrFpInductionRecipe>(&Phi);

    if (!WidenOriginalIV || !WidenOriginalIV->isCanonical() ||
        WidenOriginalIV->getScalarType() != WidenNewIV->getScalarType())
      continue;

    // Replace WidenNewIV with WidenOriginalIV if WidenOriginalIV provides
    // everything WidenNewIV's users need. That is, WidenOriginalIV will
    // generate a vector phi or all users of WidenNewIV demand the first lane
    // only.
    if (any_of(WidenOriginalIV->users(),
               [WidenOriginalIV](VPUser *U) {
                 return !U->usesScalars(WidenOriginalIV);
               }) ||
        vputils::onlyFirstLaneUsed(WidenNewIV)) {
      WidenNewIV->replaceAllUsesWith(WidenOriginalIV);
      WidenNewIV->eraseFromParent();
      return;
    }
  }
}

void VPlanTransforms::removeDeadRecipes(VPlan &Plan) {
  ReversePostOrderTraversal<VPBlockDeepTraversalWrapper<VPBlockBase *>> RPOT(
      Plan.getEntry());

  for (VPBasicBlock *VPBB : reverse(VPBlockUtils::blocksOnly<VPBasicBlock>(RPOT))) {
    // The recipes in the block are processed in reverse order, to catch chains
    // of dead recipes.
    for (VPRecipeBase &R : make_early_inc_range(reverse(*VPBB))) {
      if (R.mayHaveSideEffects() || any_of(R.definedValues(), [](VPValue *V) {
            return V->getNumUsers() > 0;
          }))
        continue;
      R.eraseFromParent();
    }
  }
}

void VPlanTransforms::optimizeInductions(VPlan &Plan, ScalarEvolution &SE) {
  SmallVector<VPRecipeBase *> ToRemove;
  VPBasicBlock *HeaderVPBB = Plan.getVectorLoopRegion()->getEntryBasicBlock();
  bool HasOnlyVectorVFs = !Plan.hasVF(ElementCount::getFixed(1));
  for (VPRecipeBase &Phi : HeaderVPBB->phis()) {
    auto *WideIV = dyn_cast<VPWidenIntOrFpInductionRecipe>(&Phi);
    if (!WideIV)
      continue;
    if (HasOnlyVectorVFs && none_of(WideIV->users(), [WideIV](VPUser *U) {
          return U->usesScalars(WideIV);
        }))
      continue;

    auto IP = HeaderVPBB->getFirstNonPhi();
    VPCanonicalIVPHIRecipe *CanonicalIV = Plan.getCanonicalIV();
    Type *ResultTy = WideIV->getPHINode()->getType();
    if (Instruction *TruncI = WideIV->getTruncInst())
      ResultTy = TruncI->getType();
    const InductionDescriptor &ID = WideIV->getInductionDescriptor();
    VPValue *Step = WideIV->getStepValue();
    VPValue *BaseIV = CanonicalIV;
    if (!CanonicalIV->isCanonical(ID.getKind(), WideIV->getStartValue(), Step,
                                  ResultTy)) {
      BaseIV = new VPDerivedIVRecipe(ID, WideIV->getStartValue(), CanonicalIV,
                                     Step, ResultTy);
      HeaderVPBB->insert(BaseIV->getDefiningRecipe(), IP);
    }

    VPScalarIVStepsRecipe *Steps = new VPScalarIVStepsRecipe(ID, BaseIV, Step);
    HeaderVPBB->insert(Steps, IP);

    // Update scalar users of IV to use Step instead. Use SetVector to ensure
    // the list of users doesn't contain duplicates.
    SetVector<VPUser *> Users(WideIV->user_begin(), WideIV->user_end());
    for (VPUser *U : Users) {
      if (HasOnlyVectorVFs && !U->usesScalars(WideIV))
        continue;
      for (unsigned I = 0, E = U->getNumOperands(); I != E; I++) {
        if (U->getOperand(I) != WideIV)
          continue;
        U->setOperand(I, Steps);
      }
    }
  }
}

void VPlanTransforms::removeRedundantExpandSCEVRecipes(VPlan &Plan) {
  DenseMap<const SCEV *, VPValue *> SCEV2VPV;

  for (VPRecipeBase &R :
       make_early_inc_range(*Plan.getEntry()->getEntryBasicBlock())) {
    auto *ExpR = dyn_cast<VPExpandSCEVRecipe>(&R);
    if (!ExpR)
      continue;

    auto I = SCEV2VPV.insert({ExpR->getSCEV(), ExpR});
    if (I.second)
      continue;
    ExpR->replaceAllUsesWith(I.first->second);
    ExpR->eraseFromParent();
  }
}

static bool canSimplifyBranchOnCond(VPInstruction *Term) {
  VPInstruction *Not = dyn_cast<VPInstruction>(Term->getOperand(0));
  if (!Not || Not->getOpcode() != VPInstruction::Not)
    return false;

  VPInstruction *ALM = dyn_cast<VPInstruction>(Not->getOperand(0));
  return ALM && ALM->getOpcode() == VPInstruction::ActiveLaneMask;
}

void VPlanTransforms::optimizeForVFAndUF(VPlan &Plan, ElementCount BestVF,
                                         unsigned BestUF,
                                         PredicatedScalarEvolution &PSE) {
  assert(Plan.hasVF(BestVF) && "BestVF is not available in Plan");
  assert(Plan.hasUF(BestUF) && "BestUF is not available in Plan");
  VPBasicBlock *ExitingVPBB =
      Plan.getVectorLoopRegion()->getExitingBasicBlock();
  auto *Term = dyn_cast<VPInstruction>(&ExitingVPBB->back());
  // Try to simplify the branch condition if TC <= VF * UF when preparing to
  // execute the plan for the main vector loop. We only do this if the
  // terminator is:
  //  1. BranchOnCount, or
  //  2. BranchOnCond where the input is Not(ActiveLaneMask).
  if (!Term || (Term->getOpcode() != VPInstruction::BranchOnCount &&
                (Term->getOpcode() != VPInstruction::BranchOnCond ||
                 !canSimplifyBranchOnCond(Term))))
    return;

  Type *IdxTy =
      Plan.getCanonicalIV()->getStartValue()->getLiveInIRValue()->getType();
  const SCEV *TripCount = createTripCountSCEV(IdxTy, PSE);
  ScalarEvolution &SE = *PSE.getSE();
  const SCEV *C =
      SE.getConstant(TripCount->getType(), BestVF.getKnownMinValue() * BestUF);
  if (TripCount->isZero() ||
      !SE.isKnownPredicate(CmpInst::ICMP_ULE, TripCount, C))
    return;

  LLVMContext &Ctx = SE.getContext();
  auto *BOC = new VPInstruction(
      VPInstruction::BranchOnCond,
      {Plan.getVPValueOrAddLiveIn(ConstantInt::getTrue(Ctx))});
  Term->eraseFromParent();
  ExitingVPBB->appendRecipe(BOC);
  Plan.setVF(BestVF);
  Plan.setUF(BestUF);
  // TODO: Further simplifications are possible
  //      1. Replace inductions with constants.
  //      2. Replace vector loop region with VPBasicBlock.
}

#ifndef NDEBUG
static VPRegionBlock *GetReplicateRegion(VPRecipeBase *R) {
  auto *Region = dyn_cast_or_null<VPRegionBlock>(R->getParent()->getParent());
  if (Region && Region->isReplicator()) {
    assert(Region->getNumSuccessors() == 1 &&
           Region->getNumPredecessors() == 1 && "Expected SESE region!");
    assert(R->getParent()->size() == 1 &&
           "A recipe in an original replicator region must be the only "
           "recipe in its block");
    return Region;
  }
  return nullptr;
}
#endif

static bool properlyDominates(const VPRecipeBase *A, const VPRecipeBase *B,
                              VPDominatorTree &VPDT) {
  if (A == B)
    return false;

  auto LocalComesBefore = [](const VPRecipeBase *A, const VPRecipeBase *B) {
    for (auto &R : *A->getParent()) {
      if (&R == A)
        return true;
      if (&R == B)
        return false;
    }
    llvm_unreachable("recipe not found");
  };
  const VPBlockBase *ParentA = A->getParent();
  const VPBlockBase *ParentB = B->getParent();
  if (ParentA == ParentB)
    return LocalComesBefore(A, B);

  assert(!GetReplicateRegion(const_cast<VPRecipeBase *>(A)) &&
         "No replicate regions expected at this point");
  assert(!GetReplicateRegion(const_cast<VPRecipeBase *>(B)) &&
         "No replicate regions expected at this point");
  return VPDT.properlyDominates(ParentA, ParentB);
}

/// Sink users of \p FOR after the recipe defining the previous value \p
/// Previous of the recurrence. \returns true if all users of \p FOR could be
/// re-arranged as needed or false if it is not possible.
static bool
sinkRecurrenceUsersAfterPrevious(VPFirstOrderRecurrencePHIRecipe *FOR,
                                 VPRecipeBase *Previous,
                                 VPDominatorTree &VPDT) {
  // Collect recipes that need sinking.
  SmallVector<VPRecipeBase *> WorkList;
  SmallPtrSet<VPRecipeBase *, 8> Seen;
  Seen.insert(Previous);
  auto TryToPushSinkCandidate = [&](VPRecipeBase *SinkCandidate) {
    // The previous value must not depend on the users of the recurrence phi. In
    // that case, FOR is not a fixed order recurrence.
    if (SinkCandidate == Previous)
      return false;

    if (isa<VPHeaderPHIRecipe>(SinkCandidate) ||
        !Seen.insert(SinkCandidate).second ||
        properlyDominates(Previous, SinkCandidate, VPDT))
      return true;

    if (SinkCandidate->mayHaveSideEffects())
      return false;

    WorkList.push_back(SinkCandidate);
    return true;
  };

  // Recursively sink users of FOR after Previous.
  WorkList.push_back(FOR);
  for (unsigned I = 0; I != WorkList.size(); ++I) {
    VPRecipeBase *Current = WorkList[I];
    assert(Current->getNumDefinedValues() == 1 &&
           "only recipes with a single defined value expected");

    for (VPUser *User : Current->getVPSingleValue()->users()) {
      if (auto *R = dyn_cast<VPRecipeBase>(User))
        if (!TryToPushSinkCandidate(R))
          return false;
    }
  }

  // Keep recipes to sink ordered by dominance so earlier instructions are
  // processed first.
  sort(WorkList, [&VPDT](const VPRecipeBase *A, const VPRecipeBase *B) {
    return properlyDominates(A, B, VPDT);
  });

  for (VPRecipeBase *SinkCandidate : WorkList) {
    if (SinkCandidate == FOR)
      continue;

    SinkCandidate->moveAfter(Previous);
    Previous = SinkCandidate;
  }
  return true;
}

bool VPlanTransforms::adjustFixedOrderRecurrences(VPlan &Plan,
                                                  VPBuilder &Builder) {
  VPDominatorTree VPDT;
  VPDT.recalculate(Plan);

  SmallVector<VPFirstOrderRecurrencePHIRecipe *> RecurrencePhis;
  for (VPRecipeBase &R :
       Plan.getVectorLoopRegion()->getEntry()->getEntryBasicBlock()->phis())
    if (auto *FOR = dyn_cast<VPFirstOrderRecurrencePHIRecipe>(&R))
      RecurrencePhis.push_back(FOR);

  for (VPFirstOrderRecurrencePHIRecipe *FOR : RecurrencePhis) {
    SmallPtrSet<VPFirstOrderRecurrencePHIRecipe *, 4> SeenPhis;
    VPRecipeBase *Previous = FOR->getBackedgeValue()->getDefiningRecipe();
    // Fixed-order recurrences do not contain cycles, so this loop is guaranteed
    // to terminate.
    while (auto *PrevPhi =
               dyn_cast_or_null<VPFirstOrderRecurrencePHIRecipe>(Previous)) {
      assert(PrevPhi->getParent() == FOR->getParent());
      assert(SeenPhis.insert(PrevPhi).second);
      Previous = PrevPhi->getBackedgeValue()->getDefiningRecipe();
    }

    if (!sinkRecurrenceUsersAfterPrevious(FOR, Previous, VPDT))
      return false;

    // Introduce a recipe to combine the incoming and previous values of a
    // fixed-order recurrence.
    VPBasicBlock *InsertBlock = Previous->getParent();
    if (isa<VPHeaderPHIRecipe>(Previous))
      Builder.setInsertPoint(InsertBlock, InsertBlock->getFirstNonPhi());
    else
      Builder.setInsertPoint(InsertBlock, std::next(Previous->getIterator()));

    auto *RecurSplice = cast<VPInstruction>(
        Builder.createNaryOp(VPInstruction::FirstOrderRecurrenceSplice,
                             {FOR, FOR->getBackedgeValue()}));

    FOR->replaceAllUsesWith(RecurSplice);
    // Set the first operand of RecurSplice to FOR again, after replacing
    // all users.
    RecurSplice->setOperand(0, FOR);
  }
  return true;
}

void VPlanTransforms::clearReductionWrapFlags(VPlan &Plan) {
  for (VPRecipeBase &R :
       Plan.getVectorLoopRegion()->getEntryBasicBlock()->phis()) {
    auto *PhiR = dyn_cast<VPReductionPHIRecipe>(&R);
    if (!PhiR)
      continue;
    const RecurrenceDescriptor &RdxDesc = PhiR->getRecurrenceDescriptor();
    RecurKind RK = RdxDesc.getRecurrenceKind();
    if (RK != RecurKind::Add && RK != RecurKind::Mul)
      continue;

    SmallSetVector<VPValue *, 8> Worklist;
    Worklist.insert(PhiR);

    for (unsigned I = 0; I != Worklist.size(); ++I) {
      VPValue *Cur = Worklist[I];
      if (auto *RecWithFlags =
              dyn_cast<VPRecipeWithIRFlags>(Cur->getDefiningRecipe())) {
        RecWithFlags->dropPoisonGeneratingFlags();
      }

      for (VPUser *U : Cur->users()) {
        auto *UserRecipe = dyn_cast<VPRecipeBase>(U);
        if (!UserRecipe)
          continue;
        for (VPValue *V : UserRecipe->definedValues())
          Worklist.insert(V);
      }
    }
  }
}
