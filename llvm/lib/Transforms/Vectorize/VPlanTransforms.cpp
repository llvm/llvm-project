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
#include "VPRecipeBuilder.h"
#include "VPlan.h"
#include "VPlanAnalysis.h"
#include "VPlanCFG.h"
#include "VPlanDominatorTree.h"
#include "VPlanHelpers.h"
#include "VPlanPatternMatch.h"
#include "VPlanUtils.h"
#include "VPlanVerifier.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetFolder.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/TypeSize.h"

using namespace llvm;

bool VPlanTransforms::tryToConvertVPInstructionsToVPRecipes(
    VPlanPtr &Plan,
    function_ref<const InductionDescriptor *(PHINode *)>
        GetIntOrFpInductionDescriptor,
    ScalarEvolution &SE, const TargetLibraryInfo &TLI) {

  ReversePostOrderTraversal<VPBlockDeepTraversalWrapper<VPBlockBase *>> RPOT(
      Plan->getVectorLoopRegion());
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(RPOT)) {
    // Skip blocks outside region
    if (!VPBB->getParent())
      break;
    VPRecipeBase *Term = VPBB->getTerminator();
    auto EndIter = Term ? Term->getIterator() : VPBB->end();
    // Introduce each ingredient into VPlan.
    for (VPRecipeBase &Ingredient :
         make_early_inc_range(make_range(VPBB->begin(), EndIter))) {

      VPValue *VPV = Ingredient.getVPSingleValue();
      if (!VPV->getUnderlyingValue())
        continue;

      Instruction *Inst = cast<Instruction>(VPV->getUnderlyingValue());

      VPRecipeBase *NewRecipe = nullptr;
      if (auto *VPPhi = dyn_cast<VPWidenPHIRecipe>(&Ingredient)) {
        auto *Phi = cast<PHINode>(VPPhi->getUnderlyingValue());
        const auto *II = GetIntOrFpInductionDescriptor(Phi);
        if (!II)
          continue;

        VPValue *Start = Plan->getOrAddLiveIn(II->getStartValue());
        VPValue *Step =
            vputils::getOrCreateVPValueForSCEVExpr(*Plan, II->getStep(), SE);
        NewRecipe = new VPWidenIntOrFpInductionRecipe(
            Phi, Start, Step, &Plan->getVF(), *II, Ingredient.getDebugLoc());
      } else {
        assert(isa<VPInstruction>(&Ingredient) &&
               "only VPInstructions expected here");
        assert(!isa<PHINode>(Inst) && "phis should be handled above");
        // Create VPWidenMemoryRecipe for loads and stores.
        if (LoadInst *Load = dyn_cast<LoadInst>(Inst)) {
          NewRecipe = new VPWidenLoadRecipe(
              *Load, Ingredient.getOperand(0), nullptr /*Mask*/,
              false /*Consecutive*/, false /*Reverse*/, VPIRMetadata(*Load),
              Ingredient.getDebugLoc());
        } else if (StoreInst *Store = dyn_cast<StoreInst>(Inst)) {
          NewRecipe = new VPWidenStoreRecipe(
              *Store, Ingredient.getOperand(1), Ingredient.getOperand(0),
              nullptr /*Mask*/, false /*Consecutive*/, false /*Reverse*/,
              VPIRMetadata(*Store), Ingredient.getDebugLoc());
        } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Inst)) {
          NewRecipe = new VPWidenGEPRecipe(GEP, Ingredient.operands());
        } else if (CallInst *CI = dyn_cast<CallInst>(Inst)) {
          Intrinsic::ID VectorID = getVectorIntrinsicIDForCall(CI, &TLI);
          if (VectorID == Intrinsic::not_intrinsic)
            return false;
          NewRecipe = new VPWidenIntrinsicRecipe(
              *CI, getVectorIntrinsicIDForCall(CI, &TLI),
              {Ingredient.op_begin(), Ingredient.op_end() - 1}, CI->getType(),
              CI->getDebugLoc());
        } else if (SelectInst *SI = dyn_cast<SelectInst>(Inst)) {
          NewRecipe = new VPWidenSelectRecipe(*SI, Ingredient.operands());
        } else if (auto *CI = dyn_cast<CastInst>(Inst)) {
          NewRecipe = new VPWidenCastRecipe(
              CI->getOpcode(), Ingredient.getOperand(0), CI->getType(), *CI);
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
  return true;
}

static bool sinkScalarOperands(VPlan &Plan) {
  auto Iter = vp_depth_first_deep(Plan.getEntry());
  bool Changed = false;
  // First, collect the operands of all recipes in replicate blocks as seeds for
  // sinking.
  SetVector<std::pair<VPBasicBlock *, VPSingleDefRecipe *>> WorkList;
  for (VPRegionBlock *VPR : VPBlockUtils::blocksOnly<VPRegionBlock>(Iter)) {
    VPBasicBlock *EntryVPBB = VPR->getEntryBasicBlock();
    if (!VPR->isReplicator() || EntryVPBB->getSuccessors().size() != 2)
      continue;
    VPBasicBlock *VPBB = dyn_cast<VPBasicBlock>(EntryVPBB->getSuccessors()[0]);
    if (!VPBB || VPBB->getSingleSuccessor() != VPR->getExitingBasicBlock())
      continue;
    for (auto &Recipe : *VPBB) {
      for (VPValue *Op : Recipe.operands())
        if (auto *Def =
                dyn_cast_or_null<VPSingleDefRecipe>(Op->getDefiningRecipe()))
          WorkList.insert(std::make_pair(VPBB, Def));
    }
  }

  bool ScalarVFOnly = Plan.hasScalarVFOnly();
  // Try to sink each replicate or scalar IV steps recipe in the worklist.
  for (unsigned I = 0; I != WorkList.size(); ++I) {
    VPBasicBlock *SinkTo;
    VPSingleDefRecipe *SinkCandidate;
    std::tie(SinkTo, SinkCandidate) = WorkList[I];
    if (SinkCandidate->getParent() == SinkTo ||
        SinkCandidate->mayHaveSideEffects() ||
        SinkCandidate->mayReadOrWriteMemory())
      continue;
    if (auto *RepR = dyn_cast<VPReplicateRecipe>(SinkCandidate)) {
      if (!ScalarVFOnly && RepR->isSingleScalar())
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
      auto *UI = cast<VPRecipeBase>(U);
      if (UI->getParent() == SinkTo)
        return true;
      NeedsDuplicating = UI->onlyFirstLaneUsed(SinkCandidate);
      // We only know how to duplicate VPReplicateRecipes and
      // VPScalarIVStepsRecipes for now.
      return NeedsDuplicating &&
             isa<VPReplicateRecipe, VPScalarIVStepsRecipe>(SinkCandidate);
    };
    if (!all_of(SinkCandidate->users(), CanSinkWithUser))
      continue;

    if (NeedsDuplicating) {
      if (ScalarVFOnly)
        continue;
      VPSingleDefRecipe *Clone;
      if (auto *SinkCandidateRepR =
              dyn_cast<VPReplicateRecipe>(SinkCandidate)) {
        // TODO: Handle converting to uniform recipes as separate transform,
        // then cloning should be sufficient here.
        Instruction *I = SinkCandidate->getUnderlyingInstr();
        Clone = new VPReplicateRecipe(I, SinkCandidate->operands(), true,
                                      nullptr /*Mask*/, *SinkCandidateRepR);
        // TODO: add ".cloned" suffix to name of Clone's VPValue.
      } else {
        Clone = SinkCandidate->clone();
      }

      Clone->insertBefore(SinkCandidate);
      SinkCandidate->replaceUsesWithIf(Clone, [SinkTo](VPUser &U, unsigned) {
        return cast<VPRecipeBase>(&U)->getParent() != SinkTo;
      });
    }
    SinkCandidate->moveBefore(*SinkTo, SinkTo->getFirstNonPhi());
    for (VPValue *Op : SinkCandidate->operands())
      if (auto *Def =
              dyn_cast_or_null<VPSingleDefRecipe>(Op->getDefiningRecipe()))
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
  SmallPtrSet<VPRegionBlock *, 4> TransformedRegions;

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
    if (TransformedRegions.contains(Region1))
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
      Phi1ToMoveV->replaceUsesWithIf(PredInst1, [Then2](VPUser &U, unsigned) {
        return cast<VPRecipeBase>(&U)->getParent() == Then2;
      });

      // Remove phi recipes that are unused after merging the regions.
      if (Phi1ToMove.getVPSingleValue()->getNumUsers() == 0) {
        Phi1ToMove.eraseFromParent();
        continue;
      }
      Phi1ToMove.moveBefore(*Merge2, Merge2->begin());
    }

    // Remove the dead recipes in Region1's entry block.
    for (VPRecipeBase &R :
         make_early_inc_range(reverse(*Region1->getEntryBasicBlock())))
      R.eraseFromParent();

    // Finally, remove the first region.
    for (VPBlockBase *Pred : make_early_inc_range(Region1->getPredecessors())) {
      VPBlockUtils::disconnectBlocks(Pred, Region1);
      VPBlockUtils::connectBlocks(Pred, MiddleBasicBlock);
    }
    VPBlockUtils::disconnectBlocks(Region1, MiddleBasicBlock);
    TransformedRegions.insert(Region1);
  }

  return !TransformedRegions.empty();
}

static VPRegionBlock *createReplicateRegion(VPReplicateRecipe *PredRecipe,
                                            VPlan &Plan) {
  Instruction *Instr = PredRecipe->getUnderlyingInstr();
  // Build the triangular if-then region.
  std::string RegionName = (Twine("pred.") + Instr->getOpcodeName()).str();
  assert(Instr->getParent() && "Predicated instruction not in any basic block");
  auto *BlockInMask = PredRecipe->getMask();
  auto *MaskDef = BlockInMask->getDefiningRecipe();
  auto *BOMRecipe = new VPBranchOnMaskRecipe(
      BlockInMask, MaskDef ? MaskDef->getDebugLoc() : DebugLoc());
  auto *Entry =
      Plan.createVPBasicBlock(Twine(RegionName) + ".entry", BOMRecipe);

  // Replace predicated replicate recipe with a replicate recipe without a
  // mask but in the replicate region.
  auto *RecipeWithoutMask = new VPReplicateRecipe(
      PredRecipe->getUnderlyingInstr(),
      make_range(PredRecipe->op_begin(), std::prev(PredRecipe->op_end())),
      PredRecipe->isSingleScalar(), nullptr /*Mask*/, *PredRecipe);
  auto *Pred =
      Plan.createVPBasicBlock(Twine(RegionName) + ".if", RecipeWithoutMask);

  VPPredInstPHIRecipe *PHIRecipe = nullptr;
  if (PredRecipe->getNumUsers() != 0) {
    PHIRecipe = new VPPredInstPHIRecipe(RecipeWithoutMask,
                                        RecipeWithoutMask->getDebugLoc());
    PredRecipe->replaceAllUsesWith(PHIRecipe);
    PHIRecipe->setOperand(0, RecipeWithoutMask);
  }
  PredRecipe->eraseFromParent();
  auto *Exiting =
      Plan.createVPBasicBlock(Twine(RegionName) + ".continue", PHIRecipe);
  VPRegionBlock *Region =
      Plan.createVPRegionBlock(Entry, Exiting, RegionName, true);

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
    VPRegionBlock *Region = createReplicateRegion(RepR, Plan);
    Region->setParent(CurrentBlock->getParent());
    VPBlockUtils::insertOnEdge(CurrentBlock, SplitBlock, Region);

    VPRegionBlock *ParentRegion = Region->getParent();
    if (ParentRegion && ParentRegion->getExiting() == CurrentBlock)
      ParentRegion->setExiting(SplitBlock);
  }
}

/// Remove redundant VPBasicBlocks by merging them into their predecessor if
/// the predecessor has a single successor.
static bool mergeBlocksIntoPredecessors(VPlan &Plan) {
  SmallVector<VPBasicBlock *> WorkList;
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(
           vp_depth_first_deep(Plan.getEntry()))) {
    // Don't fold the blocks in the skeleton of the Plan into their single
    // predecessors for now.
    // TODO: Remove restriction once more of the skeleton is modeled in VPlan.
    if (!VPBB->getParent())
      continue;
    auto *PredVPBB =
        dyn_cast_or_null<VPBasicBlock>(VPBB->getSinglePredecessor());
    if (!PredVPBB || PredVPBB->getNumSuccessors() != 1 ||
        isa<VPIRBasicBlock>(PredVPBB))
      continue;
    WorkList.push_back(VPBB);
  }

  for (VPBasicBlock *VPBB : WorkList) {
    VPBasicBlock *PredVPBB = cast<VPBasicBlock>(VPBB->getSinglePredecessor());
    for (VPRecipeBase &R : make_early_inc_range(*VPBB))
      R.moveBefore(*PredVPBB, PredVPBB->end());
    VPBlockUtils::disconnectBlocks(PredVPBB, VPBB);
    auto *ParentRegion = VPBB->getParent();
    if (ParentRegion && ParentRegion->getExiting() == VPBB)
      ParentRegion->setExiting(PredVPBB);
    for (auto *Succ : to_vector(VPBB->successors())) {
      VPBlockUtils::disconnectBlocks(VPBB, Succ);
      VPBlockUtils::connectBlocks(PredVPBB, Succ);
    }
    // VPBB is now dead and will be cleaned up when the plan gets destroyed.
  }
  return !WorkList.empty();
}

void VPlanTransforms::createAndOptimizeReplicateRegions(VPlan &Plan) {
  // Convert masked VPReplicateRecipes to if-then region blocks.
  addReplicateRegions(Plan);

  bool ShouldSimplify = true;
  while (ShouldSimplify) {
    ShouldSimplify = sinkScalarOperands(Plan);
    ShouldSimplify |= mergeReplicateRegionsIntoSuccessors(Plan);
    ShouldSimplify |= mergeBlocksIntoPredecessors(Plan);
  }
}

/// Remove redundant casts of inductions.
///
/// Such redundant casts are casts of induction variables that can be ignored,
/// because we already proved that the casted phi is equal to the uncasted phi
/// in the vectorized loop. There is no need to vectorize the cast - the same
/// value can be used for both the phi and casts in the vector loop.
static void removeRedundantInductionCasts(VPlan &Plan) {
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
      VPSingleDefRecipe *FoundUserCast = nullptr;
      for (auto *U : FindMyCast->users()) {
        auto *UserCast = dyn_cast<VPSingleDefRecipe>(U);
        if (UserCast && UserCast->getUnderlyingValue() == IRCast) {
          FoundUserCast = UserCast;
          break;
        }
      }
      FindMyCast = FoundUserCast;
    }
    FindMyCast->replaceAllUsesWith(IV);
  }
}

/// Try to replace VPWidenCanonicalIVRecipes with a widened canonical IV
/// recipe, if it exists.
static void removeRedundantCanonicalIVs(VPlan &Plan) {
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

    if (!WidenOriginalIV || !WidenOriginalIV->isCanonical())
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

/// Returns true if \p R is dead and can be removed.
static bool isDeadRecipe(VPRecipeBase &R) {
  using namespace llvm::PatternMatch;
  // Do remove conditional assume instructions as their conditions may be
  // flattened.
  auto *RepR = dyn_cast<VPReplicateRecipe>(&R);
  bool IsConditionalAssume =
      RepR && RepR->isPredicated() &&
      match(RepR->getUnderlyingInstr(), m_Intrinsic<Intrinsic::assume>());
  if (IsConditionalAssume)
    return true;

  if (R.mayHaveSideEffects())
    return false;

  // Recipe is dead if no user keeps the recipe alive.
  return all_of(R.definedValues(),
                [](VPValue *V) { return V->getNumUsers() == 0; });
}

void VPlanTransforms::removeDeadRecipes(VPlan &Plan) {
  ReversePostOrderTraversal<VPBlockDeepTraversalWrapper<VPBlockBase *>> RPOT(
      Plan.getEntry());

  for (VPBasicBlock *VPBB : reverse(VPBlockUtils::blocksOnly<VPBasicBlock>(RPOT))) {
    // The recipes in the block are processed in reverse order, to catch chains
    // of dead recipes.
    for (VPRecipeBase &R : make_early_inc_range(reverse(*VPBB))) {
      if (isDeadRecipe(R))
        R.eraseFromParent();
    }
  }
}

static VPScalarIVStepsRecipe *
createScalarIVSteps(VPlan &Plan, InductionDescriptor::InductionKind Kind,
                    Instruction::BinaryOps InductionOpcode,
                    FPMathOperator *FPBinOp, Instruction *TruncI,
                    VPValue *StartV, VPValue *Step, DebugLoc DL,
                    VPBuilder &Builder) {
  VPBasicBlock *HeaderVPBB = Plan.getVectorLoopRegion()->getEntryBasicBlock();
  VPCanonicalIVPHIRecipe *CanonicalIV = Plan.getCanonicalIV();
  VPSingleDefRecipe *BaseIV = Builder.createDerivedIV(
      Kind, FPBinOp, StartV, CanonicalIV, Step, "offset.idx");

  // Truncate base induction if needed.
  Type *CanonicalIVType = CanonicalIV->getScalarType();
  VPTypeAnalysis TypeInfo(CanonicalIVType);
  Type *ResultTy = TypeInfo.inferScalarType(BaseIV);
  if (TruncI) {
    Type *TruncTy = TruncI->getType();
    assert(ResultTy->getScalarSizeInBits() > TruncTy->getScalarSizeInBits() &&
           "Not truncating.");
    assert(ResultTy->isIntegerTy() && "Truncation requires an integer type");
    BaseIV = Builder.createScalarCast(Instruction::Trunc, BaseIV, TruncTy, DL);
    ResultTy = TruncTy;
  }

  // Truncate step if needed.
  Type *StepTy = TypeInfo.inferScalarType(Step);
  if (ResultTy != StepTy) {
    assert(StepTy->getScalarSizeInBits() > ResultTy->getScalarSizeInBits() &&
           "Not truncating.");
    assert(StepTy->isIntegerTy() && "Truncation requires an integer type");
    auto *VecPreheader =
        cast<VPBasicBlock>(HeaderVPBB->getSingleHierarchicalPredecessor());
    VPBuilder::InsertPointGuard Guard(Builder);
    Builder.setInsertPoint(VecPreheader);
    Step = Builder.createScalarCast(Instruction::Trunc, Step, ResultTy, DL);
  }
  return Builder.createScalarIVSteps(InductionOpcode, FPBinOp, BaseIV, Step,
                                     &Plan.getVF(), DL);
}

static SmallVector<VPUser *> collectUsersRecursively(VPValue *V) {
  SetVector<VPUser *> Users(llvm::from_range, V->users());
  for (unsigned I = 0; I != Users.size(); ++I) {
    VPRecipeBase *Cur = cast<VPRecipeBase>(Users[I]);
    if (isa<VPHeaderPHIRecipe>(Cur))
      continue;
    for (VPValue *V : Cur->definedValues())
      Users.insert_range(V->users());
  }
  return Users.takeVector();
}

/// Legalize VPWidenPointerInductionRecipe, by replacing it with a PtrAdd
/// (IndStart, ScalarIVSteps (0, Step)) if only its scalar values are used, as
/// VPWidenPointerInductionRecipe will generate vectors only. If some users
/// require vectors while other require scalars, the scalar uses need to extract
/// the scalars from the generated vectors (Note that this is different to how
/// int/fp inductions are handled). Legalize extract-from-ends using uniform
/// VPReplicateRecipe of wide inductions to use regular VPReplicateRecipe, so
/// the correct end value is available. Also optimize
/// VPWidenIntOrFpInductionRecipe, if any of its users needs scalar values, by
/// providing them scalar steps built on the canonical scalar IV and update the
/// original IV's users. This is an optional optimization to reduce the needs of
/// vector extracts.
static void legalizeAndOptimizeInductions(VPlan &Plan) {
  using namespace llvm::VPlanPatternMatch;
  VPBasicBlock *HeaderVPBB = Plan.getVectorLoopRegion()->getEntryBasicBlock();
  bool HasOnlyVectorVFs = !Plan.hasScalarVFOnly();
  VPBuilder Builder(HeaderVPBB, HeaderVPBB->getFirstNonPhi());
  for (VPRecipeBase &Phi : HeaderVPBB->phis()) {
    auto *PhiR = dyn_cast<VPWidenInductionRecipe>(&Phi);
    if (!PhiR)
      continue;

    // Try to narrow wide and replicating recipes to uniform recipes, based on
    // VPlan analysis.
    // TODO: Apply to all recipes in the future, to replace legacy uniformity
    // analysis.
    auto Users = collectUsersRecursively(PhiR);
    for (VPUser *U : reverse(Users)) {
      auto *Def = dyn_cast<VPSingleDefRecipe>(U);
      auto *RepR = dyn_cast<VPReplicateRecipe>(U);
      // Skip recipes that shouldn't be narrowed.
      if (!Def || !isa<VPReplicateRecipe, VPWidenRecipe>(Def) ||
          Def->getNumUsers() == 0 || !Def->getUnderlyingValue() ||
          (RepR && (RepR->isSingleScalar() || RepR->isPredicated())))
        continue;

      // Skip recipes that may have other lanes than their first used.
      if (!vputils::isSingleScalar(Def) && !vputils::onlyFirstLaneUsed(Def))
        continue;

      auto *Clone = new VPReplicateRecipe(Def->getUnderlyingInstr(),
                                          Def->operands(), /*IsUniform*/ true);
      Clone->insertAfter(Def);
      Def->replaceAllUsesWith(Clone);
    }

    // Replace wide pointer inductions which have only their scalars used by
    // PtrAdd(IndStart, ScalarIVSteps (0, Step)).
    if (auto *PtrIV = dyn_cast<VPWidenPointerInductionRecipe>(&Phi)) {
      if (!PtrIV->onlyScalarsGenerated(Plan.hasScalableVF()))
        continue;

      const InductionDescriptor &ID = PtrIV->getInductionDescriptor();
      VPValue *StartV =
          Plan.getOrAddLiveIn(ConstantInt::get(ID.getStep()->getType(), 0));
      VPValue *StepV = PtrIV->getOperand(1);
      VPScalarIVStepsRecipe *Steps = createScalarIVSteps(
          Plan, InductionDescriptor::IK_IntInduction, Instruction::Add, nullptr,
          nullptr, StartV, StepV, PtrIV->getDebugLoc(), Builder);

      VPValue *PtrAdd = Builder.createPtrAdd(PtrIV->getStartValue(), Steps,
                                             PtrIV->getDebugLoc(), "next.gep");

      PtrIV->replaceAllUsesWith(PtrAdd);
      continue;
    }

    // Replace widened induction with scalar steps for users that only use
    // scalars.
    auto *WideIV = cast<VPWidenIntOrFpInductionRecipe>(&Phi);
    if (HasOnlyVectorVFs && none_of(WideIV->users(), [WideIV](VPUser *U) {
          return U->usesScalars(WideIV);
        }))
      continue;

    const InductionDescriptor &ID = WideIV->getInductionDescriptor();
    VPScalarIVStepsRecipe *Steps = createScalarIVSteps(
        Plan, ID.getKind(), ID.getInductionOpcode(),
        dyn_cast_or_null<FPMathOperator>(ID.getInductionBinOp()),
        WideIV->getTruncInst(), WideIV->getStartValue(), WideIV->getStepValue(),
        WideIV->getDebugLoc(), Builder);

    // Update scalar users of IV to use Step instead.
    if (!HasOnlyVectorVFs)
      WideIV->replaceAllUsesWith(Steps);
    else
      WideIV->replaceUsesWithIf(Steps, [WideIV](VPUser &U, unsigned) {
        return U.usesScalars(WideIV);
      });
  }
}

/// Check if \p VPV is an untruncated wide induction, either before or after the
/// increment. If so return the header IV (before the increment), otherwise
/// return null.
static VPWidenInductionRecipe *getOptimizableIVOf(VPValue *VPV) {
  auto *WideIV = dyn_cast<VPWidenInductionRecipe>(VPV);
  if (WideIV) {
    // VPV itself is a wide induction, separately compute the end value for exit
    // users if it is not a truncated IV.
    auto *IntOrFpIV = dyn_cast<VPWidenIntOrFpInductionRecipe>(WideIV);
    return (IntOrFpIV && IntOrFpIV->getTruncInst()) ? nullptr : WideIV;
  }

  // Check if VPV is an optimizable induction increment.
  VPRecipeBase *Def = VPV->getDefiningRecipe();
  if (!Def || Def->getNumOperands() != 2)
    return nullptr;
  WideIV = dyn_cast<VPWidenInductionRecipe>(Def->getOperand(0));
  if (!WideIV)
    WideIV = dyn_cast<VPWidenInductionRecipe>(Def->getOperand(1));
  if (!WideIV)
    return nullptr;

  auto IsWideIVInc = [&]() {
    using namespace VPlanPatternMatch;
    auto &ID = WideIV->getInductionDescriptor();

    // Check if VPV increments the induction by the induction step.
    VPValue *IVStep = WideIV->getStepValue();
    switch (ID.getInductionOpcode()) {
    case Instruction::Add:
      return match(VPV, m_c_Binary<Instruction::Add>(m_Specific(WideIV),
                                                     m_Specific(IVStep)));
    case Instruction::FAdd:
      return match(VPV, m_c_Binary<Instruction::FAdd>(m_Specific(WideIV),
                                                      m_Specific(IVStep)));
    case Instruction::FSub:
      return match(VPV, m_Binary<Instruction::FSub>(m_Specific(WideIV),
                                                    m_Specific(IVStep)));
    case Instruction::Sub: {
      // IVStep will be the negated step of the subtraction. Check if Step == -1
      // * IVStep.
      VPValue *Step;
      if (!match(VPV,
                 m_Binary<Instruction::Sub>(m_VPValue(), m_VPValue(Step))) ||
          !Step->isLiveIn() || !IVStep->isLiveIn())
        return false;
      auto *StepCI = dyn_cast<ConstantInt>(Step->getLiveInIRValue());
      auto *IVStepCI = dyn_cast<ConstantInt>(IVStep->getLiveInIRValue());
      return StepCI && IVStepCI &&
             StepCI->getValue() == (-1 * IVStepCI->getValue());
    }
    default:
      return ID.getKind() == InductionDescriptor::IK_PtrInduction &&
             match(VPV, m_GetElementPtr(m_Specific(WideIV),
                                        m_Specific(WideIV->getStepValue())));
    }
    llvm_unreachable("should have been covered by switch above");
  };
  return IsWideIVInc() ? WideIV : nullptr;
}

/// Attempts to optimize the induction variable exit values for users in the
/// early exit block.
static VPValue *optimizeEarlyExitInductionUser(VPlan &Plan,
                                               VPTypeAnalysis &TypeInfo,
                                               VPBlockBase *PredVPBB,
                                               VPValue *Op) {
  using namespace VPlanPatternMatch;

  VPValue *Incoming, *Mask;
  if (!match(Op, m_VPInstruction<Instruction::ExtractElement>(
                     m_VPValue(Incoming),
                     m_VPInstruction<VPInstruction::FirstActiveLane>(
                         m_VPValue(Mask)))))
    return nullptr;

  auto *WideIV = getOptimizableIVOf(Incoming);
  if (!WideIV)
    return nullptr;

  auto *WideIntOrFp = dyn_cast<VPWidenIntOrFpInductionRecipe>(WideIV);
  if (WideIntOrFp && WideIntOrFp->getTruncInst())
    return nullptr;

  // Calculate the final index.
  VPValue *EndValue = Plan.getCanonicalIV();
  auto CanonicalIVType = Plan.getCanonicalIV()->getScalarType();
  VPBuilder B(cast<VPBasicBlock>(PredVPBB));

  DebugLoc DL = cast<VPInstruction>(Op)->getDebugLoc();
  VPValue *FirstActiveLane =
      B.createNaryOp(VPInstruction::FirstActiveLane, Mask, DL);
  Type *FirstActiveLaneType = TypeInfo.inferScalarType(FirstActiveLane);
  if (CanonicalIVType != FirstActiveLaneType) {
    Instruction::CastOps CastOp =
        CanonicalIVType->getScalarSizeInBits() <
                FirstActiveLaneType->getScalarSizeInBits()
            ? Instruction::Trunc
            : Instruction::ZExt;
    FirstActiveLane =
        B.createScalarCast(CastOp, FirstActiveLane, CanonicalIVType, DL);
  }
  EndValue = B.createNaryOp(Instruction::Add, {EndValue, FirstActiveLane}, DL);

  // `getOptimizableIVOf()` always returns the pre-incremented IV, so if it
  // changed it means the exit is using the incremented value, so we need to
  // add the step.
  if (Incoming != WideIV) {
    VPValue *One = Plan.getOrAddLiveIn(ConstantInt::get(CanonicalIVType, 1));
    EndValue = B.createNaryOp(Instruction::Add, {EndValue, One}, DL);
  }

  if (!WideIntOrFp || !WideIntOrFp->isCanonical()) {
    const InductionDescriptor &ID = WideIV->getInductionDescriptor();
    VPValue *Start = WideIV->getStartValue();
    VPValue *Step = WideIV->getStepValue();
    EndValue = B.createDerivedIV(
        ID.getKind(), dyn_cast_or_null<FPMathOperator>(ID.getInductionBinOp()),
        Start, EndValue, Step);
  }

  return EndValue;
}

/// Attempts to optimize the induction variable exit values for users in the
/// exit block coming from the latch in the original scalar loop.
static VPValue *
optimizeLatchExitInductionUser(VPlan &Plan, VPTypeAnalysis &TypeInfo,
                               VPBlockBase *PredVPBB, VPValue *Op,
                               DenseMap<VPValue *, VPValue *> &EndValues) {
  using namespace VPlanPatternMatch;

  VPValue *Incoming;
  if (!match(Op, m_VPInstruction<VPInstruction::ExtractLastElement>(
                     m_VPValue(Incoming))))
    return nullptr;

  auto *WideIV = getOptimizableIVOf(Incoming);
  if (!WideIV)
    return nullptr;

  VPValue *EndValue = EndValues.lookup(WideIV);
  assert(EndValue && "end value must have been pre-computed");

  // `getOptimizableIVOf()` always returns the pre-incremented IV, so if it
  // changed it means the exit is using the incremented value, so we don't
  // need to subtract the step.
  if (Incoming != WideIV)
    return EndValue;

  // Otherwise, subtract the step from the EndValue.
  VPBuilder B(cast<VPBasicBlock>(PredVPBB)->getTerminator());
  VPValue *Step = WideIV->getStepValue();
  Type *ScalarTy = TypeInfo.inferScalarType(WideIV);
  if (ScalarTy->isIntegerTy())
    return B.createNaryOp(Instruction::Sub, {EndValue, Step}, {}, "ind.escape");
  if (ScalarTy->isPointerTy()) {
    auto *Zero = Plan.getOrAddLiveIn(
        ConstantInt::get(Step->getLiveInIRValue()->getType(), 0));
    return B.createPtrAdd(EndValue,
                          B.createNaryOp(Instruction::Sub, {Zero, Step}), {},
                          "ind.escape");
  }
  if (ScalarTy->isFloatingPointTy()) {
    const auto &ID = WideIV->getInductionDescriptor();
    return B.createNaryOp(
        ID.getInductionBinOp()->getOpcode() == Instruction::FAdd
            ? Instruction::FSub
            : Instruction::FAdd,
        {EndValue, Step}, {ID.getInductionBinOp()->getFastMathFlags()});
  }
  llvm_unreachable("all possible induction types must be handled");
  return nullptr;
}

void VPlanTransforms::optimizeInductionExitUsers(
    VPlan &Plan, DenseMap<VPValue *, VPValue *> &EndValues) {
  VPBlockBase *MiddleVPBB = Plan.getMiddleBlock();
  VPTypeAnalysis TypeInfo(Plan.getCanonicalIV()->getScalarType());
  for (VPIRBasicBlock *ExitVPBB : Plan.getExitBlocks()) {
    for (VPRecipeBase &R : ExitVPBB->phis()) {
      auto *ExitIRI = cast<VPIRPhi>(&R);

      for (auto [Idx, PredVPBB] : enumerate(ExitVPBB->getPredecessors())) {
        VPValue *Escape = nullptr;
        if (PredVPBB == MiddleVPBB)
          Escape = optimizeLatchExitInductionUser(
              Plan, TypeInfo, PredVPBB, ExitIRI->getOperand(Idx), EndValues);
        else
          Escape = optimizeEarlyExitInductionUser(Plan, TypeInfo, PredVPBB,
                                                  ExitIRI->getOperand(Idx));
        if (Escape)
          ExitIRI->setOperand(Idx, Escape);
      }
    }
  }
}

/// Remove redundant EpxandSCEVRecipes in \p Plan's entry block by replacing
/// them with already existing recipes expanding the same SCEV expression.
static void removeRedundantExpandSCEVRecipes(VPlan &Plan) {
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

static void recursivelyDeleteDeadRecipes(VPValue *V) {
  SmallVector<VPValue *> WorkList;
  SmallPtrSet<VPValue *, 8> Seen;
  WorkList.push_back(V);

  while (!WorkList.empty()) {
    VPValue *Cur = WorkList.pop_back_val();
    if (!Seen.insert(Cur).second)
      continue;
    VPRecipeBase *R = Cur->getDefiningRecipe();
    if (!R)
      continue;
    if (!isDeadRecipe(*R))
      continue;
    WorkList.append(R->op_begin(), R->op_end());
    R->eraseFromParent();
  }
}

/// Try to fold \p R using TargetFolder to a constant. Will succeed and return a
/// non-nullptr Value for a handled \p Opcode if all \p Operands are constant.
static Value *tryToConstantFold(const VPRecipeBase &R, unsigned Opcode,
                                ArrayRef<VPValue *> Operands,
                                const DataLayout &DL,
                                VPTypeAnalysis &TypeInfo) {
  SmallVector<Value *, 4> Ops;
  for (VPValue *Op : Operands) {
    if (!Op->isLiveIn() || !Op->getLiveInIRValue())
      return nullptr;
    Ops.push_back(Op->getLiveInIRValue());
  }

  TargetFolder Folder(DL);
  if (Instruction::isBinaryOp(Opcode))
    return Folder.FoldBinOp(static_cast<Instruction::BinaryOps>(Opcode), Ops[0],
                            Ops[1]);
  if (Instruction::isCast(Opcode))
    return Folder.FoldCast(static_cast<Instruction::CastOps>(Opcode), Ops[0],
                           TypeInfo.inferScalarType(R.getVPSingleValue()));
  switch (Opcode) {
  case VPInstruction::LogicalAnd:
    return Folder.FoldSelect(Ops[0], Ops[1],
                             ConstantInt::getNullValue(Ops[1]->getType()));
  case VPInstruction::Not:
    return Folder.FoldBinOp(Instruction::BinaryOps::Xor, Ops[0],
                            Constant::getAllOnesValue(Ops[0]->getType()));
  case Instruction::Select:
    return Folder.FoldSelect(Ops[0], Ops[1], Ops[2]);
  case Instruction::ICmp:
  case Instruction::FCmp:
    return Folder.FoldCmp(cast<VPRecipeWithIRFlags>(R).getPredicate(), Ops[0],
                          Ops[1]);
  case Instruction::GetElementPtr: {
    auto &RFlags = cast<VPRecipeWithIRFlags>(R);
    auto *GEP = cast<GetElementPtrInst>(RFlags.getUnderlyingInstr());
    return Folder.FoldGEP(GEP->getSourceElementType(), Ops[0], drop_begin(Ops),
                          RFlags.getGEPNoWrapFlags());
  }
  case VPInstruction::PtrAdd:
    return Folder.FoldGEP(IntegerType::getInt8Ty(TypeInfo.getContext()), Ops[0],
                          Ops[1],
                          cast<VPRecipeWithIRFlags>(R).getGEPNoWrapFlags());
  case Instruction::InsertElement:
    return Folder.FoldInsertElement(Ops[0], Ops[1], Ops[2]);
  case Instruction::ExtractElement:
    return Folder.FoldExtractElement(Ops[0], Ops[1]);
  }
  return nullptr;
}

/// Try to simplify recipe \p R.
static void simplifyRecipe(VPRecipeBase &R, VPTypeAnalysis &TypeInfo) {
  using namespace llvm::VPlanPatternMatch;

  // Constant folding.
  if (TypeSwitch<VPRecipeBase *, bool>(&R)
          .Case<VPInstruction, VPWidenRecipe, VPWidenCastRecipe,
                VPReplicateRecipe>([&](auto *I) {
            VPlan *Plan = R.getParent()->getPlan();
            const DataLayout &DL =
                Plan->getScalarHeader()->getIRBasicBlock()->getDataLayout();
            Value *V = tryToConstantFold(*I, I->getOpcode(), I->operands(), DL,
                                         TypeInfo);
            if (V)
              I->replaceAllUsesWith(Plan->getOrAddLiveIn(V));
            return V;
          })
          .Default([](auto *) { return false; }))
    return;

  // Fold PredPHI constant -> constant.
  if (auto *PredPHI = dyn_cast<VPPredInstPHIRecipe>(&R)) {
    VPlan *Plan = R.getParent()->getPlan();
    VPValue *Op = PredPHI->getOperand(0);
    if (!Op->isLiveIn() || !Op->getLiveInIRValue())
      return;
    if (auto *C = dyn_cast<Constant>(Op->getLiveInIRValue()))
      PredPHI->replaceAllUsesWith(Plan->getOrAddLiveIn(C));
  }

  // VPScalarIVSteps can only be simplified after unrolling. VPScalarIVSteps for
  // part 0 can be replaced by their start value, if only the first lane is
  // demanded.
  if (auto *Steps = dyn_cast<VPScalarIVStepsRecipe>(&R)) {
    if (Steps->getParent()->getPlan()->isUnrolled() && Steps->isPart0() &&
        vputils::onlyFirstLaneUsed(Steps)) {
      Steps->replaceAllUsesWith(Steps->getOperand(0));
      return;
    }
  }

  VPValue *A;
  if (match(&R, m_Trunc(m_ZExtOrSExt(m_VPValue(A))))) {
    VPValue *Trunc = R.getVPSingleValue();
    Type *TruncTy = TypeInfo.inferScalarType(Trunc);
    Type *ATy = TypeInfo.inferScalarType(A);
    if (TruncTy == ATy) {
      Trunc->replaceAllUsesWith(A);
    } else {
      // Don't replace a scalarizing recipe with a widened cast.
      if (isa<VPReplicateRecipe>(&R))
        return;
      if (ATy->getScalarSizeInBits() < TruncTy->getScalarSizeInBits()) {

        unsigned ExtOpcode = match(R.getOperand(0), m_SExt(m_VPValue()))
                                 ? Instruction::SExt
                                 : Instruction::ZExt;
        auto *VPC =
            new VPWidenCastRecipe(Instruction::CastOps(ExtOpcode), A, TruncTy);
        if (auto *UnderlyingExt = R.getOperand(0)->getUnderlyingValue()) {
          // UnderlyingExt has distinct return type, used to retain legacy cost.
          VPC->setUnderlyingValue(UnderlyingExt);
        }
        VPC->insertBefore(&R);
        Trunc->replaceAllUsesWith(VPC);
      } else if (ATy->getScalarSizeInBits() > TruncTy->getScalarSizeInBits()) {
        auto *VPC = new VPWidenCastRecipe(Instruction::Trunc, A, TruncTy);
        VPC->insertBefore(&R);
        Trunc->replaceAllUsesWith(VPC);
      }
    }
#ifndef NDEBUG
    // Verify that the cached type info is for both A and its users is still
    // accurate by comparing it to freshly computed types.
    VPTypeAnalysis TypeInfo2(
        R.getParent()->getPlan()->getCanonicalIV()->getScalarType());
    assert(TypeInfo.inferScalarType(A) == TypeInfo2.inferScalarType(A));
    for (VPUser *U : A->users()) {
      auto *R = cast<VPRecipeBase>(U);
      for (VPValue *VPV : R->definedValues())
        assert(TypeInfo.inferScalarType(VPV) == TypeInfo2.inferScalarType(VPV));
    }
#endif
  }

  // Simplify (X && Y) || (X && !Y) -> X.
  // TODO: Split up into simpler, modular combines: (X && Y) || (X && Z) into X
  // && (Y || Z) and (X || !X) into true. This requires queuing newly created
  // recipes to be visited during simplification.
  VPValue *X, *Y;
  if (match(&R,
            m_c_BinaryOr(m_LogicalAnd(m_VPValue(X), m_VPValue(Y)),
                         m_LogicalAnd(m_Deferred(X), m_Not(m_Deferred(Y)))))) {
    R.getVPSingleValue()->replaceAllUsesWith(X);
    R.eraseFromParent();
    return;
  }

  // OR x, 1 -> 1.
  if (match(&R, m_c_BinaryOr(m_VPValue(X), m_AllOnes()))) {
    R.getVPSingleValue()->replaceAllUsesWith(
        R.getOperand(0) == X ? R.getOperand(1) : R.getOperand(0));
    R.eraseFromParent();
    return;
  }

  if (match(&R, m_Select(m_VPValue(), m_VPValue(X), m_Deferred(X))))
    return R.getVPSingleValue()->replaceAllUsesWith(X);

  if (match(&R, m_c_Mul(m_VPValue(A), m_SpecificInt(1))))
    return R.getVPSingleValue()->replaceAllUsesWith(A);

  if (match(&R, m_Not(m_VPValue(A)))) {
    if (match(A, m_Not(m_VPValue(A))))
      return R.getVPSingleValue()->replaceAllUsesWith(A);

    // Try to fold Not into compares by adjusting the predicate in-place.
    if (isa<VPWidenRecipe>(A) && A->getNumUsers() == 1) {
      auto *WideCmp = cast<VPWidenRecipe>(A);
      if (WideCmp->getOpcode() == Instruction::ICmp ||
          WideCmp->getOpcode() == Instruction::FCmp) {
        WideCmp->setPredicate(
            CmpInst::getInversePredicate(WideCmp->getPredicate()));
        R.getVPSingleValue()->replaceAllUsesWith(WideCmp);
        // If WideCmp doesn't have a debug location, use the one from the
        // negation, to preserve the location.
        if (!WideCmp->getDebugLoc() && R.getDebugLoc())
          WideCmp->setDebugLoc(R.getDebugLoc());
      }
    }
  }

  // Remove redundant DerviedIVs, that is 0 + A * 1 -> A and 0 + 0 * x -> 0.
  if ((match(&R,
             m_DerivedIV(m_SpecificInt(0), m_VPValue(A), m_SpecificInt(1))) ||
       match(&R,
             m_DerivedIV(m_SpecificInt(0), m_SpecificInt(0), m_VPValue()))) &&
      TypeInfo.inferScalarType(R.getOperand(1)) ==
          TypeInfo.inferScalarType(R.getVPSingleValue()))
    return R.getVPSingleValue()->replaceAllUsesWith(R.getOperand(1));

  if (match(&R, m_VPInstruction<VPInstruction::WideIVStep>(m_VPValue(X),
                                                           m_SpecificInt(1)))) {
    Type *WideStepTy = TypeInfo.inferScalarType(R.getVPSingleValue());
    if (TypeInfo.inferScalarType(X) != WideStepTy)
      X = VPBuilder(&R).createWidenCast(Instruction::Trunc, X, WideStepTy);
    R.getVPSingleValue()->replaceAllUsesWith(X);
    return;
  }

  // For i1 vp.merges produced by AnyOf reductions:
  // vp.merge true, (or x, y), x, evl -> vp.merge y, true, x, evl
  if (match(&R, m_Intrinsic<Intrinsic::vp_merge>(m_True(), m_VPValue(A),
                                                 m_VPValue(X), m_VPValue())) &&
      match(A, m_c_BinaryOr(m_Specific(X), m_VPValue(Y))) &&
      TypeInfo.inferScalarType(R.getVPSingleValue())->isIntegerTy(1)) {
    R.setOperand(1, R.getOperand(0));
    R.setOperand(0, Y);
    return;
  }
}

void VPlanTransforms::simplifyRecipes(VPlan &Plan, Type &CanonicalIVTy) {
  ReversePostOrderTraversal<VPBlockDeepTraversalWrapper<VPBlockBase *>> RPOT(
      Plan.getEntry());
  VPTypeAnalysis TypeInfo(&CanonicalIVTy);
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(RPOT)) {
    for (VPRecipeBase &R : make_early_inc_range(*VPBB)) {
      simplifyRecipe(R, TypeInfo);
    }
  }
}

static void narrowToSingleScalarRecipes(VPlan &Plan) {
  if (Plan.hasScalarVFOnly())
    return;

  // Try to narrow wide and replicating recipes to single scalar recipes,
  // based on VPlan analysis. Only process blocks in the loop region for now,
  // without traversing into nested regions, as recipes in replicate regions
  // cannot be converted yet.
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(
           vp_depth_first_shallow(Plan.getVectorLoopRegion()->getEntry()))) {
    for (VPRecipeBase &R : make_early_inc_range(reverse(*VPBB))) {
      auto *RepR = dyn_cast<VPReplicateRecipe>(&R);
      if (!RepR && !isa<VPWidenRecipe>(&R))
        continue;
      if (RepR && (RepR->isSingleScalar() || RepR->isPredicated()))
        continue;

      auto *RepOrWidenR = cast<VPSingleDefRecipe>(&R);
      // Skip recipes that aren't single scalars or don't have only their
      // scalar results used. In the latter case, we would introduce extra
      // broadcasts.
      if (!vputils::isSingleScalar(RepOrWidenR) ||
          any_of(RepOrWidenR->users(), [RepOrWidenR](VPUser *U) {
            return !U->usesScalars(RepOrWidenR);
          }))
        continue;

      auto *Clone = new VPReplicateRecipe(RepOrWidenR->getUnderlyingInstr(),
                                          RepOrWidenR->operands(),
                                          true /*IsSingleScalar*/);
      Clone->insertBefore(RepOrWidenR);
      RepOrWidenR->replaceAllUsesWith(Clone);
    }
  }
}

/// Normalize and simplify VPBlendRecipes. Should be run after simplifyRecipes
/// to make sure the masks are simplified.
static void simplifyBlends(VPlan &Plan) {
  using namespace llvm::VPlanPatternMatch;
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(
           vp_depth_first_shallow(Plan.getVectorLoopRegion()->getEntry()))) {
    for (VPRecipeBase &R : make_early_inc_range(*VPBB)) {
      auto *Blend = dyn_cast<VPBlendRecipe>(&R);
      if (!Blend)
        continue;

      // Try to remove redundant blend recipes.
      SmallPtrSet<VPValue *, 4> UniqueValues;
      if (Blend->isNormalized() || !match(Blend->getMask(0), m_False()))
        UniqueValues.insert(Blend->getIncomingValue(0));
      for (unsigned I = 1; I != Blend->getNumIncomingValues(); ++I)
        if (!match(Blend->getMask(I), m_False()))
          UniqueValues.insert(Blend->getIncomingValue(I));

      if (UniqueValues.size() == 1) {
        Blend->replaceAllUsesWith(*UniqueValues.begin());
        Blend->eraseFromParent();
        continue;
      }

      if (Blend->isNormalized())
        continue;

      // Normalize the blend so its first incoming value is used as the initial
      // value with the others blended into it.

      unsigned StartIndex = 0;
      for (unsigned I = 0; I != Blend->getNumIncomingValues(); ++I) {
        // If a value's mask is used only by the blend then is can be deadcoded.
        // TODO: Find the most expensive mask that can be deadcoded, or a mask
        // that's used by multiple blends where it can be removed from them all.
        VPValue *Mask = Blend->getMask(I);
        if (Mask->getNumUsers() == 1 && !match(Mask, m_False())) {
          StartIndex = I;
          break;
        }
      }

      SmallVector<VPValue *, 4> OperandsWithMask;
      OperandsWithMask.push_back(Blend->getIncomingValue(StartIndex));

      for (unsigned I = 0; I != Blend->getNumIncomingValues(); ++I) {
        if (I == StartIndex)
          continue;
        OperandsWithMask.push_back(Blend->getIncomingValue(I));
        OperandsWithMask.push_back(Blend->getMask(I));
      }

      auto *NewBlend = new VPBlendRecipe(
          cast<PHINode>(Blend->getUnderlyingValue()), OperandsWithMask);
      NewBlend->insertBefore(&R);

      VPValue *DeadMask = Blend->getMask(StartIndex);
      Blend->replaceAllUsesWith(NewBlend);
      Blend->eraseFromParent();
      recursivelyDeleteDeadRecipes(DeadMask);

      /// Simplify BLEND %a, %b, Not(%mask) -> BLEND %b, %a, %mask.
      VPValue *NewMask;
      if (NewBlend->getNumOperands() == 3 &&
          match(NewBlend->getMask(1), m_Not(m_VPValue(NewMask)))) {
        VPValue *Inc0 = NewBlend->getOperand(0);
        VPValue *Inc1 = NewBlend->getOperand(1);
        VPValue *OldMask = NewBlend->getOperand(2);
        NewBlend->setOperand(0, Inc1);
        NewBlend->setOperand(1, Inc0);
        NewBlend->setOperand(2, NewMask);
        if (OldMask->getNumUsers() == 0)
          cast<VPInstruction>(OldMask)->eraseFromParent();
      }
    }
  }
}

/// Optimize the width of vector induction variables in \p Plan based on a known
/// constant Trip Count, \p BestVF and \p BestUF.
static bool optimizeVectorInductionWidthForTCAndVFUF(VPlan &Plan,
                                                     ElementCount BestVF,
                                                     unsigned BestUF) {
  // Only proceed if we have not completely removed the vector region.
  if (!Plan.getVectorLoopRegion())
    return false;

  if (!Plan.getTripCount()->isLiveIn())
    return false;
  auto *TC = dyn_cast_if_present<ConstantInt>(
      Plan.getTripCount()->getUnderlyingValue());
  if (!TC || !BestVF.isFixed())
    return false;

  // Calculate the minimum power-of-2 bit width that can fit the known TC, VF
  // and UF. Returns at least 8.
  auto ComputeBitWidth = [](APInt TC, uint64_t Align) {
    APInt AlignedTC =
        Align * APIntOps::RoundingUDiv(TC, APInt(TC.getBitWidth(), Align),
                                       APInt::Rounding::UP);
    APInt MaxVal = AlignedTC - 1;
    return std::max<unsigned>(PowerOf2Ceil(MaxVal.getActiveBits()), 8);
  };
  unsigned NewBitWidth =
      ComputeBitWidth(TC->getValue(), BestVF.getKnownMinValue() * BestUF);

  LLVMContext &Ctx = Plan.getCanonicalIV()->getScalarType()->getContext();
  auto *NewIVTy = IntegerType::get(Ctx, NewBitWidth);

  bool MadeChange = false;

  VPBasicBlock *HeaderVPBB = Plan.getVectorLoopRegion()->getEntryBasicBlock();
  for (VPRecipeBase &Phi : HeaderVPBB->phis()) {
    auto *WideIV = dyn_cast<VPWidenIntOrFpInductionRecipe>(&Phi);

    // Currently only handle canonical IVs as it is trivial to replace the start
    // and stop values, and we currently only perform the optimization when the
    // IV has a single use.
    if (!WideIV || !WideIV->isCanonical() ||
        WideIV->hasMoreThanOneUniqueUser() ||
        NewIVTy == WideIV->getScalarType())
      continue;

    // Currently only handle cases where the single user is a header-mask
    // comparison with the backedge-taken-count.
    using namespace VPlanPatternMatch;
    if (!match(
            *WideIV->user_begin(),
            m_Binary<Instruction::ICmp>(
                m_Specific(WideIV),
                m_Broadcast(m_Specific(Plan.getOrCreateBackedgeTakenCount())))))
      continue;

    // Update IV operands and comparison bound to use new narrower type.
    auto *NewStart = Plan.getOrAddLiveIn(ConstantInt::get(NewIVTy, 0));
    WideIV->setStartValue(NewStart);
    auto *NewStep = Plan.getOrAddLiveIn(ConstantInt::get(NewIVTy, 1));
    WideIV->setStepValue(NewStep);
    // TODO: Remove once VPWidenIntOrFpInductionRecipe is fully expanded.
    VPInstructionWithType *OldStepVector = WideIV->getStepVector();
    assert(OldStepVector->getNumUsers() == 1 &&
           "step vector should only be used by single "
           "VPWidenIntOrFpInductionRecipe");
    auto *NewStepVector = new VPInstructionWithType(
        VPInstruction::StepVector, {}, NewIVTy, OldStepVector->getDebugLoc());
    NewStepVector->insertAfter(OldStepVector->getDefiningRecipe());
    OldStepVector->replaceAllUsesWith(NewStepVector);
    OldStepVector->eraseFromParent();

    auto *NewBTC = new VPWidenCastRecipe(
        Instruction::Trunc, Plan.getOrCreateBackedgeTakenCount(), NewIVTy);
    Plan.getVectorPreheader()->appendRecipe(NewBTC);
    auto *Cmp = cast<VPInstruction>(*WideIV->user_begin());
    Cmp->setOperand(1, NewBTC);

    MadeChange = true;
  }

  return MadeChange;
}

/// Return true if \p Cond is known to be true for given \p BestVF and \p
/// BestUF.
static bool isConditionTrueViaVFAndUF(VPValue *Cond, VPlan &Plan,
                                      ElementCount BestVF, unsigned BestUF,
                                      ScalarEvolution &SE) {
  using namespace llvm::VPlanPatternMatch;
  if (match(Cond, m_Binary<Instruction::Or>(m_VPValue(), m_VPValue())))
    return any_of(Cond->getDefiningRecipe()->operands(), [&Plan, BestVF, BestUF,
                                                          &SE](VPValue *C) {
      return isConditionTrueViaVFAndUF(C, Plan, BestVF, BestUF, SE);
    });

  auto *CanIV = Plan.getCanonicalIV();
  if (!match(Cond, m_Binary<Instruction::ICmp>(
                       m_Specific(CanIV->getBackedgeValue()),
                       m_Specific(&Plan.getVectorTripCount()))) ||
      cast<VPRecipeWithIRFlags>(Cond->getDefiningRecipe())->getPredicate() !=
          CmpInst::ICMP_EQ)
    return false;

  // The compare checks CanIV + VFxUF == vector trip count. The vector trip
  // count is not conveniently available as SCEV so far, so we compare directly
  // against the original trip count. This is stricter than necessary, as we
  // will only return true if the trip count == vector trip count.
  // TODO: Use SCEV for vector trip count once available, to cover cases where
  // vector trip count == UF * VF, but original trip count != UF * VF.
  const SCEV *TripCount =
      vputils::getSCEVExprForVPValue(Plan.getTripCount(), SE);
  assert(!isa<SCEVCouldNotCompute>(TripCount) &&
         "Trip count SCEV must be computable");
  ElementCount NumElements = BestVF.multiplyCoefficientBy(BestUF);
  const SCEV *C = SE.getElementCount(TripCount->getType(), NumElements);
  return SE.isKnownPredicate(CmpInst::ICMP_EQ, TripCount, C);
}

/// Try to simplify the branch condition of \p Plan. This may restrict the
/// resulting plan to \p BestVF and \p BestUF.
static bool simplifyBranchConditionForVFAndUF(VPlan &Plan, ElementCount BestVF,
                                              unsigned BestUF,
                                              PredicatedScalarEvolution &PSE) {
  VPRegionBlock *VectorRegion = Plan.getVectorLoopRegion();
  VPBasicBlock *ExitingVPBB = VectorRegion->getExitingBasicBlock();
  auto *Term = &ExitingVPBB->back();
  VPValue *Cond;
  ScalarEvolution &SE = *PSE.getSE();
  using namespace llvm::VPlanPatternMatch;
  if (match(Term, m_BranchOnCount(m_VPValue(), m_VPValue())) ||
      match(Term, m_BranchOnCond(
                      m_Not(m_ActiveLaneMask(m_VPValue(), m_VPValue()))))) {
    // Try to simplify the branch condition if TC <= VF * UF when the latch
    // terminator is   BranchOnCount or BranchOnCond where the input is
    // Not(ActiveLaneMask).
    const SCEV *TripCount =
        vputils::getSCEVExprForVPValue(Plan.getTripCount(), SE);
    assert(!isa<SCEVCouldNotCompute>(TripCount) &&
           "Trip count SCEV must be computable");
    ElementCount NumElements = BestVF.multiplyCoefficientBy(BestUF);
    const SCEV *C = SE.getElementCount(TripCount->getType(), NumElements);
    if (TripCount->isZero() ||
        !SE.isKnownPredicate(CmpInst::ICMP_ULE, TripCount, C))
      return false;
  } else if (match(Term, m_BranchOnCond(m_VPValue(Cond)))) {
    // For BranchOnCond, check if we can prove the condition to be true using VF
    // and UF.
    if (!isConditionTrueViaVFAndUF(Cond, Plan, BestVF, BestUF, SE))
      return false;
  } else {
    return false;
  }

  // The vector loop region only executes once. If possible, completely remove
  // the region, otherwise replace the terminator controlling the latch with
  // (BranchOnCond true).
  auto *Header = cast<VPBasicBlock>(VectorRegion->getEntry());
  auto *CanIVTy = Plan.getCanonicalIV()->getScalarType();
  if (all_of(
          Header->phis(),
          IsaPred<VPCanonicalIVPHIRecipe, VPFirstOrderRecurrencePHIRecipe>)) {
    for (VPRecipeBase &HeaderR : make_early_inc_range(Header->phis())) {
      auto *HeaderPhiR = cast<VPHeaderPHIRecipe>(&HeaderR);
      HeaderPhiR->replaceAllUsesWith(HeaderPhiR->getStartValue());
      HeaderPhiR->eraseFromParent();
    }

    VPBlockBase *Preheader = VectorRegion->getSinglePredecessor();
    VPBlockBase *Exit = VectorRegion->getSingleSuccessor();
    VPBlockUtils::disconnectBlocks(Preheader, VectorRegion);
    VPBlockUtils::disconnectBlocks(VectorRegion, Exit);

    for (VPBlockBase *B : vp_depth_first_shallow(VectorRegion->getEntry()))
      B->setParent(nullptr);

    VPBlockUtils::connectBlocks(Preheader, Header);
    VPBlockUtils::connectBlocks(ExitingVPBB, Exit);
    VPlanTransforms::simplifyRecipes(Plan, *CanIVTy);
  } else {
    // The vector region contains header phis for which we cannot remove the
    // loop region yet.
    LLVMContext &Ctx = SE.getContext();
    auto *BOC = new VPInstruction(
        VPInstruction::BranchOnCond,
        {Plan.getOrAddLiveIn(ConstantInt::getTrue(Ctx))}, Term->getDebugLoc());
    ExitingVPBB->appendRecipe(BOC);
  }

  Term->eraseFromParent();

  return true;
}

void VPlanTransforms::optimizeForVFAndUF(VPlan &Plan, ElementCount BestVF,
                                         unsigned BestUF,
                                         PredicatedScalarEvolution &PSE) {
  assert(Plan.hasVF(BestVF) && "BestVF is not available in Plan");
  assert(Plan.hasUF(BestUF) && "BestUF is not available in Plan");

  bool MadeChange =
      simplifyBranchConditionForVFAndUF(Plan, BestVF, BestUF, PSE);
  MadeChange |= optimizeVectorInductionWidthForTCAndVFUF(Plan, BestVF, BestUF);

  if (MadeChange) {
    Plan.setVF(BestVF);
    assert(Plan.getUF() == BestUF && "BestUF must match the Plan's UF");
  }
  // TODO: Further simplifications are possible
  //      1. Replace inductions with constants.
  //      2. Replace vector loop region with VPBasicBlock.
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
        VPDT.properlyDominates(Previous, SinkCandidate))
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
      if (!TryToPushSinkCandidate(cast<VPRecipeBase>(User)))
        return false;
    }
  }

  // Keep recipes to sink ordered by dominance so earlier instructions are
  // processed first.
  sort(WorkList, [&VPDT](const VPRecipeBase *A, const VPRecipeBase *B) {
    return VPDT.properlyDominates(A, B);
  });

  for (VPRecipeBase *SinkCandidate : WorkList) {
    if (SinkCandidate == FOR)
      continue;

    SinkCandidate->moveAfter(Previous);
    Previous = SinkCandidate;
  }
  return true;
}

/// Try to hoist \p Previous and its operands before all users of \p FOR.
static bool hoistPreviousBeforeFORUsers(VPFirstOrderRecurrencePHIRecipe *FOR,
                                        VPRecipeBase *Previous,
                                        VPDominatorTree &VPDT) {
  if (Previous->mayHaveSideEffects() || Previous->mayReadFromMemory())
    return false;

  // Collect recipes that need hoisting.
  SmallVector<VPRecipeBase *> HoistCandidates;
  SmallPtrSet<VPRecipeBase *, 8> Visited;
  VPRecipeBase *HoistPoint = nullptr;
  // Find the closest hoist point by looking at all users of FOR and selecting
  // the recipe dominating all other users.
  for (VPUser *U : FOR->users()) {
    auto *R = cast<VPRecipeBase>(U);
    if (!HoistPoint || VPDT.properlyDominates(R, HoistPoint))
      HoistPoint = R;
  }
  assert(all_of(FOR->users(),
                [&VPDT, HoistPoint](VPUser *U) {
                  auto *R = cast<VPRecipeBase>(U);
                  return HoistPoint == R ||
                         VPDT.properlyDominates(HoistPoint, R);
                }) &&
         "HoistPoint must dominate all users of FOR");

  auto NeedsHoisting = [HoistPoint, &VPDT,
                        &Visited](VPValue *HoistCandidateV) -> VPRecipeBase * {
    VPRecipeBase *HoistCandidate = HoistCandidateV->getDefiningRecipe();
    if (!HoistCandidate)
      return nullptr;
    VPRegionBlock *EnclosingLoopRegion =
        HoistCandidate->getParent()->getEnclosingLoopRegion();
    assert((!HoistCandidate->getParent()->getParent() ||
            HoistCandidate->getParent()->getParent() == EnclosingLoopRegion) &&
           "CFG in VPlan should still be flat, without replicate regions");
    // Hoist candidate was already visited, no need to hoist.
    if (!Visited.insert(HoistCandidate).second)
      return nullptr;

    // Candidate is outside loop region or a header phi, dominates FOR users w/o
    // hoisting.
    if (!EnclosingLoopRegion || isa<VPHeaderPHIRecipe>(HoistCandidate))
      return nullptr;

    // If we reached a recipe that dominates HoistPoint, we don't need to
    // hoist the recipe.
    if (VPDT.properlyDominates(HoistCandidate, HoistPoint))
      return nullptr;
    return HoistCandidate;
  };
  auto CanHoist = [&](VPRecipeBase *HoistCandidate) {
    // Avoid hoisting candidates with side-effects, as we do not yet analyze
    // associated dependencies.
    return !HoistCandidate->mayHaveSideEffects();
  };

  if (!NeedsHoisting(Previous->getVPSingleValue()))
    return true;

  // Recursively try to hoist Previous and its operands before all users of FOR.
  HoistCandidates.push_back(Previous);

  for (unsigned I = 0; I != HoistCandidates.size(); ++I) {
    VPRecipeBase *Current = HoistCandidates[I];
    assert(Current->getNumDefinedValues() == 1 &&
           "only recipes with a single defined value expected");
    if (!CanHoist(Current))
      return false;

    for (VPValue *Op : Current->operands()) {
      // If we reach FOR, it means the original Previous depends on some other
      // recurrence that in turn depends on FOR. If that is the case, we would
      // also need to hoist recipes involving the other FOR, which may break
      // dependencies.
      if (Op == FOR)
        return false;

      if (auto *R = NeedsHoisting(Op))
        HoistCandidates.push_back(R);
    }
  }

  // Order recipes to hoist by dominance so earlier instructions are processed
  // first.
  sort(HoistCandidates, [&VPDT](const VPRecipeBase *A, const VPRecipeBase *B) {
    return VPDT.properlyDominates(A, B);
  });

  for (VPRecipeBase *HoistCandidate : HoistCandidates) {
    HoistCandidate->moveBefore(*HoistPoint->getParent(),
                               HoistPoint->getIterator());
  }

  return true;
}

bool VPlanTransforms::adjustFixedOrderRecurrences(VPlan &Plan,
                                                  VPBuilder &LoopBuilder) {
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

    if (!sinkRecurrenceUsersAfterPrevious(FOR, Previous, VPDT) &&
        !hoistPreviousBeforeFORUsers(FOR, Previous, VPDT))
      return false;

    // Introduce a recipe to combine the incoming and previous values of a
    // fixed-order recurrence.
    VPBasicBlock *InsertBlock = Previous->getParent();
    if (isa<VPHeaderPHIRecipe>(Previous))
      LoopBuilder.setInsertPoint(InsertBlock, InsertBlock->getFirstNonPhi());
    else
      LoopBuilder.setInsertPoint(InsertBlock,
                                 std::next(Previous->getIterator()));

    auto *RecurSplice =
        LoopBuilder.createNaryOp(VPInstruction::FirstOrderRecurrenceSplice,
                                 {FOR, FOR->getBackedgeValue()});

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

    for (VPUser *U : collectUsersRecursively(PhiR))
      if (auto *RecWithFlags = dyn_cast<VPRecipeWithIRFlags>(U)) {
        RecWithFlags->dropPoisonGeneratingFlags();
      }
  }
}

/// Move loop-invariant recipes out of the vector loop region in \p Plan.
static void licm(VPlan &Plan) {
  VPBasicBlock *Preheader = Plan.getVectorPreheader();

  // Return true if we do not know how to (mechanically) hoist a given recipe
  // out of a loop region. Does not address legality concerns such as aliasing
  // or speculation safety.
  auto CannotHoistRecipe = [](VPRecipeBase &R) {
    // Allocas cannot be hoisted.
    auto *RepR = dyn_cast<VPReplicateRecipe>(&R);
    return RepR && RepR->getOpcode() == Instruction::Alloca;
  };

  // Hoist any loop invariant recipes from the vector loop region to the
  // preheader. Preform a shallow traversal of the vector loop region, to
  // exclude recipes in replicate regions.
  VPRegionBlock *LoopRegion = Plan.getVectorLoopRegion();
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(
           vp_depth_first_shallow(LoopRegion->getEntry()))) {
    for (VPRecipeBase &R : make_early_inc_range(*VPBB)) {
      if (CannotHoistRecipe(R))
        continue;
      // TODO: Relax checks in the future, e.g. we could also hoist reads, if
      // their memory location is not modified in the vector loop.
      if (R.mayHaveSideEffects() || R.mayReadFromMemory() || R.isPhi() ||
          any_of(R.operands(), [](VPValue *Op) {
            return !Op->isDefinedOutsideLoopRegions();
          }))
        continue;
      R.moveBefore(*Preheader, Preheader->end());
    }
  }
}

void VPlanTransforms::truncateToMinimalBitwidths(
    VPlan &Plan, const MapVector<Instruction *, uint64_t> &MinBWs) {
  // Keep track of created truncates, so they can be re-used. Note that we
  // cannot use RAUW after creating a new truncate, as this would could make
  // other uses have different types for their operands, making them invalidly
  // typed.
  DenseMap<VPValue *, VPWidenCastRecipe *> ProcessedTruncs;
  Type *CanonicalIVType = Plan.getCanonicalIV()->getScalarType();
  VPTypeAnalysis TypeInfo(CanonicalIVType);
  VPBasicBlock *PH = Plan.getVectorPreheader();
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(
           vp_depth_first_deep(Plan.getVectorLoopRegion()))) {
    for (VPRecipeBase &R : make_early_inc_range(*VPBB)) {
      if (!isa<VPWidenRecipe, VPWidenCastRecipe, VPReplicateRecipe,
               VPWidenSelectRecipe, VPWidenLoadRecipe, VPWidenIntrinsicRecipe>(
              &R))
        continue;

      VPValue *ResultVPV = R.getVPSingleValue();
      auto *UI = cast_or_null<Instruction>(ResultVPV->getUnderlyingValue());
      unsigned NewResSizeInBits = MinBWs.lookup(UI);
      if (!NewResSizeInBits)
        continue;

      // If the value wasn't vectorized, we must maintain the original scalar
      // type. Skip those here, after incrementing NumProcessedRecipes. Also
      // skip casts which do not need to be handled explicitly here, as
      // redundant casts will be removed during recipe simplification.
      if (isa<VPReplicateRecipe, VPWidenCastRecipe>(&R))
        continue;

      Type *OldResTy = TypeInfo.inferScalarType(ResultVPV);
      unsigned OldResSizeInBits = OldResTy->getScalarSizeInBits();
      assert(OldResTy->isIntegerTy() && "only integer types supported");
      (void)OldResSizeInBits;

      LLVMContext &Ctx = CanonicalIVType->getContext();
      auto *NewResTy = IntegerType::get(Ctx, NewResSizeInBits);

      // Any wrapping introduced by shrinking this operation shouldn't be
      // considered undefined behavior. So, we can't unconditionally copy
      // arithmetic wrapping flags to VPW.
      if (auto *VPW = dyn_cast<VPRecipeWithIRFlags>(&R))
        VPW->dropPoisonGeneratingFlags();

      using namespace llvm::VPlanPatternMatch;
      if (OldResSizeInBits != NewResSizeInBits &&
          !match(&R, m_Binary<Instruction::ICmp>(m_VPValue(), m_VPValue()))) {
        // Extend result to original width.
        auto *Ext =
            new VPWidenCastRecipe(Instruction::ZExt, ResultVPV, OldResTy);
        Ext->insertAfter(&R);
        ResultVPV->replaceAllUsesWith(Ext);
        Ext->setOperand(0, ResultVPV);
        assert(OldResSizeInBits > NewResSizeInBits && "Nothing to shrink?");
      } else {
        assert(
            match(&R, m_Binary<Instruction::ICmp>(m_VPValue(), m_VPValue())) &&
            "Only ICmps should not need extending the result.");
      }

      assert(!isa<VPWidenStoreRecipe>(&R) && "stores cannot be narrowed");
      if (isa<VPWidenLoadRecipe, VPWidenIntrinsicRecipe>(&R))
        continue;

      // Shrink operands by introducing truncates as needed.
      unsigned StartIdx = isa<VPWidenSelectRecipe>(&R) ? 1 : 0;
      for (unsigned Idx = StartIdx; Idx != R.getNumOperands(); ++Idx) {
        auto *Op = R.getOperand(Idx);
        unsigned OpSizeInBits =
            TypeInfo.inferScalarType(Op)->getScalarSizeInBits();
        if (OpSizeInBits == NewResSizeInBits)
          continue;
        assert(OpSizeInBits > NewResSizeInBits && "nothing to truncate");
        auto [ProcessedIter, IterIsEmpty] =
            ProcessedTruncs.insert({Op, nullptr});
        VPWidenCastRecipe *NewOp =
            IterIsEmpty
                ? new VPWidenCastRecipe(Instruction::Trunc, Op, NewResTy)
                : ProcessedIter->second;
        R.setOperand(Idx, NewOp);
        if (!IterIsEmpty)
          continue;
        ProcessedIter->second = NewOp;
        if (!Op->isLiveIn()) {
          NewOp->insertBefore(&R);
        } else {
          PH->appendRecipe(NewOp);
        }
      }

    }
  }
}

/// Remove BranchOnCond recipes with true conditions together with removing
/// dead edges to their successors.
static void removeBranchOnCondTrue(VPlan &Plan) {
  using namespace llvm::VPlanPatternMatch;
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(
           vp_depth_first_shallow(Plan.getEntry()))) {
    if (VPBB->getNumSuccessors() != 2 ||
        !match(&VPBB->back(), m_BranchOnCond(m_True())))
      continue;

    VPBasicBlock *RemovedSucc = cast<VPBasicBlock>(VPBB->getSuccessors()[1]);
    const auto &Preds = RemovedSucc->getPredecessors();
    assert(count(Preds, VPBB) == 1 &&
           "There must be a single edge between VPBB and its successor");
    unsigned DeadIdx = std::distance(Preds.begin(), find(Preds, VPBB));

    // Values coming from VPBB into ResumePhi recipes of RemoveSucc are removed
    // from these recipes.
    for (VPRecipeBase &R : make_early_inc_range(*RemovedSucc)) {
      assert((!isa<VPIRInstruction>(&R) ||
              !isa<PHINode>(cast<VPIRInstruction>(&R)->getInstruction())) &&
             !isa<VPHeaderPHIRecipe>(&R) &&
             "Cannot update VPIRInstructions wrapping phis or header phis yet");
      auto *VPI = dyn_cast<VPInstruction>(&R);
      if (!VPI || VPI->getOpcode() != VPInstruction::ResumePhi)
        break;
      VPBuilder B(VPI);
      SmallVector<VPValue *> NewOperands;
      // Create new operand list, with the dead incoming value filtered out.
      for (const auto &[Idx, Op] : enumerate(VPI->operands())) {
        if (Idx == DeadIdx)
          continue;
        NewOperands.push_back(Op);
      }
      VPI->replaceAllUsesWith(B.createNaryOp(VPInstruction::ResumePhi,
                                             NewOperands, VPI->getDebugLoc(),
                                             VPI->getName()));
      VPI->eraseFromParent();
    }
    // Disconnect blocks and remove the terminator. RemovedSucc will be deleted
    // automatically on VPlan destruction if it becomes unreachable.
    VPBlockUtils::disconnectBlocks(VPBB, RemovedSucc);
    VPBB->back().eraseFromParent();
  }
}

void VPlanTransforms::optimize(VPlan &Plan) {
  runPass(removeRedundantCanonicalIVs, Plan);
  runPass(removeRedundantInductionCasts, Plan);

  runPass(simplifyRecipes, Plan, *Plan.getCanonicalIV()->getScalarType());
  runPass(simplifyBlends, Plan);
  runPass(removeDeadRecipes, Plan);
  runPass(narrowToSingleScalarRecipes, Plan);
  runPass(legalizeAndOptimizeInductions, Plan);
  runPass(removeRedundantExpandSCEVRecipes, Plan);
  runPass(simplifyRecipes, Plan, *Plan.getCanonicalIV()->getScalarType());
  runPass(removeBranchOnCondTrue, Plan);
  runPass(removeDeadRecipes, Plan);

  runPass(createAndOptimizeReplicateRegions, Plan);
  runPass(mergeBlocksIntoPredecessors, Plan);
  runPass(licm, Plan);
}

// Add a VPActiveLaneMaskPHIRecipe and related recipes to \p Plan and replace
// the loop terminator with a branch-on-cond recipe with the negated
// active-lane-mask as operand. Note that this turns the loop into an
// uncountable one. Only the existing terminator is replaced, all other existing
// recipes/users remain unchanged, except for poison-generating flags being
// dropped from the canonical IV increment. Return the created
// VPActiveLaneMaskPHIRecipe.
//
// The function uses the following definitions:
//
//  %TripCount = DataWithControlFlowWithoutRuntimeCheck ?
//    calculate-trip-count-minus-VF (original TC) : original TC
//  %IncrementValue = DataWithControlFlowWithoutRuntimeCheck ?
//     CanonicalIVPhi : CanonicalIVIncrement
//  %StartV is the canonical induction start value.
//
// The function adds the following recipes:
//
// vector.ph:
//   %TripCount = calculate-trip-count-minus-VF (original TC)
//       [if DataWithControlFlowWithoutRuntimeCheck]
//   %EntryInc = canonical-iv-increment-for-part %StartV
//   %EntryALM = active-lane-mask %EntryInc, %TripCount
//
// vector.body:
//   ...
//   %P = active-lane-mask-phi [ %EntryALM, %vector.ph ], [ %ALM, %vector.body ]
//   ...
//   %InLoopInc = canonical-iv-increment-for-part %IncrementValue
//   %ALM = active-lane-mask %InLoopInc, TripCount
//   %Negated = Not %ALM
//   branch-on-cond %Negated
//
static VPActiveLaneMaskPHIRecipe *addVPLaneMaskPhiAndUpdateExitBranch(
    VPlan &Plan, bool DataAndControlFlowWithoutRuntimeCheck) {
  VPRegionBlock *TopRegion = Plan.getVectorLoopRegion();
  VPBasicBlock *EB = TopRegion->getExitingBasicBlock();
  auto *CanonicalIVPHI = Plan.getCanonicalIV();
  VPValue *StartV = CanonicalIVPHI->getStartValue();

  auto *CanonicalIVIncrement =
      cast<VPInstruction>(CanonicalIVPHI->getBackedgeValue());
  // TODO: Check if dropping the flags is needed if
  // !DataAndControlFlowWithoutRuntimeCheck.
  CanonicalIVIncrement->dropPoisonGeneratingFlags();
  DebugLoc DL = CanonicalIVIncrement->getDebugLoc();
  // We can't use StartV directly in the ActiveLaneMask VPInstruction, since
  // we have to take unrolling into account. Each part needs to start at
  //   Part * VF
  auto *VecPreheader = Plan.getVectorPreheader();
  VPBuilder Builder(VecPreheader);

  // Create the ActiveLaneMask instruction using the correct start values.
  VPValue *TC = Plan.getTripCount();

  VPValue *TripCount, *IncrementValue;
  if (!DataAndControlFlowWithoutRuntimeCheck) {
    // When the loop is guarded by a runtime overflow check for the loop
    // induction variable increment by VF, we can increment the value before
    // the get.active.lane mask and use the unmodified tripcount.
    IncrementValue = CanonicalIVIncrement;
    TripCount = TC;
  } else {
    // When avoiding a runtime check, the active.lane.mask inside the loop
    // uses a modified trip count and the induction variable increment is
    // done after the active.lane.mask intrinsic is called.
    IncrementValue = CanonicalIVPHI;
    TripCount = Builder.createNaryOp(VPInstruction::CalculateTripCountMinusVF,
                                     {TC}, DL);
  }
  auto *EntryIncrement = Builder.createOverflowingOp(
      VPInstruction::CanonicalIVIncrementForPart, {StartV}, {false, false}, DL,
      "index.part.next");

  // Create the active lane mask instruction in the VPlan preheader.
  auto *EntryALM =
      Builder.createNaryOp(VPInstruction::ActiveLaneMask, {EntryIncrement, TC},
                           DL, "active.lane.mask.entry");

  // Now create the ActiveLaneMaskPhi recipe in the main loop using the
  // preheader ActiveLaneMask instruction.
  auto *LaneMaskPhi = new VPActiveLaneMaskPHIRecipe(EntryALM, DebugLoc());
  LaneMaskPhi->insertAfter(CanonicalIVPHI);

  // Create the active lane mask for the next iteration of the loop before the
  // original terminator.
  VPRecipeBase *OriginalTerminator = EB->getTerminator();
  Builder.setInsertPoint(OriginalTerminator);
  auto *InLoopIncrement =
      Builder.createOverflowingOp(VPInstruction::CanonicalIVIncrementForPart,
                                  {IncrementValue}, {false, false}, DL);
  auto *ALM = Builder.createNaryOp(VPInstruction::ActiveLaneMask,
                                   {InLoopIncrement, TripCount}, DL,
                                   "active.lane.mask.next");
  LaneMaskPhi->addOperand(ALM);

  // Replace the original terminator with BranchOnCond. We have to invert the
  // mask here because a true condition means jumping to the exit block.
  auto *NotMask = Builder.createNot(ALM, DL);
  Builder.createNaryOp(VPInstruction::BranchOnCond, {NotMask}, DL);
  OriginalTerminator->eraseFromParent();
  return LaneMaskPhi;
}

/// Collect all VPValues representing a header mask through the (ICMP_ULE,
/// WideCanonicalIV, backedge-taken-count) pattern.
/// TODO: Introduce explicit recipe for header-mask instead of searching
/// for the header-mask pattern manually.
static SmallVector<VPValue *> collectAllHeaderMasks(VPlan &Plan) {
  SmallVector<VPValue *> WideCanonicalIVs;
  auto *FoundWidenCanonicalIVUser =
      find_if(Plan.getCanonicalIV()->users(),
              [](VPUser *U) { return isa<VPWidenCanonicalIVRecipe>(U); });
  assert(count_if(Plan.getCanonicalIV()->users(),
                  [](VPUser *U) { return isa<VPWidenCanonicalIVRecipe>(U); }) <=
             1 &&
         "Must have at most one VPWideCanonicalIVRecipe");
  if (FoundWidenCanonicalIVUser != Plan.getCanonicalIV()->users().end()) {
    auto *WideCanonicalIV =
        cast<VPWidenCanonicalIVRecipe>(*FoundWidenCanonicalIVUser);
    WideCanonicalIVs.push_back(WideCanonicalIV);
  }

  // Also include VPWidenIntOrFpInductionRecipes that represent a widened
  // version of the canonical induction.
  VPBasicBlock *HeaderVPBB = Plan.getVectorLoopRegion()->getEntryBasicBlock();
  for (VPRecipeBase &Phi : HeaderVPBB->phis()) {
    auto *WidenOriginalIV = dyn_cast<VPWidenIntOrFpInductionRecipe>(&Phi);
    if (WidenOriginalIV && WidenOriginalIV->isCanonical())
      WideCanonicalIVs.push_back(WidenOriginalIV);
  }

  // Walk users of wide canonical IVs and collect to all compares of the form
  // (ICMP_ULE, WideCanonicalIV, backedge-taken-count).
  SmallVector<VPValue *> HeaderMasks;
  for (auto *Wide : WideCanonicalIVs) {
    for (VPUser *U : SmallVector<VPUser *>(Wide->users())) {
      auto *HeaderMask = dyn_cast<VPInstruction>(U);
      if (!HeaderMask || !vputils::isHeaderMask(HeaderMask, Plan))
        continue;

      assert(HeaderMask->getOperand(0) == Wide &&
             "WidenCanonicalIV must be the first operand of the compare");
      HeaderMasks.push_back(HeaderMask);
    }
  }
  return HeaderMasks;
}

void VPlanTransforms::addActiveLaneMask(
    VPlan &Plan, bool UseActiveLaneMaskForControlFlow,
    bool DataAndControlFlowWithoutRuntimeCheck) {
  assert((!DataAndControlFlowWithoutRuntimeCheck ||
          UseActiveLaneMaskForControlFlow) &&
         "DataAndControlFlowWithoutRuntimeCheck implies "
         "UseActiveLaneMaskForControlFlow");

  auto *FoundWidenCanonicalIVUser =
      find_if(Plan.getCanonicalIV()->users(),
              [](VPUser *U) { return isa<VPWidenCanonicalIVRecipe>(U); });
  assert(FoundWidenCanonicalIVUser &&
         "Must have widened canonical IV when tail folding!");
  auto *WideCanonicalIV =
      cast<VPWidenCanonicalIVRecipe>(*FoundWidenCanonicalIVUser);
  VPSingleDefRecipe *LaneMask;
  if (UseActiveLaneMaskForControlFlow) {
    LaneMask = addVPLaneMaskPhiAndUpdateExitBranch(
        Plan, DataAndControlFlowWithoutRuntimeCheck);
  } else {
    VPBuilder B = VPBuilder::getToInsertAfter(WideCanonicalIV);
    LaneMask = B.createNaryOp(VPInstruction::ActiveLaneMask,
                              {WideCanonicalIV, Plan.getTripCount()}, nullptr,
                              "active.lane.mask");
  }

  // Walk users of WideCanonicalIV and replace all compares of the form
  // (ICMP_ULE, WideCanonicalIV, backedge-taken-count) with an
  // active-lane-mask.
  for (VPValue *HeaderMask : collectAllHeaderMasks(Plan))
    HeaderMask->replaceAllUsesWith(LaneMask);
}

/// Try to convert \p CurRecipe to a corresponding EVL-based recipe. Returns
/// nullptr if no EVL-based recipe could be created.
/// \p HeaderMask  Header Mask.
/// \p CurRecipe   Recipe to be transform.
/// \p TypeInfo    VPlan-based type analysis.
/// \p AllOneMask  The vector mask parameter of vector-predication intrinsics.
/// \p EVL         The explicit vector length parameter of vector-predication
/// intrinsics.
/// \p PrevEVL     The explicit vector length of the previous iteration. Only
/// required if \p CurRecipe is a VPInstruction::FirstOrderRecurrenceSplice.
static VPRecipeBase *createEVLRecipe(VPValue *HeaderMask,
                                     VPRecipeBase &CurRecipe,
                                     VPTypeAnalysis &TypeInfo,
                                     VPValue &AllOneMask, VPValue &EVL,
                                     VPValue *PrevEVL) {
  using namespace llvm::VPlanPatternMatch;
  auto GetNewMask = [&](VPValue *OrigMask) -> VPValue * {
    assert(OrigMask && "Unmasked recipe when folding tail");
    // HeaderMask will be handled using EVL.
    VPValue *Mask;
    if (match(OrigMask, m_LogicalAnd(m_Specific(HeaderMask), m_VPValue(Mask))))
      return Mask;
    return HeaderMask == OrigMask ? nullptr : OrigMask;
  };

  return TypeSwitch<VPRecipeBase *, VPRecipeBase *>(&CurRecipe)
      .Case<VPWidenLoadRecipe>([&](VPWidenLoadRecipe *L) {
        VPValue *NewMask = GetNewMask(L->getMask());
        return new VPWidenLoadEVLRecipe(*L, EVL, NewMask);
      })
      .Case<VPWidenStoreRecipe>([&](VPWidenStoreRecipe *S) {
        VPValue *NewMask = GetNewMask(S->getMask());
        return new VPWidenStoreEVLRecipe(*S, EVL, NewMask);
      })
      .Case<VPReductionRecipe>([&](VPReductionRecipe *Red) {
        VPValue *NewMask = GetNewMask(Red->getCondOp());
        return new VPReductionEVLRecipe(*Red, EVL, NewMask);
      })
      .Case<VPWidenSelectRecipe>([&](VPWidenSelectRecipe *Sel) {
        SmallVector<VPValue *> Ops(Sel->operands());
        Ops.push_back(&EVL);
        return new VPWidenIntrinsicRecipe(Intrinsic::vp_select, Ops,
                                          TypeInfo.inferScalarType(Sel),
                                          Sel->getDebugLoc());
      })
      .Case<VPInstruction>([&](VPInstruction *VPI) -> VPRecipeBase * {
        if (VPI->getOpcode() == VPInstruction::FirstOrderRecurrenceSplice) {
          assert(PrevEVL && "Fixed-order recurrences require previous EVL");
          VPValue *MinusOneVPV = VPI->getParent()->getPlan()->getOrAddLiveIn(
              ConstantInt::getSigned(Type::getInt32Ty(TypeInfo.getContext()),
                                     -1));
          SmallVector<VPValue *> Ops(VPI->operands());
          Ops.append({MinusOneVPV, &AllOneMask, PrevEVL, &EVL});
          return new VPWidenIntrinsicRecipe(Intrinsic::experimental_vp_splice,
                                            Ops, TypeInfo.inferScalarType(VPI),
                                            VPI->getDebugLoc());
        }

        VPValue *LHS, *RHS;
        // Transform select with a header mask condition
        //   select(header_mask, LHS, RHS)
        // into vector predication merge.
        //   vp.merge(all-true, LHS, RHS, EVL)
        if (!match(VPI, m_Select(m_Specific(HeaderMask), m_VPValue(LHS),
                                 m_VPValue(RHS))))
          return nullptr;
        // Use all true as the condition because this transformation is
        // limited to selects whose condition is a header mask.
        return new VPWidenIntrinsicRecipe(
            Intrinsic::vp_merge, {&AllOneMask, LHS, RHS, &EVL},
            TypeInfo.inferScalarType(LHS), VPI->getDebugLoc());
      })
      .Default([&](VPRecipeBase *R) { return nullptr; });
}

/// Replace recipes with their EVL variants.
static void transformRecipestoEVLRecipes(VPlan &Plan, VPValue &EVL) {
  Type *CanonicalIVType = Plan.getCanonicalIV()->getScalarType();
  VPTypeAnalysis TypeInfo(CanonicalIVType);
  LLVMContext &Ctx = CanonicalIVType->getContext();
  VPValue *AllOneMask = Plan.getOrAddLiveIn(ConstantInt::getTrue(Ctx));
  VPRegionBlock *LoopRegion = Plan.getVectorLoopRegion();
  VPBasicBlock *Header = LoopRegion->getEntryBasicBlock();

  // Create a scalar phi to track the previous EVL if fixed-order recurrence is
  // contained.
  VPInstruction *PrevEVL = nullptr;
  bool ContainsFORs =
      any_of(Header->phis(), IsaPred<VPFirstOrderRecurrencePHIRecipe>);
  if (ContainsFORs) {
    // TODO: Use VPInstruction::ExplicitVectorLength to get maximum EVL.
    VPValue *MaxEVL = &Plan.getVF();
    // Emit VPScalarCastRecipe in preheader if VF is not a 32 bits integer.
    VPBuilder Builder(LoopRegion->getPreheaderVPBB());
    if (unsigned VFSize =
            TypeInfo.inferScalarType(MaxEVL)->getScalarSizeInBits();
        VFSize != 32) {
      MaxEVL = Builder.createScalarCast(
          VFSize > 32 ? Instruction::Trunc : Instruction::ZExt, MaxEVL,
          Type::getInt32Ty(Ctx), DebugLoc());
    }
    Builder.setInsertPoint(Header, Header->getFirstNonPhi());
    PrevEVL = Builder.createScalarPhi({MaxEVL, &EVL}, DebugLoc(), "prev.evl");
  }

  for (VPUser *U : to_vector(Plan.getVF().users())) {
    if (auto *R = dyn_cast<VPVectorEndPointerRecipe>(U))
      R->setOperand(1, &EVL);
  }

  SmallVector<VPRecipeBase *> ToErase;

  for (VPValue *HeaderMask : collectAllHeaderMasks(Plan)) {
    for (VPUser *U : collectUsersRecursively(HeaderMask)) {
      auto *CurRecipe = cast<VPRecipeBase>(U);
      VPRecipeBase *EVLRecipe = createEVLRecipe(
          HeaderMask, *CurRecipe, TypeInfo, *AllOneMask, EVL, PrevEVL);
      if (!EVLRecipe)
        continue;

      [[maybe_unused]] unsigned NumDefVal = EVLRecipe->getNumDefinedValues();
      assert(NumDefVal == CurRecipe->getNumDefinedValues() &&
             "New recipe must define the same number of values as the "
             "original.");
      assert(
          NumDefVal <= 1 &&
          "Only supports recipes with a single definition or without users.");
      EVLRecipe->insertBefore(CurRecipe);
      if (isa<VPSingleDefRecipe, VPWidenLoadEVLRecipe>(EVLRecipe)) {
        VPValue *CurVPV = CurRecipe->getVPSingleValue();
        CurVPV->replaceAllUsesWith(EVLRecipe->getVPSingleValue());
      }
      // Defer erasing recipes till the end so that we don't invalidate the
      // VPTypeAnalysis cache.
      ToErase.push_back(CurRecipe);
    }
  }

  for (VPRecipeBase *R : reverse(ToErase)) {
    SmallVector<VPValue *> PossiblyDead(R->operands());
    R->eraseFromParent();
    for (VPValue *Op : PossiblyDead)
      recursivelyDeleteDeadRecipes(Op);
  }
}

/// Add a VPEVLBasedIVPHIRecipe and related recipes to \p Plan and
/// replaces all uses except the canonical IV increment of
/// VPCanonicalIVPHIRecipe with a VPEVLBasedIVPHIRecipe. VPCanonicalIVPHIRecipe
/// is used only for loop iterations counting after this transformation.
///
/// The function uses the following definitions:
///  %StartV is the canonical induction start value.
///
/// The function adds the following recipes:
///
/// vector.ph:
/// ...
///
/// vector.body:
/// ...
/// %EVLPhi = EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI [ %StartV, %vector.ph ],
///                                               [ %NextEVLIV, %vector.body ]
/// %AVL = sub original TC, %EVLPhi
/// %VPEVL = EXPLICIT-VECTOR-LENGTH %AVL
/// ...
/// %NextEVLIV = add IVSize (cast i32 %VPEVVL to IVSize), %EVLPhi
/// ...
///
/// If MaxSafeElements is provided, the function adds the following recipes:
/// vector.ph:
/// ...
///
/// vector.body:
/// ...
/// %EVLPhi = EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI [ %StartV, %vector.ph ],
///                                               [ %NextEVLIV, %vector.body ]
/// %AVL = sub original TC, %EVLPhi
/// %cmp = cmp ult %AVL, MaxSafeElements
/// %SAFE_AVL = select %cmp, %AVL, MaxSafeElements
/// %VPEVL = EXPLICIT-VECTOR-LENGTH %SAFE_AVL
/// ...
/// %NextEVLIV = add IVSize (cast i32 %VPEVL to IVSize), %EVLPhi
/// ...
///
bool VPlanTransforms::tryAddExplicitVectorLength(
    VPlan &Plan, const std::optional<unsigned> &MaxSafeElements) {
  VPBasicBlock *Header = Plan.getVectorLoopRegion()->getEntryBasicBlock();
  // The transform updates all users of inductions to work based on EVL, instead
  // of the VF directly. At the moment, widened inductions cannot be updated, so
  // bail out if the plan contains any.
  bool ContainsWidenInductions = any_of(
      Header->phis(),
      IsaPred<VPWidenIntOrFpInductionRecipe, VPWidenPointerInductionRecipe>);
  if (ContainsWidenInductions)
    return false;

  auto *CanonicalIVPHI = Plan.getCanonicalIV();
  VPValue *StartV = CanonicalIVPHI->getStartValue();

  // Create the ExplicitVectorLengthPhi recipe in the main loop.
  auto *EVLPhi = new VPEVLBasedIVPHIRecipe(StartV, DebugLoc());
  EVLPhi->insertAfter(CanonicalIVPHI);
  VPBuilder Builder(Header, Header->getFirstNonPhi());
  // Compute original TC - IV as the AVL (application vector length).
  VPValue *AVL = Builder.createNaryOp(
      Instruction::Sub, {Plan.getTripCount(), EVLPhi}, DebugLoc(), "avl");
  if (MaxSafeElements) {
    // Support for MaxSafeDist for correct loop emission.
    VPValue *AVLSafe = Plan.getOrAddLiveIn(
        ConstantInt::get(CanonicalIVPHI->getScalarType(), *MaxSafeElements));
    VPValue *Cmp = Builder.createICmp(ICmpInst::ICMP_ULT, AVL, AVLSafe);
    AVL = Builder.createSelect(Cmp, AVL, AVLSafe, DebugLoc(), "safe_avl");
  }
  auto *VPEVL = Builder.createNaryOp(VPInstruction::ExplicitVectorLength, AVL,
                                     DebugLoc());

  auto *CanonicalIVIncrement =
      cast<VPInstruction>(CanonicalIVPHI->getBackedgeValue());
  Builder.setInsertPoint(CanonicalIVIncrement);
  VPSingleDefRecipe *OpVPEVL = VPEVL;
  if (unsigned IVSize = CanonicalIVPHI->getScalarType()->getScalarSizeInBits();
      IVSize != 32) {
    OpVPEVL = Builder.createScalarCast(
        IVSize < 32 ? Instruction::Trunc : Instruction::ZExt, OpVPEVL,
        CanonicalIVPHI->getScalarType(), CanonicalIVIncrement->getDebugLoc());
  }
  auto *NextEVLIV = Builder.createOverflowingOp(
      Instruction::Add, {OpVPEVL, EVLPhi},
      {CanonicalIVIncrement->hasNoUnsignedWrap(),
       CanonicalIVIncrement->hasNoSignedWrap()},
      CanonicalIVIncrement->getDebugLoc(), "index.evl.next");
  EVLPhi->addOperand(NextEVLIV);

  transformRecipestoEVLRecipes(Plan, *VPEVL);

  // Replace all uses of VPCanonicalIVPHIRecipe by
  // VPEVLBasedIVPHIRecipe except for the canonical IV increment.
  CanonicalIVPHI->replaceAllUsesWith(EVLPhi);
  CanonicalIVIncrement->setOperand(0, CanonicalIVPHI);
  // TODO: support unroll factor > 1.
  Plan.setUF(1);
  return true;
}

void VPlanTransforms::dropPoisonGeneratingRecipes(
    VPlan &Plan,
    const std::function<bool(BasicBlock *)> &BlockNeedsPredication) {
  // Collect recipes in the backward slice of `Root` that may generate a poison
  // value that is used after vectorization.
  SmallPtrSet<VPRecipeBase *, 16> Visited;
  auto CollectPoisonGeneratingInstrsInBackwardSlice([&](VPRecipeBase *Root) {
    SmallVector<VPRecipeBase *, 16> Worklist;
    Worklist.push_back(Root);

    // Traverse the backward slice of Root through its use-def chain.
    while (!Worklist.empty()) {
      VPRecipeBase *CurRec = Worklist.pop_back_val();

      if (!Visited.insert(CurRec).second)
        continue;

      // Prune search if we find another recipe generating a widen memory
      // instruction. Widen memory instructions involved in address computation
      // will lead to gather/scatter instructions, which don't need to be
      // handled.
      if (isa<VPWidenMemoryRecipe, VPInterleaveRecipe, VPScalarIVStepsRecipe,
              VPHeaderPHIRecipe>(CurRec))
        continue;

      // This recipe contributes to the address computation of a widen
      // load/store. If the underlying instruction has poison-generating flags,
      // drop them directly.
      if (auto *RecWithFlags = dyn_cast<VPRecipeWithIRFlags>(CurRec)) {
        VPValue *A, *B;
        using namespace llvm::VPlanPatternMatch;
        // Dropping disjoint from an OR may yield incorrect results, as some
        // analysis may have converted it to an Add implicitly (e.g. SCEV used
        // for dependence analysis). Instead, replace it with an equivalent Add.
        // This is possible as all users of the disjoint OR only access lanes
        // where the operands are disjoint or poison otherwise.
        if (match(RecWithFlags, m_BinaryOr(m_VPValue(A), m_VPValue(B))) &&
            RecWithFlags->isDisjoint()) {
          VPBuilder Builder(RecWithFlags);
          VPInstruction *New = Builder.createOverflowingOp(
              Instruction::Add, {A, B}, {false, false},
              RecWithFlags->getDebugLoc());
          New->setUnderlyingValue(RecWithFlags->getUnderlyingValue());
          RecWithFlags->replaceAllUsesWith(New);
          RecWithFlags->eraseFromParent();
          CurRec = New;
        } else
          RecWithFlags->dropPoisonGeneratingFlags();
      } else {
        Instruction *Instr = dyn_cast_or_null<Instruction>(
            CurRec->getVPSingleValue()->getUnderlyingValue());
        (void)Instr;
        assert((!Instr || !Instr->hasPoisonGeneratingFlags()) &&
               "found instruction with poison generating flags not covered by "
               "VPRecipeWithIRFlags");
      }

      // Add new definitions to the worklist.
      for (VPValue *Operand : CurRec->operands())
        if (VPRecipeBase *OpDef = Operand->getDefiningRecipe())
          Worklist.push_back(OpDef);
    }
  });

  // Traverse all the recipes in the VPlan and collect the poison-generating
  // recipes in the backward slice starting at the address of a VPWidenRecipe or
  // VPInterleaveRecipe.
  auto Iter = vp_depth_first_deep(Plan.getEntry());
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(Iter)) {
    for (VPRecipeBase &Recipe : *VPBB) {
      if (auto *WidenRec = dyn_cast<VPWidenMemoryRecipe>(&Recipe)) {
        Instruction &UnderlyingInstr = WidenRec->getIngredient();
        VPRecipeBase *AddrDef = WidenRec->getAddr()->getDefiningRecipe();
        if (AddrDef && WidenRec->isConsecutive() &&
            BlockNeedsPredication(UnderlyingInstr.getParent()))
          CollectPoisonGeneratingInstrsInBackwardSlice(AddrDef);
      } else if (auto *InterleaveRec = dyn_cast<VPInterleaveRecipe>(&Recipe)) {
        VPRecipeBase *AddrDef = InterleaveRec->getAddr()->getDefiningRecipe();
        if (AddrDef) {
          // Check if any member of the interleave group needs predication.
          const InterleaveGroup<Instruction> *InterGroup =
              InterleaveRec->getInterleaveGroup();
          bool NeedPredication = false;
          for (int I = 0, NumMembers = InterGroup->getNumMembers();
               I < NumMembers; ++I) {
            Instruction *Member = InterGroup->getMember(I);
            if (Member)
              NeedPredication |= BlockNeedsPredication(Member->getParent());
          }

          if (NeedPredication)
            CollectPoisonGeneratingInstrsInBackwardSlice(AddrDef);
        }
      }
    }
  }
}

void VPlanTransforms::createInterleaveGroups(
    VPlan &Plan,
    const SmallPtrSetImpl<const InterleaveGroup<Instruction> *>
        &InterleaveGroups,
    VPRecipeBuilder &RecipeBuilder, const bool &ScalarEpilogueAllowed) {
  if (InterleaveGroups.empty())
    return;

  // Interleave memory: for each Interleave Group we marked earlier as relevant
  // for this VPlan, replace the Recipes widening its memory instructions with a
  // single VPInterleaveRecipe at its insertion point.
  VPDominatorTree VPDT;
  VPDT.recalculate(Plan);
  for (const auto *IG : InterleaveGroups) {
    SmallVector<VPValue *, 4> StoredValues;
    for (unsigned i = 0; i < IG->getFactor(); ++i)
      if (auto *SI = dyn_cast_or_null<StoreInst>(IG->getMember(i))) {
        auto *StoreR = cast<VPWidenStoreRecipe>(RecipeBuilder.getRecipe(SI));
        StoredValues.push_back(StoreR->getStoredValue());
      }

    bool NeedsMaskForGaps =
        IG->requiresScalarEpilogue() && !ScalarEpilogueAllowed;

    Instruction *IRInsertPos = IG->getInsertPos();
    auto *InsertPos =
        cast<VPWidenMemoryRecipe>(RecipeBuilder.getRecipe(IRInsertPos));

    // Get or create the start address for the interleave group.
    auto *Start =
        cast<VPWidenMemoryRecipe>(RecipeBuilder.getRecipe(IG->getMember(0)));
    VPValue *Addr = Start->getAddr();
    VPRecipeBase *AddrDef = Addr->getDefiningRecipe();
    if (AddrDef && !VPDT.properlyDominates(AddrDef, InsertPos)) {
      // TODO: Hoist Addr's defining recipe (and any operands as needed) to
      // InsertPos or sink loads above zero members to join it.
      bool InBounds = false;
      if (auto *Gep = dyn_cast<GetElementPtrInst>(
              getLoadStorePointerOperand(IRInsertPos)->stripPointerCasts()))
        InBounds = Gep->isInBounds();

      // We cannot re-use the address of member zero because it does not
      // dominate the insert position. Instead, use the address of the insert
      // position and create a PtrAdd adjusting it to the address of member
      // zero.
      assert(IG->getIndex(IRInsertPos) != 0 &&
             "index of insert position shouldn't be zero");
      auto &DL = IRInsertPos->getDataLayout();
      APInt Offset(32,
                   DL.getTypeAllocSize(getLoadStoreType(IRInsertPos)) *
                       IG->getIndex(IRInsertPos),
                   /*IsSigned=*/true);
      VPValue *OffsetVPV = Plan.getOrAddLiveIn(
          ConstantInt::get(IRInsertPos->getParent()->getContext(), -Offset));
      VPBuilder B(InsertPos);
      Addr = InBounds ? B.createInBoundsPtrAdd(InsertPos->getAddr(), OffsetVPV)
                      : B.createPtrAdd(InsertPos->getAddr(), OffsetVPV);
    }
    auto *VPIG = new VPInterleaveRecipe(IG, Addr, StoredValues,
                                        InsertPos->getMask(), NeedsMaskForGaps, InsertPos->getDebugLoc());
    VPIG->insertBefore(InsertPos);

    unsigned J = 0;
    for (unsigned i = 0; i < IG->getFactor(); ++i)
      if (Instruction *Member = IG->getMember(i)) {
        VPRecipeBase *MemberR = RecipeBuilder.getRecipe(Member);
        if (!Member->getType()->isVoidTy()) {
          VPValue *OriginalV = MemberR->getVPSingleValue();
          OriginalV->replaceAllUsesWith(VPIG->getVPValue(J));
          J++;
        }
        MemberR->eraseFromParent();
      }
  }
}

void VPlanTransforms::dissolveLoopRegions(VPlan &Plan) {
  // Replace loop regions with explicity CFG.
  SmallVector<VPRegionBlock *> LoopRegions;
  for (VPRegionBlock *R : VPBlockUtils::blocksOnly<VPRegionBlock>(
           vp_depth_first_deep(Plan.getEntry()))) {
    if (!R->isReplicator())
      LoopRegions.push_back(R);
  }
  for (VPRegionBlock *R : LoopRegions)
    R->dissolveToCFGLoop();
}

// Expand VPExtendedReductionRecipe to VPWidenCastRecipe + VPReductionRecipe.
static void expandVPExtendedReduction(VPExtendedReductionRecipe *ExtRed) {
  VPWidenCastRecipe *Ext;
  // Only ZExt contains non-neg flags.
  if (ExtRed->isZExt())
    Ext = new VPWidenCastRecipe(ExtRed->getExtOpcode(), ExtRed->getVecOp(),
                                ExtRed->getResultType(), ExtRed->isNonNeg(),
                                ExtRed->getDebugLoc());
  else
    Ext = new VPWidenCastRecipe(ExtRed->getExtOpcode(), ExtRed->getVecOp(),
                                ExtRed->getResultType(), ExtRed->getDebugLoc());

  auto *Red = new VPReductionRecipe(
      ExtRed->getRecurrenceKind(), FastMathFlags(), ExtRed->getChainOp(), Ext,
      ExtRed->getCondOp(), ExtRed->isOrdered(), ExtRed->getDebugLoc());
  Ext->insertBefore(ExtRed);
  Red->insertBefore(ExtRed);
  ExtRed->replaceAllUsesWith(Red);
  ExtRed->eraseFromParent();
}

// Expand VPMulAccumulateReductionRecipe to VPWidenRecipe (mul) +
// VPReductionRecipe (reduce.add)
// + VPWidenCastRecipe (optional).
static void
expandVPMulAccumulateReduction(VPMulAccumulateReductionRecipe *MulAcc) {
  // Generate inner VPWidenCastRecipes if necessary.
  // Note that we will drop the extend after mul which transforms
  // reduce.add(ext(mul(ext, ext))) to reduce.add(mul(ext, ext)).
  VPValue *Op0, *Op1;
  if (MulAcc->isExtended()) {
    Type *RedTy = MulAcc->getResultType();
    if (MulAcc->isZExt())
      Op0 = new VPWidenCastRecipe(MulAcc->getExtOpcode(), MulAcc->getVecOp0(),
                                  RedTy, MulAcc->isNonNeg(),
                                  MulAcc->getDebugLoc());
    else
      Op0 = new VPWidenCastRecipe(MulAcc->getExtOpcode(), MulAcc->getVecOp0(),
                                  RedTy, MulAcc->getDebugLoc());
    Op0->getDefiningRecipe()->insertBefore(MulAcc);
    // Prevent reduce.add(mul(ext(A), ext(A))) generate duplicate
    // VPWidenCastRecipe.
    if (MulAcc->getVecOp0() == MulAcc->getVecOp1()) {
      Op1 = Op0;
    } else {
      if (MulAcc->isZExt())
        Op1 = new VPWidenCastRecipe(MulAcc->getExtOpcode(), MulAcc->getVecOp1(),
                                    RedTy, MulAcc->isNonNeg(),
                                    MulAcc->getDebugLoc());
      else
        Op1 = new VPWidenCastRecipe(MulAcc->getExtOpcode(), MulAcc->getVecOp1(),
                                    RedTy, MulAcc->getDebugLoc());
      Op1->getDefiningRecipe()->insertBefore(MulAcc);
    }
  } else {
    // No extends in this MulAccRecipe.
    Op0 = MulAcc->getVecOp0();
    Op1 = MulAcc->getVecOp1();
  }

  std::array<VPValue *, 2> MulOps = {Op0, Op1};
  auto *Mul = new VPWidenRecipe(
      Instruction::Mul, ArrayRef(MulOps), MulAcc->hasNoUnsignedWrap(),
      MulAcc->hasNoSignedWrap(), MulAcc->getDebugLoc());
  Mul->insertBefore(MulAcc);

  auto *Red = new VPReductionRecipe(
      MulAcc->getRecurrenceKind(), FastMathFlags(), MulAcc->getChainOp(), Mul,
      MulAcc->getCondOp(), MulAcc->isOrdered(), MulAcc->getDebugLoc());
  Red->insertBefore(MulAcc);

  MulAcc->replaceAllUsesWith(Red);
  MulAcc->eraseFromParent();
}

void VPlanTransforms::convertToConcreteRecipes(VPlan &Plan,
                                               Type &CanonicalIVTy) {
  using namespace llvm::VPlanPatternMatch;

  VPTypeAnalysis TypeInfo(&CanonicalIVTy);
  SmallVector<VPRecipeBase *> ToRemove;
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(
           vp_depth_first_deep(Plan.getEntry()))) {
    for (VPRecipeBase &R : make_early_inc_range(*VPBB)) {
      if (isa<VPCanonicalIVPHIRecipe, VPEVLBasedIVPHIRecipe>(&R)) {
        auto *PhiR = cast<VPHeaderPHIRecipe>(&R);
        StringRef Name =
            isa<VPCanonicalIVPHIRecipe>(PhiR) ? "index" : "evl.based.iv";
        VPBuilder Builder(PhiR);
        auto *ScalarR = Builder.createScalarPhi(
            {PhiR->getStartValue(), PhiR->getBackedgeValue()},
            PhiR->getDebugLoc(), Name);
        PhiR->replaceAllUsesWith(ScalarR);
        ToRemove.push_back(PhiR);
        continue;
      }

      VPValue *VectorStep;
      VPValue *ScalarStep;
      if (!match(&R, m_VPInstruction<VPInstruction::WideIVStep>(
                         m_VPValue(VectorStep), m_VPValue(ScalarStep))))
        continue;

      // Expand WideIVStep.
      auto *VPI = cast<VPInstruction>(&R);
      VPBuilder Builder(VPI);
      Type *IVTy = TypeInfo.inferScalarType(VPI);
      if (TypeInfo.inferScalarType(VectorStep) != IVTy) {
        Instruction::CastOps CastOp = IVTy->isFloatingPointTy()
                                          ? Instruction::UIToFP
                                          : Instruction::Trunc;
        VectorStep = Builder.createWidenCast(CastOp, VectorStep, IVTy);
      }

      [[maybe_unused]] auto *ConstStep =
          ScalarStep->isLiveIn()
              ? dyn_cast<ConstantInt>(ScalarStep->getLiveInIRValue())
              : nullptr;
      assert(!ConstStep || ConstStep->getValue() != 1);
      (void)ConstStep;
      if (TypeInfo.inferScalarType(ScalarStep) != IVTy) {
        ScalarStep =
            Builder.createWidenCast(Instruction::Trunc, ScalarStep, IVTy);
      }

      std::optional<FastMathFlags> FMFs;
      if (IVTy->isFloatingPointTy())
        FMFs = VPI->getFastMathFlags();

      unsigned MulOpc =
          IVTy->isFloatingPointTy() ? Instruction::FMul : Instruction::Mul;
      VPInstruction *Mul = Builder.createNaryOp(
          MulOpc, {VectorStep, ScalarStep}, FMFs, R.getDebugLoc());
      VectorStep = Mul;
      VPI->replaceAllUsesWith(VectorStep);
      ToRemove.push_back(VPI);
    }
    for (VPRecipeBase &R : make_early_inc_range(*VPBB)) {
      if (auto *ExtRed = dyn_cast<VPExtendedReductionRecipe>(&R)) {
        expandVPExtendedReduction(ExtRed);
        continue;
      }
      if (auto *MulAcc = dyn_cast<VPMulAccumulateReductionRecipe>(&R))
        expandVPMulAccumulateReduction(MulAcc);
    }
  }

  for (VPRecipeBase *R : ToRemove)
    R->eraseFromParent();
}

void VPlanTransforms::handleUncountableEarlyExit(
    VPBasicBlock *EarlyExitingVPBB, VPBasicBlock *EarlyExitVPBB, VPlan &Plan,
    VPBasicBlock *HeaderVPBB, VPBasicBlock *LatchVPBB, VFRange &Range) {
  using namespace llvm::VPlanPatternMatch;

  VPBlockBase *MiddleVPBB = LatchVPBB->getSuccessors()[0];
  if (!EarlyExitVPBB->getSinglePredecessor() &&
      EarlyExitVPBB->getPredecessors()[1] == MiddleVPBB) {
    assert(EarlyExitVPBB->getNumPredecessors() == 2 &&
           EarlyExitVPBB->getPredecessors()[0] == EarlyExitingVPBB &&
           "unsupported early exit VPBB");
    // Early exit operand should always be last phi operand. If EarlyExitVPBB
    // has two predecessors and EarlyExitingVPBB is the first, swap the operands
    // of the phis.
    for (VPRecipeBase &R : EarlyExitVPBB->phis())
      cast<VPIRPhi>(&R)->swapOperands();
  }

  VPBuilder Builder(LatchVPBB->getTerminator());
  VPBlockBase *TrueSucc = EarlyExitingVPBB->getSuccessors()[0];
  assert(
      match(EarlyExitingVPBB->getTerminator(), m_BranchOnCond(m_VPValue())) &&
      "Terminator must be be BranchOnCond");
  VPValue *CondOfEarlyExitingVPBB =
      EarlyExitingVPBB->getTerminator()->getOperand(0);
  auto *CondToEarlyExit = TrueSucc == EarlyExitVPBB
                              ? CondOfEarlyExitingVPBB
                              : Builder.createNot(CondOfEarlyExitingVPBB);

  // Split the middle block and have it conditionally branch to the early exit
  // block if CondToEarlyExit.
  VPValue *IsEarlyExitTaken =
      Builder.createNaryOp(VPInstruction::AnyOf, {CondToEarlyExit});
  VPBasicBlock *NewMiddle = Plan.createVPBasicBlock("middle.split");
  VPBasicBlock *VectorEarlyExitVPBB =
      Plan.createVPBasicBlock("vector.early.exit");
  VPBlockUtils::insertOnEdge(LatchVPBB, MiddleVPBB, NewMiddle);
  VPBlockUtils::connectBlocks(NewMiddle, VectorEarlyExitVPBB);
  NewMiddle->swapSuccessors();

  VPBlockUtils::connectBlocks(VectorEarlyExitVPBB, EarlyExitVPBB);

  // Update the exit phis in the early exit block.
  VPBuilder MiddleBuilder(NewMiddle);
  VPBuilder EarlyExitB(VectorEarlyExitVPBB);
  for (VPRecipeBase &R : EarlyExitVPBB->phis()) {
    auto *ExitIRI = cast<VPIRPhi>(&R);
    // Early exit operand should always be last, i.e., 0 if EarlyExitVPBB has
    // a single predecessor and 1 if it has two.
    unsigned EarlyExitIdx = ExitIRI->getNumOperands() - 1;
    if (ExitIRI->getNumOperands() != 1) {
      // The first of two operands corresponds to the latch exit, via MiddleVPBB
      // predecessor. Extract its last lane.
      ExitIRI->extractLastLaneOfFirstOperand(MiddleBuilder);
    }

    VPValue *IncomingFromEarlyExit = ExitIRI->getOperand(EarlyExitIdx);
    auto IsVector = [](ElementCount VF) { return VF.isVector(); };
    // When the VFs are vectors, need to add `extract` to get the incoming value
    // from early exit. When the range contains scalar VF, limit the range to
    // scalar VF to prevent mis-compilation for the range containing both scalar
    // and vector VFs.
    if (!IncomingFromEarlyExit->isLiveIn() &&
        LoopVectorizationPlanner::getDecisionAndClampRange(IsVector, Range)) {
      // Update the incoming value from the early exit.
      VPValue *FirstActiveLane = EarlyExitB.createNaryOp(
          VPInstruction::FirstActiveLane, {CondToEarlyExit}, nullptr,
          "first.active.lane");
      IncomingFromEarlyExit = EarlyExitB.createNaryOp(
          Instruction::ExtractElement, {IncomingFromEarlyExit, FirstActiveLane},
          nullptr, "early.exit.value");
      ExitIRI->setOperand(EarlyExitIdx, IncomingFromEarlyExit);
    }
  }
  MiddleBuilder.createNaryOp(VPInstruction::BranchOnCond, {IsEarlyExitTaken});

  // Replace the condition controlling the non-early exit from the vector loop
  // with one exiting if either the original condition of the vector latch is
  // true or the early exit has been taken.
  auto *LatchExitingBranch = cast<VPInstruction>(LatchVPBB->getTerminator());
  assert(LatchExitingBranch->getOpcode() == VPInstruction::BranchOnCount &&
         "Unexpected terminator");
  auto *IsLatchExitTaken =
      Builder.createICmp(CmpInst::ICMP_EQ, LatchExitingBranch->getOperand(0),
                         LatchExitingBranch->getOperand(1));
  auto *AnyExitTaken = Builder.createNaryOp(
      Instruction::Or, {IsEarlyExitTaken, IsLatchExitTaken});
  Builder.createNaryOp(VPInstruction::BranchOnCond, AnyExitTaken);
  LatchExitingBranch->eraseFromParent();
}

/// This function tries convert extended in-loop reductions to
/// VPExtendedReductionRecipe and clamp the \p Range if it is beneficial and
/// valid. The created recipe must be lowered to concrete
/// recipes before execution.
static VPExtendedReductionRecipe *
tryToMatchAndCreateExtendedReduction(VPReductionRecipe *Red, VPCostContext &Ctx,
                                     VFRange &Range) {
  using namespace VPlanPatternMatch;

  Type *RedTy = Ctx.Types.inferScalarType(Red);
  VPValue *VecOp = Red->getVecOp();

  // Clamp the range if using extended-reduction is profitable.
  auto IsExtendedRedValidAndClampRange = [&](unsigned Opcode, bool isZExt,
                                             Type *SrcTy) -> bool {
    return LoopVectorizationPlanner::getDecisionAndClampRange(
        [&](ElementCount VF) {
          auto *SrcVecTy = cast<VectorType>(toVectorTy(SrcTy, VF));
          TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput;
          InstructionCost ExtRedCost = Ctx.TTI.getExtendedReductionCost(
              Opcode, isZExt, RedTy, SrcVecTy, Red->getFastMathFlags(),
              CostKind);
          InstructionCost ExtCost =
              cast<VPWidenCastRecipe>(VecOp)->computeCost(VF, Ctx);
          InstructionCost RedCost = Red->computeCost(VF, Ctx);
          return ExtRedCost.isValid() && ExtRedCost < ExtCost + RedCost;
        },
        Range);
  };

  VPValue *A;
  // Match reduce(ext)).
  if (match(VecOp, m_ZExtOrSExt(m_VPValue(A))) &&
      IsExtendedRedValidAndClampRange(
          RecurrenceDescriptor::getOpcode(Red->getRecurrenceKind()),
          cast<VPWidenCastRecipe>(VecOp)->getOpcode() ==
              Instruction::CastOps::ZExt,
          Ctx.Types.inferScalarType(A)))
    return new VPExtendedReductionRecipe(Red, cast<VPWidenCastRecipe>(VecOp));

  return nullptr;
}

/// This function tries convert extended in-loop reductions to
/// VPMulAccumulateReductionRecipe and clamp the \p Range if it is beneficial
/// and valid. The created VPExtendedReductionRecipe must be lower to concrete
/// recipes before execution. Patterns of MulAccumulateReduction:
///   reduce.add(mul(...)),
///   reduce.add(mul(ext(A), ext(B))),
///   reduce.add(ext(mul(ext(A), ext(B)))).
static VPMulAccumulateReductionRecipe *
tryToMatchAndCreateMulAccumulateReduction(VPReductionRecipe *Red,
                                          VPCostContext &Ctx, VFRange &Range) {
  using namespace VPlanPatternMatch;

  unsigned Opcode = RecurrenceDescriptor::getOpcode(Red->getRecurrenceKind());
  if (Opcode != Instruction::Add)
    return nullptr;

  Type *RedTy = Ctx.Types.inferScalarType(Red);

  // Clamp the range if using multiply-accumulate-reduction is profitable.
  auto IsMulAccValidAndClampRange =
      [&](bool isZExt, VPWidenRecipe *Mul, VPWidenCastRecipe *Ext0,
          VPWidenCastRecipe *Ext1, VPWidenCastRecipe *OuterExt) -> bool {
    return LoopVectorizationPlanner::getDecisionAndClampRange(
        [&](ElementCount VF) {
          TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput;
          Type *SrcTy =
              Ext0 ? Ctx.Types.inferScalarType(Ext0->getOperand(0)) : RedTy;
          auto *SrcVecTy = cast<VectorType>(toVectorTy(SrcTy, VF));
          InstructionCost MulAccCost =
              Ctx.TTI.getMulAccReductionCost(isZExt, RedTy, SrcVecTy, CostKind);
          InstructionCost MulCost = Mul->computeCost(VF, Ctx);
          InstructionCost RedCost = Red->computeCost(VF, Ctx);
          InstructionCost ExtCost = 0;
          if (Ext0)
            ExtCost += Ext0->computeCost(VF, Ctx);
          if (Ext1)
            ExtCost += Ext1->computeCost(VF, Ctx);
          if (OuterExt)
            ExtCost += OuterExt->computeCost(VF, Ctx);

          return MulAccCost.isValid() &&
                 MulAccCost < ExtCost + MulCost + RedCost;
        },
        Range);
  };

  VPValue *VecOp = Red->getVecOp();
  VPValue *A, *B;
  // Try to match reduce.add(mul(...)).
  if (match(VecOp, m_Mul(m_VPValue(A), m_VPValue(B)))) {
    auto *RecipeA =
        dyn_cast_if_present<VPWidenCastRecipe>(A->getDefiningRecipe());
    auto *RecipeB =
        dyn_cast_if_present<VPWidenCastRecipe>(B->getDefiningRecipe());
    auto *Mul = cast<VPWidenRecipe>(VecOp->getDefiningRecipe());

    // Match reduce.add(mul(ext, ext)).
    if (RecipeA && RecipeB &&
        (RecipeA->getOpcode() == RecipeB->getOpcode() || A == B) &&
        match(RecipeA, m_ZExtOrSExt(m_VPValue())) &&
        match(RecipeB, m_ZExtOrSExt(m_VPValue())) &&
        IsMulAccValidAndClampRange(RecipeA->getOpcode() ==
                                       Instruction::CastOps::ZExt,
                                   Mul, RecipeA, RecipeB, nullptr))
      return new VPMulAccumulateReductionRecipe(Red, Mul, RecipeA, RecipeB,
                                                RecipeA->getResultType());
    // Match reduce.add(mul).
    if (IsMulAccValidAndClampRange(true, Mul, nullptr, nullptr, nullptr))
      return new VPMulAccumulateReductionRecipe(Red, Mul, RedTy);
  }
  // Match reduce.add(ext(mul(ext(A), ext(B)))).
  // All extend recipes must have same opcode or A == B
  // which can be transform to reduce.add(zext(mul(sext(A), sext(B)))).
  if (match(VecOp, m_ZExtOrSExt(m_Mul(m_ZExtOrSExt(m_VPValue()),
                                      m_ZExtOrSExt(m_VPValue()))))) {
    auto *Ext = cast<VPWidenCastRecipe>(VecOp->getDefiningRecipe());
    auto *Mul = cast<VPWidenRecipe>(Ext->getOperand(0)->getDefiningRecipe());
    auto *Ext0 =
        cast<VPWidenCastRecipe>(Mul->getOperand(0)->getDefiningRecipe());
    auto *Ext1 =
        cast<VPWidenCastRecipe>(Mul->getOperand(1)->getDefiningRecipe());
    if ((Ext->getOpcode() == Ext0->getOpcode() || Ext0 == Ext1) &&
        Ext0->getOpcode() == Ext1->getOpcode() &&
        IsMulAccValidAndClampRange(Ext0->getOpcode() ==
                                       Instruction::CastOps::ZExt,
                                   Mul, Ext0, Ext1, Ext))
      return new VPMulAccumulateReductionRecipe(Red, Mul, Ext0, Ext1,
                                                Ext->getResultType());
  }
  return nullptr;
}

/// This function tries to create abstract recipes from the reduction recipe for
/// following optimizations and cost estimation.
static void tryToCreateAbstractReductionRecipe(VPReductionRecipe *Red,
                                               VPCostContext &Ctx,
                                               VFRange &Range) {
  VPReductionRecipe *AbstractR = nullptr;

  if (auto *MulAcc = tryToMatchAndCreateMulAccumulateReduction(Red, Ctx, Range))
    AbstractR = MulAcc;
  else if (auto *ExtRed = tryToMatchAndCreateExtendedReduction(Red, Ctx, Range))
    AbstractR = ExtRed;
  // Cannot create abstract inloop reduction recipes.
  if (!AbstractR)
    return;

  AbstractR->insertBefore(Red);
  Red->replaceAllUsesWith(AbstractR);
}

void VPlanTransforms::convertToAbstractRecipes(VPlan &Plan, VPCostContext &Ctx,
                                               VFRange &Range) {
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(
           vp_depth_first_deep(Plan.getVectorLoopRegion()))) {
    for (VPRecipeBase &R : *VPBB) {
      if (auto *Red = dyn_cast<VPReductionRecipe>(&R))
        tryToCreateAbstractReductionRecipe(Red, Ctx, Range);
    }
  }
}

void VPlanTransforms::materializeStepVectors(VPlan &Plan) {
  for (auto &Phi : Plan.getVectorLoopRegion()->getEntryBasicBlock()->phis()) {
    auto *IVR = dyn_cast<VPWidenIntOrFpInductionRecipe>(&Phi);
    if (!IVR)
      continue;

    Type *Ty = IVR->getPHINode()->getType();
    if (TruncInst *Trunc = IVR->getTruncInst())
      Ty = Trunc->getType();
    if (Ty->isFloatingPointTy())
      Ty = IntegerType::get(Ty->getContext(), Ty->getScalarSizeInBits());

    VPBuilder Builder(Plan.getVectorPreheader());
    VPInstruction *StepVector = Builder.createNaryOp(
        VPInstruction::StepVector, {}, Ty, {}, IVR->getDebugLoc());
    assert(IVR->getNumOperands() == 3 &&
           "can only add step vector before unrolling");
    IVR->addOperand(StepVector);
  }
}

void VPlanTransforms::materializeBroadcasts(VPlan &Plan) {
  if (Plan.hasScalarVFOnly())
    return;

#ifndef NDEBUG
  VPDominatorTree VPDT;
  VPDT.recalculate(Plan);
#endif

  SmallVector<VPValue *> VPValues;
  if (Plan.getOrCreateBackedgeTakenCount()->getNumUsers() > 0)
    VPValues.push_back(Plan.getOrCreateBackedgeTakenCount());
  append_range(VPValues, Plan.getLiveIns());
  for (VPRecipeBase &R : *Plan.getEntry())
    append_range(VPValues, R.definedValues());

  auto *VectorPreheader = Plan.getVectorPreheader();
  for (VPValue *VPV : VPValues) {
    if (all_of(VPV->users(),
               [VPV](VPUser *U) { return U->usesScalars(VPV); }) ||
        (VPV->isLiveIn() && VPV->getLiveInIRValue() &&
         isa<Constant>(VPV->getLiveInIRValue())))
      continue;

    // Add explicit broadcast at the insert point that dominates all users.
    VPBasicBlock *HoistBlock = VectorPreheader;
    VPBasicBlock::iterator HoistPoint = VectorPreheader->end();
    for (VPUser *User : VPV->users()) {
      if (User->usesScalars(VPV))
        continue;
      if (cast<VPRecipeBase>(User)->getParent() == VectorPreheader)
        HoistPoint = HoistBlock->begin();
      else
        assert(VPDT.dominates(VectorPreheader,
                              cast<VPRecipeBase>(User)->getParent()) &&
               "All users must be in the vector preheader or dominated by it");
    }

    VPBuilder Builder(cast<VPBasicBlock>(HoistBlock), HoistPoint);
    auto *Broadcast = Builder.createNaryOp(VPInstruction::Broadcast, {VPV});
    VPV->replaceUsesWithIf(Broadcast,
                           [VPV, Broadcast](VPUser &U, unsigned Idx) {
                             return Broadcast != &U && !U.usesScalars(VPV);
                           });
  }
}

/// Returns true if \p V is VPWidenLoadRecipe or VPInterleaveRecipe that can be
/// converted to a narrower recipe. \p V is used by a wide recipe \p WideMember
/// that feeds a store interleave group at index \p Idx, \p WideMember0 is the
/// recipe feeding the same interleave group at index 0. A VPWidenLoadRecipe can
/// be narrowed to an index-independent load if it feeds all wide ops at all
/// indices (checked by via the operands of the wide recipe at lane0, \p
/// WideMember0). A VPInterleaveRecipe can be narrowed to a wide load, if \p V
/// is defined at \p Idx of a load interleave group.
static bool canNarrowLoad(VPWidenRecipe *WideMember0, VPWidenRecipe *WideMember,
                          VPValue *V, unsigned Idx) {
  auto *DefR = V->getDefiningRecipe();
  if (!DefR)
    return false;
  if (auto *W = dyn_cast<VPWidenLoadRecipe>(DefR))
    return !W->getMask() &&
           all_of(zip(WideMember0->operands(), WideMember->operands()),
                  [V](const auto P) {
                    // V must be as at the same places in both WideMember0 and
                    // WideMember.
                    const auto &[WideMember0Op, WideMemberOp] = P;
                    return (WideMember0Op == V) == (WideMemberOp == V);
                  });

  if (auto *IR = dyn_cast<VPInterleaveRecipe>(DefR))
    return IR->getInterleaveGroup()->getFactor() ==
               IR->getInterleaveGroup()->getNumMembers() &&
           IR->getVPValue(Idx) == V;
  return false;
}

/// Returns true if \p IR is a full interleave group with factor and number of
/// members both equal to \p VF. The interleave group must also access the full
/// vector width \p VectorRegWidth.
static bool isConsecutiveInterleaveGroup(VPInterleaveRecipe *InterleaveR,
                                         unsigned VF, VPTypeAnalysis &TypeInfo,
                                         unsigned VectorRegWidth) {
  if (!InterleaveR)
    return false;

  Type *GroupElementTy = nullptr;
  if (InterleaveR->getStoredValues().empty()) {
    GroupElementTy = TypeInfo.inferScalarType(InterleaveR->getVPValue(0));
    if (!all_of(InterleaveR->definedValues(),
                [&TypeInfo, GroupElementTy](VPValue *Op) {
                  return TypeInfo.inferScalarType(Op) == GroupElementTy;
                }))
      return false;
  } else {
    GroupElementTy =
        TypeInfo.inferScalarType(InterleaveR->getStoredValues()[0]);
    if (!all_of(InterleaveR->getStoredValues(),
                [&TypeInfo, GroupElementTy](VPValue *Op) {
                  return TypeInfo.inferScalarType(Op) == GroupElementTy;
                }))
      return false;
  }

  unsigned GroupSize = GroupElementTy->getScalarSizeInBits() * VF;
  auto IG = InterleaveR->getInterleaveGroup();
  return IG->getFactor() == VF && IG->getNumMembers() == VF &&
         GroupSize == VectorRegWidth;
}

void VPlanTransforms::narrowInterleaveGroups(VPlan &Plan, ElementCount VF,
                                             unsigned VectorRegWidth) {
  using namespace llvm::VPlanPatternMatch;
  VPRegionBlock *VectorLoop = Plan.getVectorLoopRegion();
  if (VF.isScalable() || !VectorLoop)
    return;

  VPCanonicalIVPHIRecipe *CanonicalIV = Plan.getCanonicalIV();
  Type *CanonicalIVType = CanonicalIV->getScalarType();
  VPTypeAnalysis TypeInfo(CanonicalIVType);

  unsigned FixedVF = VF.getFixedValue();
  SmallVector<VPInterleaveRecipe *> StoreGroups;
  for (auto &R : *VectorLoop->getEntryBasicBlock()) {
    if (isa<VPCanonicalIVPHIRecipe>(&R) ||
        match(&R, m_BranchOnCount(m_VPValue(), m_VPValue())))
      continue;

    // Bail out on recipes not supported at the moment:
    //  * phi recipes other than the canonical induction
    //  * recipes writing to memory except interleave groups
    // Only support plans with a canonical induction phi.
    if (R.isPhi())
      return;

    auto *InterleaveR = dyn_cast<VPInterleaveRecipe>(&R);
    if (R.mayWriteToMemory() && !InterleaveR)
      return;

    // Do not narrow interleave groups if there are VectorPointer recipes and
    // the plan was unrolled. The recipe implicitly uses VF from
    // VPTransformState.
    // TODO: Remove restriction once the VF for the VectorPointer offset is
    // modeled explicitly as operand.
    if (isa<VPVectorPointerRecipe>(&R) && Plan.getUF() > 1)
      return;

    // All other ops are allowed, but we reject uses that cannot be converted
    // when checking all allowed consumers (store interleave groups) below.
    if (!InterleaveR)
      continue;

    // Bail out on non-consecutive interleave groups.
    if (!isConsecutiveInterleaveGroup(InterleaveR, FixedVF, TypeInfo,
                                      VectorRegWidth))
      return;

    // Skip read interleave groups.
    if (InterleaveR->getStoredValues().empty())
      continue;

    // For now, we only support full interleave groups storing load interleave
    // groups.
    if (all_of(enumerate(InterleaveR->getStoredValues()), [](auto Op) {
          VPRecipeBase *DefR = Op.value()->getDefiningRecipe();
          if (!DefR)
            return false;
          auto *IR = dyn_cast<VPInterleaveRecipe>(DefR);
          return IR &&
                 IR->getInterleaveGroup()->getFactor() ==
                     IR->getInterleaveGroup()->getNumMembers() &&
                 IR->getVPValue(Op.index()) == Op.value();
        })) {
      StoreGroups.push_back(InterleaveR);
      continue;
    }

    // Check if all values feeding InterleaveR are matching wide recipes, which
    // operands that can be narrowed.
    auto *WideMember0 = dyn_cast_or_null<VPWidenRecipe>(
        InterleaveR->getStoredValues()[0]->getDefiningRecipe());
    if (!WideMember0)
      return;
    for (const auto &[I, V] : enumerate(InterleaveR->getStoredValues())) {
      auto *R = dyn_cast<VPWidenRecipe>(V->getDefiningRecipe());
      if (!R || R->getOpcode() != WideMember0->getOpcode() ||
          R->getNumOperands() > 2)
        return;
      if (any_of(R->operands(), [WideMember0, Idx = I, R](VPValue *V) {
            return !canNarrowLoad(WideMember0, R, V, Idx);
          }))
        return;
    }
    StoreGroups.push_back(InterleaveR);
  }

  if (StoreGroups.empty())
    return;

  // Convert InterleaveGroup \p R to a single VPWidenLoadRecipe.
  auto NarrowOp = [](VPRecipeBase *R) -> VPValue * {
    if (auto *LoadGroup = dyn_cast<VPInterleaveRecipe>(R)) {
      // Narrow interleave group to wide load, as transformed VPlan will only
      // process one original iteration.
      auto *L = new VPWidenLoadRecipe(
          *cast<LoadInst>(LoadGroup->getInterleaveGroup()->getInsertPos()),
          LoadGroup->getAddr(), LoadGroup->getMask(), /*Consecutive=*/true,
          /*Reverse=*/false, {}, LoadGroup->getDebugLoc());
      L->insertBefore(LoadGroup);
      return L;
    }

    auto *WideLoad = cast<VPWidenLoadRecipe>(R);

    // Narrow wide load to uniform scalar load, as transformed VPlan will only
    // process one original iteration.
    auto *N = new VPReplicateRecipe(&WideLoad->getIngredient(),
                                    WideLoad->operands(), /*IsUniform*/ true,
                                    /*Mask*/ nullptr, *WideLoad);
    N->insertBefore(WideLoad);
    return N;
  };

  // Narrow operation tree rooted at store groups.
  for (auto *StoreGroup : StoreGroups) {
    VPValue *Res = nullptr;
    if (auto *WideMember0 = dyn_cast<VPWidenRecipe>(
            StoreGroup->getStoredValues()[0]->getDefiningRecipe())) {
      for (unsigned Idx = 0, E = WideMember0->getNumOperands(); Idx != E; ++Idx)
        WideMember0->setOperand(
            Idx, NarrowOp(WideMember0->getOperand(Idx)->getDefiningRecipe()));
      Res = WideMember0;
    } else {
      Res = NarrowOp(StoreGroup->getStoredValues()[0]->getDefiningRecipe());
    }

    auto *S = new VPWidenStoreRecipe(
        *cast<StoreInst>(StoreGroup->getInterleaveGroup()->getInsertPos()),
        StoreGroup->getAddr(), Res, nullptr, /*Consecutive=*/true,
        /*Reverse=*/false, {}, StoreGroup->getDebugLoc());
    S->insertBefore(StoreGroup);
    StoreGroup->eraseFromParent();
  }

  // Adjust induction to reflect that the transformed plan only processes one
  // original iteration.
  auto *CanIV = Plan.getCanonicalIV();
  auto *Inc = cast<VPInstruction>(CanIV->getBackedgeValue());
  Inc->setOperand(1, Plan.getOrAddLiveIn(ConstantInt::get(
                         CanIV->getScalarType(), 1 * Plan.getUF())));
  Plan.getVF().replaceAllUsesWith(
      Plan.getOrAddLiveIn(ConstantInt::get(CanIV->getScalarType(), 1)));
  removeDeadRecipes(Plan);
}
