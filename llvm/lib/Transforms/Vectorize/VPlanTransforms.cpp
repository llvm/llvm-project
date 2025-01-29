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
#include "VPlanPatternMatch.h"
#include "VPlanUtils.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/PatternMatch.h"

using namespace llvm;

void VPlanTransforms::VPInstructionsToVPRecipes(
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
              false /*Consecutive*/, false /*Reverse*/,
              Ingredient.getDebugLoc());
        } else if (StoreInst *Store = dyn_cast<StoreInst>(Inst)) {
          NewRecipe = new VPWidenStoreRecipe(
              *Store, Ingredient.getOperand(1), Ingredient.getOperand(0),
              nullptr /*Mask*/, false /*Consecutive*/, false /*Reverse*/,
              Ingredient.getDebugLoc());
        } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Inst)) {
          NewRecipe = new VPWidenGEPRecipe(GEP, Ingredient.operands());
        } else if (CallInst *CI = dyn_cast<CallInst>(Inst)) {
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
      auto *UI = cast<VPRecipeBase>(U);
      if (UI->getParent() == SinkTo)
        return true;
      NeedsDuplicating = UI->onlyFirstLaneUsed(SinkCandidate);
      // We only know how to duplicate VPRecipeRecipes for now.
      return NeedsDuplicating && isa<VPReplicateRecipe>(SinkCandidate);
    };
    if (!all_of(SinkCandidate->users(), CanSinkWithUser))
      continue;

    if (NeedsDuplicating) {
      if (ScalarVFOnly)
        continue;
      Instruction *I = SinkCandidate->getUnderlyingInstr();
      auto *Clone = new VPReplicateRecipe(I, SinkCandidate->operands(), true);
      // TODO: add ".cloned" suffix to name of Clone's VPValue.

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
  auto *BOMRecipe = new VPBranchOnMaskRecipe(BlockInMask);
  auto *Entry =
      Plan.createVPBasicBlock(Twine(RegionName) + ".entry", BOMRecipe);

  // Replace predicated replicate recipe with a replicate recipe without a
  // mask but in the replicate region.
  auto *RecipeWithoutMask = new VPReplicateRecipe(
      PredRecipe->getUnderlyingInstr(),
      make_range(PredRecipe->op_begin(), std::prev(PredRecipe->op_end())),
      PredRecipe->isUniform());
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
    VPBlockBase *Region = createReplicateRegion(RepR, Plan);
    Region->setParent(CurrentBlock->getParent());
    VPBlockUtils::insertOnEdge(CurrentBlock, SplitBlock, Region);
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
    auto *ParentRegion = cast_or_null<VPRegionBlock>(VPBB->getParent());
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
  return Builder.createScalarIVSteps(InductionOpcode, FPBinOp, BaseIV, Step);
}

static SmallVector<VPUser *> collectUsersRecursively(VPValue *V) {
  SetVector<VPUser *> Users(V->user_begin(), V->user_end());
  for (unsigned I = 0; I != Users.size(); ++I) {
    VPRecipeBase *Cur = cast<VPRecipeBase>(Users[I]);
    if (isa<VPHeaderPHIRecipe>(Cur))
      continue;
    for (VPValue *V : Cur->definedValues())
      Users.insert(V->user_begin(), V->user_end());
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
  bool HasOnlyVectorVFs = !Plan.hasVF(ElementCount::getFixed(1));
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
          (RepR && (RepR->isUniform() || RepR->isPredicated())))
        continue;

      // Skip recipes that may have other lanes than their first used.
      if (!vputils::isUniformAfterVectorization(Def) &&
          !vputils::onlyFirstLaneUsed(Def))
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

void VPlanTransforms::optimizeInductionExitUsers(
    VPlan &Plan, DenseMap<VPValue *, VPValue *> &EndValues) {
  using namespace VPlanPatternMatch;
  SmallVector<VPIRBasicBlock *> ExitVPBBs(Plan.getExitBlocks());
  if (ExitVPBBs.size() != 1)
    return;

  VPIRBasicBlock *ExitVPBB = ExitVPBBs[0];
  VPBlockBase *PredVPBB = ExitVPBB->getSinglePredecessor();
  if (!PredVPBB)
    return;
  assert(PredVPBB == Plan.getMiddleBlock() &&
         "predecessor must be the middle block");

  VPTypeAnalysis TypeInfo(Plan.getCanonicalIV()->getScalarType());
  VPBuilder B(Plan.getMiddleBlock()->getTerminator());
  for (VPRecipeBase &R : *ExitVPBB) {
    auto *ExitIRI = cast<VPIRInstruction>(&R);
    if (!isa<PHINode>(ExitIRI->getInstruction()))
      break;

    VPValue *Incoming;
    if (!match(ExitIRI->getOperand(0),
               m_VPInstruction<VPInstruction::ExtractFromEnd>(
                   m_VPValue(Incoming), m_SpecificInt(1))))
      continue;

    auto *WideIV = getOptimizableIVOf(Incoming);
    if (!WideIV)
      continue;
    VPValue *EndValue = EndValues.lookup(WideIV);
    assert(EndValue && "end value must have been pre-computed");

    if (Incoming != WideIV) {
      ExitIRI->setOperand(0, EndValue);
      continue;
    }

    VPValue *Escape = nullptr;
    VPValue *Step = WideIV->getStepValue();
    Type *ScalarTy = TypeInfo.inferScalarType(WideIV);
    if (ScalarTy->isIntegerTy()) {
      Escape =
          B.createNaryOp(Instruction::Sub, {EndValue, Step}, {}, "ind.escape");
    } else if (ScalarTy->isPointerTy()) {
      auto *Zero = Plan.getOrAddLiveIn(
          ConstantInt::get(Step->getLiveInIRValue()->getType(), 0));
      Escape = B.createPtrAdd(EndValue,
                              B.createNaryOp(Instruction::Sub, {Zero, Step}),
                              {}, "ind.escape");
    } else if (ScalarTy->isFloatingPointTy()) {
      const auto &ID = WideIV->getInductionDescriptor();
      Escape = B.createNaryOp(
          ID.getInductionBinOp()->getOpcode() == Instruction::FAdd
              ? Instruction::FSub
              : Instruction::FAdd,
          {EndValue, Step}, {ID.getInductionBinOp()->getFastMathFlags()});
    } else {
      llvm_unreachable("all possible induction types must be handled");
    }
    ExitIRI->setOperand(0, Escape);
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

/// Try to simplify recipe \p R.
static void simplifyRecipe(VPRecipeBase &R, VPTypeAnalysis &TypeInfo) {
  using namespace llvm::VPlanPatternMatch;

  if (auto *Blend = dyn_cast<VPBlendRecipe>(&R)) {
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
      return;
    }

    if (Blend->isNormalized())
      return;

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
    return;
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
  VPValue *X, *Y, *X1, *Y1;
  if (match(&R,
            m_c_BinaryOr(m_LogicalAnd(m_VPValue(X), m_VPValue(Y)),
                         m_LogicalAnd(m_VPValue(X1), m_Not(m_VPValue(Y1))))) &&
      X == X1 && Y == Y1) {
    R.getVPSingleValue()->replaceAllUsesWith(X);
    R.eraseFromParent();
    return;
  }

  if (match(&R, m_c_Mul(m_VPValue(A), m_SpecificInt(1))))
    return R.getVPSingleValue()->replaceAllUsesWith(A);

  if (match(&R, m_Not(m_Not(m_VPValue(A)))))
    return R.getVPSingleValue()->replaceAllUsesWith(A);

  // Remove redundant DerviedIVs, that is 0 + A * 1 -> A and 0 + 0 * x -> 0.
  if ((match(&R,
             m_DerivedIV(m_SpecificInt(0), m_VPValue(A), m_SpecificInt(1))) ||
       match(&R,
             m_DerivedIV(m_SpecificInt(0), m_SpecificInt(0), m_VPValue()))) &&
      TypeInfo.inferScalarType(R.getOperand(1)) ==
          TypeInfo.inferScalarType(R.getVPSingleValue()))
    return R.getVPSingleValue()->replaceAllUsesWith(R.getOperand(1));
}

/// Try to simplify the recipes in \p Plan. Use \p CanonicalIVTy as type for all
/// un-typed live-ins in VPTypeAnalysis.
static void simplifyRecipes(VPlan &Plan, Type *CanonicalIVTy) {
  ReversePostOrderTraversal<VPBlockDeepTraversalWrapper<VPBlockBase *>> RPOT(
      Plan.getEntry());
  VPTypeAnalysis TypeInfo(CanonicalIVTy);
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(RPOT)) {
    for (VPRecipeBase &R : make_early_inc_range(*VPBB)) {
      simplifyRecipe(R, TypeInfo);
    }
  }
}

void VPlanTransforms::optimizeForVFAndUF(VPlan &Plan, ElementCount BestVF,
                                         unsigned BestUF,
                                         PredicatedScalarEvolution &PSE) {
  assert(Plan.hasVF(BestVF) && "BestVF is not available in Plan");
  assert(Plan.hasUF(BestUF) && "BestUF is not available in Plan");
  VPRegionBlock *VectorRegion = Plan.getVectorLoopRegion();
  VPBasicBlock *ExitingVPBB = VectorRegion->getExitingBasicBlock();
  auto *Term = &ExitingVPBB->back();
  // Try to simplify the branch condition if TC <= VF * UF when preparing to
  // execute the plan for the main vector loop. We only do this if the
  // terminator is:
  //  1. BranchOnCount, or
  //  2. BranchOnCond where the input is Not(ActiveLaneMask).
  using namespace llvm::VPlanPatternMatch;
  if (!match(Term, m_BranchOnCount(m_VPValue(), m_VPValue())) &&
      !match(Term,
             m_BranchOnCond(m_Not(m_ActiveLaneMask(m_VPValue(), m_VPValue())))))
    return;

  ScalarEvolution &SE = *PSE.getSE();
  const SCEV *TripCount =
      vputils::getSCEVExprForVPValue(Plan.getTripCount(), SE);
  assert(!isa<SCEVCouldNotCompute>(TripCount) &&
         "Trip count SCEV must be computable");
  ElementCount NumElements = BestVF.multiplyCoefficientBy(BestUF);
  const SCEV *C = SE.getElementCount(TripCount->getType(), NumElements);
  if (TripCount->isZero() ||
      !SE.isKnownPredicate(CmpInst::ICMP_ULE, TripCount, C))
    return;

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
    simplifyRecipes(Plan, CanIVTy);
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
  VPlanTransforms::removeDeadRecipes(Plan);

  Plan.setVF(BestVF);
  Plan.setUF(BestUF);
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

    auto *RecurSplice = cast<VPInstruction>(
        LoopBuilder.createNaryOp(VPInstruction::FirstOrderRecurrenceSplice,
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
#ifndef NDEBUG
  // Count the processed recipes and cross check the count later with MinBWs
  // size, to make sure all entries in MinBWs have been handled.
  unsigned NumProcessedRecipes = 0;
#endif
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
               VPWidenSelectRecipe, VPWidenLoadRecipe>(&R))
        continue;

      VPValue *ResultVPV = R.getVPSingleValue();
      auto *UI = cast_or_null<Instruction>(ResultVPV->getUnderlyingValue());
      unsigned NewResSizeInBits = MinBWs.lookup(UI);
      if (!NewResSizeInBits)
        continue;

#ifndef NDEBUG
      NumProcessedRecipes++;
#endif
      // If the value wasn't vectorized, we must maintain the original scalar
      // type. Skip those here, after incrementing NumProcessedRecipes. Also
      // skip casts which do not need to be handled explicitly here, as
      // redundant casts will be removed during recipe simplification.
      if (isa<VPReplicateRecipe, VPWidenCastRecipe>(&R)) {
#ifndef NDEBUG
        // If any of the operands is a live-in and not used by VPWidenRecipe or
        // VPWidenSelectRecipe, but in MinBWs, make sure it is counted as
        // processed as well. When MinBWs is currently constructed, there is no
        // information about whether recipes are widened or replicated and in
        // case they are reciplicated the operands are not truncated. Counting
        // them them here ensures we do not miss any recipes in MinBWs.
        // TODO: Remove once the analysis is done on VPlan.
        for (VPValue *Op : R.operands()) {
          if (!Op->isLiveIn())
            continue;
          auto *UV = dyn_cast_or_null<Instruction>(Op->getUnderlyingValue());
          if (UV && MinBWs.contains(UV) && !ProcessedTruncs.contains(Op) &&
              none_of(Op->users(),
                      IsaPred<VPWidenRecipe, VPWidenSelectRecipe>)) {
            // Add an entry to ProcessedTruncs to avoid counting the same
            // operand multiple times.
            ProcessedTruncs[Op] = nullptr;
            NumProcessedRecipes += 1;
          }
        }
#endif
        continue;
      }

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
      if (isa<VPWidenLoadRecipe>(&R))
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
#ifndef NDEBUG
          auto *OpInst = dyn_cast<Instruction>(Op->getLiveInIRValue());
          bool IsContained = MinBWs.contains(OpInst);
          NumProcessedRecipes += IsContained;
#endif
        }
      }

    }
  }

  assert(MinBWs.size() == NumProcessedRecipes &&
         "some entries in MinBWs haven't been processed");
}

void VPlanTransforms::optimize(VPlan &Plan) {
  removeRedundantCanonicalIVs(Plan);
  removeRedundantInductionCasts(Plan);

  simplifyRecipes(Plan, Plan.getCanonicalIV()->getScalarType());
  removeDeadRecipes(Plan);
  legalizeAndOptimizeInductions(Plan);
  removeRedundantExpandSCEVRecipes(Plan);
  simplifyRecipes(Plan, Plan.getCanonicalIV()->getScalarType());
  removeDeadRecipes(Plan);

  createAndOptimizeReplicateRegions(Plan);
  mergeBlocksIntoPredecessors(Plan);
  licm(Plan);
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
static VPRecipeBase *createEVLRecipe(VPValue *HeaderMask,
                                     VPRecipeBase &CurRecipe,
                                     VPTypeAnalysis &TypeInfo,
                                     VPValue &AllOneMask, VPValue &EVL) {
  using namespace llvm::VPlanPatternMatch;
  auto GetNewMask = [&](VPValue *OrigMask) -> VPValue * {
    assert(OrigMask && "Unmasked recipe when folding tail");
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
      .Case<VPWidenRecipe>([&](VPWidenRecipe *W) -> VPRecipeBase * {
        unsigned Opcode = W->getOpcode();
        if (!Instruction::isBinaryOp(Opcode) && !Instruction::isUnaryOp(Opcode))
          return nullptr;
        return new VPWidenEVLRecipe(*W, EVL);
      })
      .Case<VPReductionRecipe>([&](VPReductionRecipe *Red) {
        VPValue *NewMask = GetNewMask(Red->getCondOp());
        return new VPReductionEVLRecipe(*Red, EVL, NewMask);
      })
      .Case<VPWidenIntrinsicRecipe, VPWidenCastRecipe>(
          [&](auto *CR) -> VPRecipeBase * {
            Intrinsic::ID VPID;
            if (auto *CallR = dyn_cast<VPWidenIntrinsicRecipe>(CR)) {
              VPID =
                  VPIntrinsic::getForIntrinsic(CallR->getVectorIntrinsicID());
            } else {
              auto *CastR = cast<VPWidenCastRecipe>(CR);
              VPID = VPIntrinsic::getForOpcode(CastR->getOpcode());
            }

            // Not all intrinsics have a corresponding VP intrinsic.
            if (VPID == Intrinsic::not_intrinsic)
              return nullptr;
            assert(VPIntrinsic::getMaskParamPos(VPID) &&
                   VPIntrinsic::getVectorLengthParamPos(VPID) &&
                   "Expected VP intrinsic to have mask and EVL");

            SmallVector<VPValue *> Ops(CR->operands());
            Ops.push_back(&AllOneMask);
            Ops.push_back(&EVL);
            return new VPWidenIntrinsicRecipe(
                VPID, Ops, TypeInfo.inferScalarType(CR), CR->getDebugLoc());
          })
      .Case<VPWidenSelectRecipe>([&](VPWidenSelectRecipe *Sel) {
        SmallVector<VPValue *> Ops(Sel->operands());
        Ops.push_back(&EVL);
        return new VPWidenIntrinsicRecipe(Intrinsic::vp_select, Ops,
                                          TypeInfo.inferScalarType(Sel),
                                          Sel->getDebugLoc());
      })
      .Case<VPInstruction>([&](VPInstruction *VPI) -> VPRecipeBase * {
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

  for (VPUser *U : to_vector(Plan.getVF().users())) {
    if (auto *R = dyn_cast<VPReverseVectorPointerRecipe>(U))
      R->setOperand(1, &EVL);
  }

  SmallVector<VPRecipeBase *> ToErase;

  for (VPValue *HeaderMask : collectAllHeaderMasks(Plan)) {
    for (VPUser *U : collectUsersRecursively(HeaderMask)) {
      auto *CurRecipe = cast<VPRecipeBase>(U);
      VPRecipeBase *EVLRecipe =
          createEVLRecipe(HeaderMask, *CurRecipe, TypeInfo, *AllOneMask, EVL);
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
  VPSingleDefRecipe *OpVPEVL = VPEVL;
  if (unsigned IVSize = CanonicalIVPHI->getScalarType()->getScalarSizeInBits();
      IVSize != 32) {
    OpVPEVL = new VPScalarCastRecipe(
        IVSize < 32 ? Instruction::Trunc : Instruction::ZExt, OpVPEVL,
        CanonicalIVPHI->getScalarType(), CanonicalIVIncrement->getDebugLoc());
    OpVPEVL->insertBefore(CanonicalIVIncrement);
  }
  auto *NextEVLIV =
      new VPInstruction(Instruction::Add, {OpVPEVL, EVLPhi},
                        {CanonicalIVIncrement->hasNoUnsignedWrap(),
                         CanonicalIVIncrement->hasNoSignedWrap()},
                        CanonicalIVIncrement->getDebugLoc(), "index.evl.next");
  NextEVLIV->insertBefore(CanonicalIVIncrement);
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
    VPlan &Plan, function_ref<bool(BasicBlock *)> BlockNeedsPredication) {
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
    VPRecipeBuilder &RecipeBuilder, bool ScalarEpilogueAllowed) {
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
                                        InsertPos->getMask(), NeedsMaskForGaps);
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

void VPlanTransforms::convertToConcreteRecipes(VPlan &Plan) {
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(
           vp_depth_first_deep(Plan.getEntry()))) {
    for (VPRecipeBase &R : make_early_inc_range(VPBB->phis())) {
      if (!isa<VPCanonicalIVPHIRecipe, VPEVLBasedIVPHIRecipe>(&R))
        continue;
      auto *PhiR = cast<VPHeaderPHIRecipe>(&R);
      StringRef Name =
          isa<VPCanonicalIVPHIRecipe>(PhiR) ? "index" : "evl.based.iv";
      auto *ScalarR =
          new VPScalarPHIRecipe(PhiR->getStartValue(), PhiR->getBackedgeValue(),
                                PhiR->getDebugLoc(), Name);
      ScalarR->insertBefore(PhiR);
      PhiR->replaceAllUsesWith(ScalarR);
      PhiR->eraseFromParent();
    }
  }
}

void VPlanTransforms::handleUncountableEarlyExit(
    VPlan &Plan, ScalarEvolution &SE, Loop *OrigLoop,
    BasicBlock *UncountableExitingBlock, VPRecipeBuilder &RecipeBuilder) {
  VPRegionBlock *LoopRegion = Plan.getVectorLoopRegion();
  auto *LatchVPBB = cast<VPBasicBlock>(LoopRegion->getExiting());
  VPBuilder Builder(LatchVPBB->getTerminator());
  auto *MiddleVPBB = Plan.getMiddleBlock();
  VPValue *IsEarlyExitTaken = nullptr;

  // Process the uncountable exiting block. Update IsEarlyExitTaken, which
  // tracks if the uncountable early exit has been taken. Also split the middle
  // block and have it conditionally branch to the early exit block if
  // EarlyExitTaken.
  auto *EarlyExitingBranch =
      cast<BranchInst>(UncountableExitingBlock->getTerminator());
  BasicBlock *TrueSucc = EarlyExitingBranch->getSuccessor(0);
  BasicBlock *FalseSucc = EarlyExitingBranch->getSuccessor(1);

  // The early exit block may or may not be the same as the "countable" exit
  // block. Creates a new VPIRBB for the early exit block in case it is distinct
  // from the countable exit block.
  // TODO: Introduce both exit blocks during VPlan skeleton construction.
  VPIRBasicBlock *VPEarlyExitBlock;
  if (OrigLoop->getUniqueExitBlock()) {
    VPEarlyExitBlock = cast<VPIRBasicBlock>(MiddleVPBB->getSuccessors()[0]);
  } else {
    VPEarlyExitBlock = Plan.createVPIRBasicBlock(
        !OrigLoop->contains(TrueSucc) ? TrueSucc : FalseSucc);
  }

  VPValue *EarlyExitNotTakenCond = RecipeBuilder.getBlockInMask(
      OrigLoop->contains(TrueSucc) ? TrueSucc : FalseSucc);
  auto *EarlyExitTakenCond = Builder.createNot(EarlyExitNotTakenCond);
  IsEarlyExitTaken =
      Builder.createNaryOp(VPInstruction::AnyOf, {EarlyExitTakenCond});

  VPBasicBlock *NewMiddle = Plan.createVPBasicBlock("middle.split");
  VPBasicBlock *VectorEarlyExitVPBB =
      Plan.createVPBasicBlock("vector.early.exit");
  VPBlockUtils::insertOnEdge(LoopRegion, MiddleVPBB, NewMiddle);
  VPBlockUtils::connectBlocks(NewMiddle, VectorEarlyExitVPBB);
  NewMiddle->swapSuccessors();

  VPBlockUtils::connectBlocks(VectorEarlyExitVPBB, VPEarlyExitBlock);

  // Update the exit phis in the early exit block.
  VPBuilder MiddleBuilder(NewMiddle);
  VPBuilder EarlyExitB(VectorEarlyExitVPBB);
  for (VPRecipeBase &R : *VPEarlyExitBlock) {
    auto *ExitIRI = cast<VPIRInstruction>(&R);
    auto *ExitPhi = dyn_cast<PHINode>(&ExitIRI->getInstruction());
    if (!ExitPhi)
      break;

    VPValue *IncomingFromEarlyExit = RecipeBuilder.getVPValueOrAddLiveIn(
        ExitPhi->getIncomingValueForBlock(UncountableExitingBlock));

    if (OrigLoop->getUniqueExitBlock()) {
      // If there's a unique exit block, VPEarlyExitBlock has 2 predecessors
      // (MiddleVPBB and NewMiddle). Add the incoming value from MiddleVPBB
      // which is coming from the original latch.
      VPValue *IncomingFromLatch = RecipeBuilder.getVPValueOrAddLiveIn(
          ExitPhi->getIncomingValueForBlock(OrigLoop->getLoopLatch()));
      ExitIRI->addOperand(IncomingFromLatch);
      ExitIRI->extractLastLaneOfOperand(MiddleBuilder);
    }

    // Add the incoming value from the early exit.
    if (!IncomingFromEarlyExit->isLiveIn())
      IncomingFromEarlyExit =
          EarlyExitB.createNaryOp(VPInstruction::ExtractFirstActive,
                                  {IncomingFromEarlyExit, EarlyExitTakenCond});
    ExitIRI->addOperand(IncomingFromEarlyExit);
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
