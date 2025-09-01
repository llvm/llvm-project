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
#include "llvm/Analysis/InstSimplifyFolder.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolutionPatternMatch.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

using namespace llvm;
using namespace VPlanPatternMatch;

cl::opt<bool> EnableWideActiveLaneMask(
    "enable-wide-lane-mask", cl::init(false), cl::Hidden,
    cl::desc("Enable use of wide get active lane mask instructions"));

bool VPlanTransforms::tryToConvertVPInstructionsToVPRecipes(
    VPlanPtr &Plan,
    function_ref<const InductionDescriptor *(PHINode *)>
        GetIntOrFpInductionDescriptor,
    const TargetLibraryInfo &TLI) {

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
      if (auto *PhiR = dyn_cast<VPPhi>(&Ingredient)) {
        auto *Phi = cast<PHINode>(PhiR->getUnderlyingValue());
        const auto *II = GetIntOrFpInductionDescriptor(Phi);
        if (!II) {
          NewRecipe = new VPWidenPHIRecipe(Phi, nullptr, PhiR->getDebugLoc());
          for (VPValue *Op : PhiR->operands())
            NewRecipe->addOperand(Op);
        } else {
          VPValue *Start = Plan->getOrAddLiveIn(II->getStartValue());
          VPValue *Step =
              vputils::getOrCreateVPValueForSCEVExpr(*Plan, II->getStep());
          NewRecipe = new VPWidenIntOrFpInductionRecipe(
              Phi, Start, Step, &Plan->getVF(), *II, Ingredient.getDebugLoc());
        }
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
          WorkList.insert({VPBB, Def});
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
        WorkList.insert({SinkTo, Def});
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
    if (!vputils::onlyScalarValuesUsed(WidenOriginalIV) ||
        vputils::onlyFirstLaneUsed(WidenNewIV)) {
      WidenNewIV->replaceAllUsesWith(WidenOriginalIV);
      WidenNewIV->eraseFromParent();
      return;
    }
  }
}

/// Returns true if \p R is dead and can be removed.
static bool isDeadRecipe(VPRecipeBase &R) {
  // Do remove conditional assume instructions as their conditions may be
  // flattened.
  auto *RepR = dyn_cast<VPReplicateRecipe>(&R);
  bool IsConditionalAssume = RepR && RepR->isPredicated() &&
                             match(RepR, m_Intrinsic<Intrinsic::assume>());
  if (IsConditionalAssume)
    return true;

  if (R.mayHaveSideEffects())
    return false;

  // Recipe is dead if no user keeps the recipe alive.
  return all_of(R.definedValues(),
                [](VPValue *V) { return V->getNumUsers() == 0; });
}

void VPlanTransforms::removeDeadRecipes(VPlan &Plan) {
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(
           vp_post_order_deep(Plan.getEntry()))) {
    // The recipes in the block are processed in reverse order, to catch chains
    // of dead recipes.
    for (VPRecipeBase &R : make_early_inc_range(reverse(*VPBB))) {
      if (isDeadRecipe(R)) {
        R.eraseFromParent();
        continue;
      }

      // Check if R is a dead VPPhi <-> update cycle and remove it.
      auto *PhiR = dyn_cast<VPPhi>(&R);
      if (!PhiR || PhiR->getNumOperands() != 2 || PhiR->getNumUsers() != 1)
        continue;
      VPValue *Incoming = PhiR->getOperand(1);
      if (*PhiR->user_begin() != Incoming->getDefiningRecipe() ||
          Incoming->getNumUsers() != 1)
        continue;
      PhiR->replaceAllUsesWith(PhiR->getOperand(0));
      PhiR->eraseFromParent();
      Incoming->getDefiningRecipe()->eraseFromParent();
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
  VPTypeAnalysis TypeInfo(Plan);
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
static VPWidenInductionRecipe *getOptimizableIVOf(VPValue *VPV,
                                                  ScalarEvolution &SE) {
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
    auto &ID = WideIV->getInductionDescriptor();

    // Check if VPV increments the induction by the induction step.
    VPValue *IVStep = WideIV->getStepValue();
    switch (ID.getInductionOpcode()) {
    case Instruction::Add:
      return match(VPV, m_c_Add(m_Specific(WideIV), m_Specific(IVStep)));
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
      if (!match(VPV, m_Sub(m_VPValue(), m_VPValue(Step))))
        return false;
      const SCEV *IVStepSCEV = vputils::getSCEVExprForVPValue(IVStep, SE);
      const SCEV *StepSCEV = vputils::getSCEVExprForVPValue(Step, SE);
      return !isa<SCEVCouldNotCompute>(IVStepSCEV) &&
             !isa<SCEVCouldNotCompute>(StepSCEV) &&
             IVStepSCEV == SE.getNegativeSCEV(StepSCEV);
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
                                               VPValue *Op,
                                               ScalarEvolution &SE) {
  VPValue *Incoming, *Mask;
  if (!match(Op, m_VPInstruction<VPInstruction::ExtractLane>(
                     m_VPInstruction<VPInstruction::FirstActiveLane>(
                         m_VPValue(Mask)),
                     m_VPValue(Incoming))))
    return nullptr;

  auto *WideIV = getOptimizableIVOf(Incoming, SE);
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
  FirstActiveLane = B.createScalarZExtOrTrunc(FirstActiveLane, CanonicalIVType,
                                              FirstActiveLaneType, DL);
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
static VPValue *optimizeLatchExitInductionUser(
    VPlan &Plan, VPTypeAnalysis &TypeInfo, VPBlockBase *PredVPBB, VPValue *Op,
    DenseMap<VPValue *, VPValue *> &EndValues, ScalarEvolution &SE) {
  VPValue *Incoming;
  if (!match(Op, m_ExtractLastElement(m_VPValue(Incoming))))
    return nullptr;

  auto *WideIV = getOptimizableIVOf(Incoming, SE);
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
    Type *StepTy = TypeInfo.inferScalarType(Step);
    auto *Zero = Plan.getOrAddLiveIn(ConstantInt::get(StepTy, 0));
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
    VPlan &Plan, DenseMap<VPValue *, VPValue *> &EndValues,
    ScalarEvolution &SE) {
  VPBlockBase *MiddleVPBB = Plan.getMiddleBlock();
  VPTypeAnalysis TypeInfo(Plan);
  for (VPIRBasicBlock *ExitVPBB : Plan.getExitBlocks()) {
    for (VPRecipeBase &R : ExitVPBB->phis()) {
      auto *ExitIRI = cast<VPIRPhi>(&R);

      for (auto [Idx, PredVPBB] : enumerate(ExitVPBB->getPredecessors())) {
        VPValue *Escape = nullptr;
        if (PredVPBB == MiddleVPBB)
          Escape = optimizeLatchExitInductionUser(Plan, TypeInfo, PredVPBB,
                                                  ExitIRI->getOperand(Idx),
                                                  EndValues, SE);
        else
          Escape = optimizeEarlyExitInductionUser(Plan, TypeInfo, PredVPBB,
                                                  ExitIRI->getOperand(Idx), SE);
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

    const auto &[V, Inserted] = SCEV2VPV.try_emplace(ExpR->getSCEV(), ExpR);
    if (Inserted)
      continue;
    ExpR->replaceAllUsesWith(V->second);
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

/// Try to fold \p R using InstSimplifyFolder. Will succeed and return a
/// non-nullptr Value for a handled \p Opcode if corresponding \p Operands are
/// foldable live-ins.
static Value *tryToFoldLiveIns(const VPRecipeBase &R, unsigned Opcode,
                               ArrayRef<VPValue *> Operands,
                               const DataLayout &DL, VPTypeAnalysis &TypeInfo) {
  SmallVector<Value *, 4> Ops;
  for (VPValue *Op : Operands) {
    if (!Op->isLiveIn() || !Op->getLiveInIRValue())
      return nullptr;
    Ops.push_back(Op->getLiveInIRValue());
  }

  InstSimplifyFolder Folder(DL);
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
  case VPInstruction::WidePtrAdd:
    return Folder.FoldGEP(IntegerType::getInt8Ty(TypeInfo.getContext()), Ops[0],
                          Ops[1],
                          cast<VPRecipeWithIRFlags>(R).getGEPNoWrapFlags());
  // An extract of a live-in is an extract of a broadcast, so return the
  // broadcasted element.
  case Instruction::ExtractElement:
    assert(!Ops[0]->getType()->isVectorTy() && "Live-ins should be scalar");
    return Ops[0];
  }
  return nullptr;
}

/// Try to simplify recipe \p R.
static void simplifyRecipe(VPRecipeBase &R, VPTypeAnalysis &TypeInfo) {
  VPlan *Plan = R.getParent()->getPlan();

  auto *Def = dyn_cast<VPSingleDefRecipe>(&R);
  if (!Def)
    return;

  // Simplification of live-in IR values for SingleDef recipes using
  // InstSimplifyFolder.
  if (TypeSwitch<VPRecipeBase *, bool>(&R)
          .Case<VPInstruction, VPWidenRecipe, VPWidenCastRecipe,
                VPReplicateRecipe, VPWidenSelectRecipe>([&](auto *I) {
            const DataLayout &DL =
                Plan->getScalarHeader()->getIRBasicBlock()->getDataLayout();
            Value *V = tryToFoldLiveIns(*I, I->getOpcode(), I->operands(), DL,
                                        TypeInfo);
            if (V)
              I->replaceAllUsesWith(Plan->getOrAddLiveIn(V));
            return V;
          })
          .Default([](auto *) { return false; }))
    return;

  // Fold PredPHI LiveIn -> LiveIn.
  if (auto *PredPHI = dyn_cast<VPPredInstPHIRecipe>(&R)) {
    VPValue *Op = PredPHI->getOperand(0);
    if (Op->isLiveIn())
      PredPHI->replaceAllUsesWith(Op);
  }

  VPValue *A;
  if (match(Def, m_Trunc(m_ZExtOrSExt(m_VPValue(A))))) {
    Type *TruncTy = TypeInfo.inferScalarType(Def);
    Type *ATy = TypeInfo.inferScalarType(A);
    if (TruncTy == ATy) {
      Def->replaceAllUsesWith(A);
    } else {
      // Don't replace a scalarizing recipe with a widened cast.
      if (isa<VPReplicateRecipe>(Def))
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
        Def->replaceAllUsesWith(VPC);
      } else if (ATy->getScalarSizeInBits() > TruncTy->getScalarSizeInBits()) {
        auto *VPC = new VPWidenCastRecipe(Instruction::Trunc, A, TruncTy);
        VPC->insertBefore(&R);
        Def->replaceAllUsesWith(VPC);
      }
    }
#ifndef NDEBUG
    // Verify that the cached type info is for both A and its users is still
    // accurate by comparing it to freshly computed types.
    VPTypeAnalysis TypeInfo2(*Plan);
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
  VPValue *X, *Y, *Z;
  if (match(Def,
            m_c_BinaryOr(m_LogicalAnd(m_VPValue(X), m_VPValue(Y)),
                         m_LogicalAnd(m_Deferred(X), m_Not(m_Deferred(Y)))))) {
    Def->replaceAllUsesWith(X);
    Def->eraseFromParent();
    return;
  }

  // OR x, 1 -> 1.
  if (match(Def, m_c_BinaryOr(m_VPValue(X), m_AllOnes()))) {
    Def->replaceAllUsesWith(Def->getOperand(0) == X ? Def->getOperand(1)
                                                    : Def->getOperand(0));
    Def->eraseFromParent();
    return;
  }

  // AND x, 0 -> 0
  if (match(&R, m_c_BinaryAnd(m_VPValue(X), m_ZeroInt()))) {
    Def->replaceAllUsesWith(R.getOperand(0) == X ? R.getOperand(1)
                                                 : R.getOperand(0));
    return;
  }

  // (x && y) || (x && z) -> x && (y || z)
  VPBuilder Builder(Def);
  if (match(Def, m_c_BinaryOr(m_LogicalAnd(m_VPValue(X), m_VPValue(Y)),
                              m_LogicalAnd(m_Deferred(X), m_VPValue(Z)))) &&
      // Simplify only if one of the operands has one use to avoid creating an
      // extra recipe.
      (!Def->getOperand(0)->hasMoreThanOneUniqueUser() ||
       !Def->getOperand(1)->hasMoreThanOneUniqueUser()))
    return Def->replaceAllUsesWith(
        Builder.createLogicalAnd(X, Builder.createOr(Y, Z)));

  if (match(Def, m_Select(m_VPValue(), m_VPValue(X), m_Deferred(X))))
    return Def->replaceAllUsesWith(X);

  // select !c, x, y -> select c, y, x
  VPValue *C;
  if (match(Def, m_Select(m_Not(m_VPValue(C)), m_VPValue(X), m_VPValue(Y)))) {
    Def->setOperand(0, C);
    Def->setOperand(1, Y);
    Def->setOperand(2, X);
    return;
  }

  if (match(Def, m_c_Mul(m_VPValue(A), m_SpecificInt(1))))
    return Def->replaceAllUsesWith(A);

  if (match(Def, m_c_Mul(m_VPValue(A), m_SpecificInt(0))))
    return Def->replaceAllUsesWith(R.getOperand(0) == A ? R.getOperand(1)
                                                        : R.getOperand(0));

  if (match(Def, m_Not(m_VPValue(A)))) {
    if (match(A, m_Not(m_VPValue(A))))
      return Def->replaceAllUsesWith(A);

    // Try to fold Not into compares by adjusting the predicate in-place.
    CmpPredicate Pred;
    if (match(A, m_Cmp(Pred, m_VPValue(), m_VPValue()))) {
      auto *Cmp = cast<VPRecipeWithIRFlags>(A);
      if (all_of(Cmp->users(), [&Cmp](VPUser *U) {
            return match(U, m_CombineOr(m_Not(m_Specific(Cmp)),
                                        m_Select(m_Specific(Cmp), m_VPValue(),
                                                 m_VPValue())));
          })) {
        Cmp->setPredicate(CmpInst::getInversePredicate(Pred));
        for (VPUser *U : to_vector(Cmp->users())) {
          auto *R = cast<VPSingleDefRecipe>(U);
          if (match(R, m_Select(m_Specific(Cmp), m_VPValue(X), m_VPValue(Y)))) {
            // select (cmp pred), x, y -> select (cmp inv_pred), y, x
            R->setOperand(1, Y);
            R->setOperand(2, X);
          } else {
            // not (cmp pred) -> cmp inv_pred
            assert(match(R, m_Not(m_Specific(Cmp))) && "Unexpected user");
            R->replaceAllUsesWith(Cmp);
          }
        }
        // If Cmp doesn't have a debug location, use the one from the negation,
        // to preserve the location.
        if (!Cmp->getDebugLoc() && R.getDebugLoc())
          Cmp->setDebugLoc(R.getDebugLoc());
      }
    }
  }

  // Remove redundant DerviedIVs, that is 0 + A * 1 -> A and 0 + 0 * x -> 0.
  if ((match(Def,
             m_DerivedIV(m_SpecificInt(0), m_VPValue(A), m_SpecificInt(1))) ||
       match(Def,
             m_DerivedIV(m_SpecificInt(0), m_SpecificInt(0), m_VPValue()))) &&
      TypeInfo.inferScalarType(Def->getOperand(1)) ==
          TypeInfo.inferScalarType(Def))
    return Def->replaceAllUsesWith(Def->getOperand(1));

  if (match(Def, m_VPInstruction<VPInstruction::WideIVStep>(
                     m_VPValue(X), m_SpecificInt(1)))) {
    Type *WideStepTy = TypeInfo.inferScalarType(Def);
    if (TypeInfo.inferScalarType(X) != WideStepTy)
      X = Builder.createWidenCast(Instruction::Trunc, X, WideStepTy);
    Def->replaceAllUsesWith(X);
    return;
  }

  // For i1 vp.merges produced by AnyOf reductions:
  // vp.merge true, (or x, y), x, evl -> vp.merge y, true, x, evl
  if (match(Def, m_Intrinsic<Intrinsic::vp_merge>(m_True(), m_VPValue(A),
                                                  m_VPValue(X), m_VPValue())) &&
      match(A, m_c_BinaryOr(m_Specific(X), m_VPValue(Y))) &&
      TypeInfo.inferScalarType(R.getVPSingleValue())->isIntegerTy(1)) {
    Def->setOperand(1, Def->getOperand(0));
    Def->setOperand(0, Y);
    return;
  }

  if (auto *Phi = dyn_cast<VPFirstOrderRecurrencePHIRecipe>(Def)) {
    if (Phi->getOperand(0) == Phi->getOperand(1))
      Def->replaceAllUsesWith(Phi->getOperand(0));
    return;
  }

  // Look through ExtractLastElement (BuildVector ....).
  if (match(&R, m_ExtractLastElement(m_BuildVector()))) {
    auto *BuildVector = cast<VPInstruction>(R.getOperand(0));
    Def->replaceAllUsesWith(
        BuildVector->getOperand(BuildVector->getNumOperands() - 1));
    return;
  }

  // Look through ExtractPenultimateElement (BuildVector ....).
  if (match(&R, m_VPInstruction<VPInstruction::ExtractPenultimateElement>(
                    m_BuildVector()))) {
    auto *BuildVector = cast<VPInstruction>(R.getOperand(0));
    Def->replaceAllUsesWith(
        BuildVector->getOperand(BuildVector->getNumOperands() - 2));
    return;
  }

  if (auto *Phi = dyn_cast<VPPhi>(Def)) {
    if (Phi->getNumOperands() == 1)
      Phi->replaceAllUsesWith(Phi->getOperand(0));
    return;
  }

  // Some simplifications can only be applied after unrolling. Perform them
  // below.
  if (!Plan->isUnrolled())
    return;

  // VPVectorPointer for part 0 can be replaced by their start pointer.
  if (auto *VecPtr = dyn_cast<VPVectorPointerRecipe>(&R)) {
    if (VecPtr->isFirstPart()) {
      VecPtr->replaceAllUsesWith(VecPtr->getOperand(0));
      return;
    }
  }

  // VPScalarIVSteps for part 0 can be replaced by their start value, if only
  // the first lane is demanded.
  if (auto *Steps = dyn_cast<VPScalarIVStepsRecipe>(Def)) {
    if (Steps->isPart0() && vputils::onlyFirstLaneUsed(Steps)) {
      Steps->replaceAllUsesWith(Steps->getOperand(0));
      return;
    }
  }
  // Simplify redundant ReductionStartVector recipes after unrolling.
  VPValue *StartV;
  if (match(Def, m_VPInstruction<VPInstruction::ReductionStartVector>(
                     m_VPValue(StartV), m_VPValue(), m_VPValue()))) {
    Def->replaceUsesWithIf(StartV, [](const VPUser &U, unsigned Idx) {
      auto *PhiR = dyn_cast<VPReductionPHIRecipe>(&U);
      return PhiR && PhiR->isInLoop();
    });
    return;
  }

  if (match(Def, m_ExtractLastElement(m_Broadcast(m_VPValue(A))))) {
    Def->replaceAllUsesWith(A);
    return;
  }

  VPInstruction *OpVPI;
  if (match(Def, m_ExtractLastElement(m_VPInstruction(OpVPI))) &&
      OpVPI->isVectorToScalar()) {
    Def->replaceAllUsesWith(OpVPI);
    return;
  }
}

void VPlanTransforms::simplifyRecipes(VPlan &Plan) {
  ReversePostOrderTraversal<VPBlockDeepTraversalWrapper<VPBlockBase *>> RPOT(
      Plan.getEntry());
  VPTypeAnalysis TypeInfo(Plan);
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
      if (!isa<VPWidenRecipe, VPWidenSelectRecipe, VPReplicateRecipe>(&R))
        continue;
      auto *RepR = dyn_cast<VPReplicateRecipe>(&R);
      if (RepR && (RepR->isSingleScalar() || RepR->isPredicated()))
        continue;

      auto *RepOrWidenR = cast<VPSingleDefRecipe>(&R);
      // Skip recipes that aren't single scalars or don't have only their
      // scalar results used. In the latter case, we would introduce extra
      // broadcasts.
      if (!vputils::isSingleScalar(RepOrWidenR) ||
          !vputils::onlyScalarValuesUsed(RepOrWidenR))
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

      auto *NewBlend =
          new VPBlendRecipe(cast_or_null<PHINode>(Blend->getUnderlyingValue()),
                            OperandsWithMask, Blend->getDebugLoc());
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

  LLVMContext &Ctx = Plan.getContext();
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
    if (!match(*WideIV->user_begin(),
               m_ICmp(m_Specific(WideIV),
                      m_Broadcast(
                          m_Specific(Plan.getOrCreateBackedgeTakenCount())))))
      continue;

    // Update IV operands and comparison bound to use new narrower type.
    auto *NewStart = Plan.getOrAddLiveIn(ConstantInt::get(NewIVTy, 0));
    WideIV->setStartValue(NewStart);
    auto *NewStep = Plan.getOrAddLiveIn(ConstantInt::get(NewIVTy, 1));
    WideIV->setStepValue(NewStep);

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
  if (match(Cond, m_BinaryOr(m_VPValue(), m_VPValue())))
    return any_of(Cond->getDefiningRecipe()->operands(), [&Plan, BestVF, BestUF,
                                                          &SE](VPValue *C) {
      return isConditionTrueViaVFAndUF(C, Plan, BestVF, BestUF, SE);
    });

  auto *CanIV = Plan.getCanonicalIV();
  if (!match(Cond, m_SpecificICmp(CmpInst::ICMP_EQ,
                                  m_Specific(CanIV->getBackedgeValue()),
                                  m_Specific(&Plan.getVectorTripCount()))))
    return false;

  // The compare checks CanIV + VFxUF == vector trip count. The vector trip
  // count is not conveniently available as SCEV so far, so we compare directly
  // against the original trip count. This is stricter than necessary, as we
  // will only return true if the trip count == vector trip count.
  const SCEV *VectorTripCount =
      vputils::getSCEVExprForVPValue(&Plan.getVectorTripCount(), SE);
  if (isa<SCEVCouldNotCompute>(VectorTripCount))
    VectorTripCount = vputils::getSCEVExprForVPValue(Plan.getTripCount(), SE);
  assert(!isa<SCEVCouldNotCompute>(VectorTripCount) &&
         "Trip count SCEV must be computable");
  ElementCount NumElements = BestVF.multiplyCoefficientBy(BestUF);
  const SCEV *C = SE.getElementCount(VectorTripCount->getType(), NumElements);
  return SE.isKnownPredicate(CmpInst::ICMP_EQ, VectorTripCount, C);
}

/// Try to replace multiple active lane masks used for control flow with
/// a single, wide active lane mask instruction followed by multiple
/// extract subvector intrinsics. This applies to the active lane mask
/// instructions both in the loop and in the preheader.
/// Incoming values of all ActiveLaneMaskPHIs are updated to use the
/// new extracts from the first active lane mask, which has it's last
/// operand (multiplier) set to UF.
static bool tryToReplaceALMWithWideALM(VPlan &Plan, ElementCount VF,
                                       unsigned UF) {
  if (!EnableWideActiveLaneMask || !VF.isVector() || UF == 1)
    return false;

  VPRegionBlock *VectorRegion = Plan.getVectorLoopRegion();
  VPBasicBlock *ExitingVPBB = VectorRegion->getExitingBasicBlock();
  auto *Term = &ExitingVPBB->back();

  using namespace llvm::VPlanPatternMatch;
  if (!match(Term, m_BranchOnCond(m_Not(m_ActiveLaneMask(
                       m_VPValue(), m_VPValue(), m_VPValue())))))
    return false;

  auto *Header = cast<VPBasicBlock>(VectorRegion->getEntry());
  LLVMContext &Ctx = Plan.getContext();

  auto ExtractFromALM = [&](VPInstruction *ALM,
                            SmallVectorImpl<VPValue *> &Extracts) {
    DebugLoc DL = ALM->getDebugLoc();
    for (unsigned Part = 0; Part < UF; ++Part) {
      SmallVector<VPValue *> Ops;
      Ops.append({ALM, Plan.getOrAddLiveIn(
                           ConstantInt::get(IntegerType::getInt64Ty(Ctx),
                                            VF.getKnownMinValue() * Part))});
      auto *Ext = new VPWidenIntrinsicRecipe(Intrinsic::vector_extract, Ops,
                                             IntegerType::getInt1Ty(Ctx), DL);
      Extracts[Part] = Ext;
      Ext->insertAfter(ALM);
    }
  };

  // Create a list of each active lane mask phi, ordered by unroll part.
  SmallVector<VPActiveLaneMaskPHIRecipe *> Phis(UF, nullptr);
  for (VPRecipeBase &R : Header->phis()) {
    auto *Phi = dyn_cast<VPActiveLaneMaskPHIRecipe>(&R);
    if (!Phi)
      continue;
    VPValue *Index = nullptr;
    match(Phi->getBackedgeValue(),
          m_ActiveLaneMask(m_VPValue(Index), m_VPValue(), m_VPValue()));
    assert(Index && "Expected index from ActiveLaneMask instruction");

    auto *II = dyn_cast<VPInstruction>(Index);
    if (II && II->getOpcode() == VPInstruction::CanonicalIVIncrementForPart) {
      auto Part = cast<ConstantInt>(II->getOperand(1)->getLiveInIRValue());
      Phis[Part->getZExtValue()] = Phi;
    } else
      // Anything other than a CanonicalIVIncrementForPart is part 0
      Phis[0] = Phi;
  }

  assert(all_of(Phis, [](VPActiveLaneMaskPHIRecipe *Phi) { return Phi; }) &&
         "Expected one VPActiveLaneMaskPHIRecipe for each unroll part");

  auto *EntryALM = cast<VPInstruction>(Phis[0]->getStartValue());
  auto *LoopALM = cast<VPInstruction>(Phis[0]->getBackedgeValue());

  assert((EntryALM->getOpcode() == VPInstruction::ActiveLaneMask &&
          LoopALM->getOpcode() == VPInstruction::ActiveLaneMask) &&
         "Expected incoming values of Phi to be ActiveLaneMasks");

  // When using wide lane masks, the return type of the get.active.lane.mask
  // intrinsic is VF x UF (last operand).
  VPValue *ALMMultiplier =
      Plan.getOrAddLiveIn(ConstantInt::get(IntegerType::getInt64Ty(Ctx), UF));
  EntryALM->setOperand(2, ALMMultiplier);
  LoopALM->setOperand(2, ALMMultiplier);

  // Create UF x extract vectors and insert into preheader.
  SmallVector<VPValue *> EntryExtracts(UF);
  ExtractFromALM(EntryALM, EntryExtracts);

  // Create UF x extract vectors and insert before the loop compare & branch,
  // updating the compare to use the first extract.
  SmallVector<VPValue *> LoopExtracts(UF);
  ExtractFromALM(LoopALM, LoopExtracts);
  VPInstruction *Not = cast<VPInstruction>(Term->getOperand(0));
  Not->setOperand(0, LoopExtracts[0]);

  // Update the incoming values of active lane mask phis.
  for (unsigned Part = 0; Part < UF; ++Part) {
    Phis[Part]->setStartValue(EntryExtracts[Part]);
    Phis[Part]->setBackedgeValue(LoopExtracts[Part]);
  }

  return true;
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
  if (match(Term, m_BranchOnCount(m_VPValue(), m_VPValue())) ||
      match(Term, m_BranchOnCond(m_Not(m_ActiveLaneMask(
                      m_VPValue(), m_VPValue(), m_VPValue()))))) {
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
  // TODO: VPWidenIntOrFpInductionRecipe is only partially supported; add
  // support for other non-canonical widen induction recipes (e.g.,
  // VPWidenPointerInductionRecipe).
  auto *Header = cast<VPBasicBlock>(VectorRegion->getEntry());
  if (all_of(Header->phis(), [](VPRecipeBase &Phi) {
        if (auto *R = dyn_cast<VPWidenIntOrFpInductionRecipe>(&Phi))
          return R->isCanonical();
        return isa<VPCanonicalIVPHIRecipe, VPEVLBasedIVPHIRecipe,
                   VPFirstOrderRecurrencePHIRecipe, VPPhi>(&Phi);
      })) {
    for (VPRecipeBase &HeaderR : make_early_inc_range(Header->phis())) {
      if (auto *R = dyn_cast<VPWidenIntOrFpInductionRecipe>(&HeaderR)) {
        VPBuilder Builder(Plan.getVectorPreheader());
        VPValue *StepV = Builder.createNaryOp(VPInstruction::StepVector, {},
                                              R->getScalarType());
        HeaderR.getVPSingleValue()->replaceAllUsesWith(StepV);
        HeaderR.eraseFromParent();
        continue;
      }
      auto *Phi = cast<VPPhiAccessors>(&HeaderR);
      HeaderR.getVPSingleValue()->replaceAllUsesWith(Phi->getIncomingValue(0));
      HeaderR.eraseFromParent();
    }

    VPBlockBase *Preheader = VectorRegion->getSinglePredecessor();
    VPBlockBase *Exit = VectorRegion->getSingleSuccessor();
    VPBlockUtils::disconnectBlocks(Preheader, VectorRegion);
    VPBlockUtils::disconnectBlocks(VectorRegion, Exit);

    for (VPBlockBase *B : vp_depth_first_shallow(VectorRegion->getEntry()))
      B->setParent(nullptr);

    VPBlockUtils::connectBlocks(Preheader, Header);
    VPBlockUtils::connectBlocks(ExitingVPBB, Exit);
    VPlanTransforms::simplifyRecipes(Plan);
  } else {
    // The vector region contains header phis for which we cannot remove the
    // loop region yet.
    auto *BOC = new VPInstruction(VPInstruction::BranchOnCond, {Plan.getTrue()},
                                  Term->getDebugLoc());
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

  bool MadeChange = tryToReplaceALMWithWideALM(Plan, BestVF, BestUF);
  MadeChange |= simplifyBranchConditionForVFAndUF(Plan, BestVF, BestUF, PSE);
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
    RecurKind RK = PhiR->getRecurrenceKind();
    if (RK != RecurKind::Add && RK != RecurKind::Mul && RK != RecurKind::Sub &&
        RK != RecurKind::AddChainWithSubs)
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
  VPTypeAnalysis TypeInfo(Plan);
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

      auto *NewResTy = IntegerType::get(Plan.getContext(), NewResSizeInBits);

      // Any wrapping introduced by shrinking this operation shouldn't be
      // considered undefined behavior. So, we can't unconditionally copy
      // arithmetic wrapping flags to VPW.
      if (auto *VPW = dyn_cast<VPRecipeWithIRFlags>(&R))
        VPW->dropPoisonGeneratingFlags();

      if (OldResSizeInBits != NewResSizeInBits &&
          !match(&R, m_ICmp(m_VPValue(), m_VPValue()))) {
        // Extend result to original width.
        auto *Ext =
            new VPWidenCastRecipe(Instruction::ZExt, ResultVPV, OldResTy);
        Ext->insertAfter(&R);
        ResultVPV->replaceAllUsesWith(Ext);
        Ext->setOperand(0, ResultVPV);
        assert(OldResSizeInBits > NewResSizeInBits && "Nothing to shrink?");
      } else {
        assert(match(&R, m_ICmp(m_VPValue(), m_VPValue())) &&
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
        auto [ProcessedIter, IterIsEmpty] = ProcessedTruncs.try_emplace(Op);
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

void VPlanTransforms::removeBranchOnConst(VPlan &Plan) {
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(
           vp_depth_first_shallow(Plan.getEntry()))) {
    VPValue *Cond;
    if (VPBB->getNumSuccessors() != 2 || VPBB == Plan.getEntry() ||
        !match(&VPBB->back(), m_BranchOnCond(m_VPValue(Cond))))
      continue;

    unsigned RemovedIdx;
    if (match(Cond, m_True()))
      RemovedIdx = 1;
    else if (match(Cond, m_False()))
      RemovedIdx = 0;
    else
      continue;

    VPBasicBlock *RemovedSucc =
        cast<VPBasicBlock>(VPBB->getSuccessors()[RemovedIdx]);
    assert(count(RemovedSucc->getPredecessors(), VPBB) == 1 &&
           "There must be a single edge between VPBB and its successor");
    // Values coming from VPBB into phi recipes of RemoveSucc are removed from
    // these recipes.
    for (VPRecipeBase &R : RemovedSucc->phis())
      cast<VPPhiAccessors>(&R)->removeIncomingValueFor(VPBB);

    // Disconnect blocks and remove the terminator. RemovedSucc will be deleted
    // automatically on VPlan destruction if it becomes unreachable.
    VPBlockUtils::disconnectBlocks(VPBB, RemovedSucc);
    VPBB->back().eraseFromParent();
  }
}

void VPlanTransforms::optimize(VPlan &Plan) {
  runPass(removeRedundantCanonicalIVs, Plan);
  runPass(removeRedundantInductionCasts, Plan);

  runPass(simplifyRecipes, Plan);
  runPass(simplifyBlends, Plan);
  runPass(removeDeadRecipes, Plan);
  runPass(narrowToSingleScalarRecipes, Plan);
  runPass(legalizeAndOptimizeInductions, Plan);
  runPass(removeRedundantExpandSCEVRecipes, Plan);
  runPass(simplifyRecipes, Plan);
  runPass(removeBranchOnConst, Plan);
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
  VPValue *ALMMultiplier = Plan.getOrAddLiveIn(
      ConstantInt::get(Plan.getCanonicalIV()->getScalarType(), 1));
  auto *EntryALM = Builder.createNaryOp(VPInstruction::ActiveLaneMask,
                                        {EntryIncrement, TC, ALMMultiplier}, DL,
                                        "active.lane.mask.entry");

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
                                   {InLoopIncrement, TripCount, ALMMultiplier},
                                   DL, "active.lane.mask.next");
  LaneMaskPhi->addOperand(ALM);

  // Replace the original terminator with BranchOnCond. We have to invert the
  // mask here because a true condition means jumping to the exit block.
  auto *NotMask = Builder.createNot(ALM, DL);
  Builder.createNaryOp(VPInstruction::BranchOnCond, {NotMask}, DL);
  OriginalTerminator->eraseFromParent();
  return LaneMaskPhi;
}

/// Collect the header mask with the pattern:
///   (ICMP_ULE, WideCanonicalIV, backedge-taken-count)
/// TODO: Introduce explicit recipe for header-mask instead of searching
/// for the header-mask pattern manually.
static VPSingleDefRecipe *findHeaderMask(VPlan &Plan) {
  SmallVector<VPValue *> WideCanonicalIVs;
  auto *FoundWidenCanonicalIVUser = find_if(Plan.getCanonicalIV()->users(),
                                            IsaPred<VPWidenCanonicalIVRecipe>);
  assert(count_if(Plan.getCanonicalIV()->users(),
                  IsaPred<VPWidenCanonicalIVRecipe>) <= 1 &&
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

  // Walk users of wide canonical IVs and find the single compare of the form
  // (ICMP_ULE, WideCanonicalIV, backedge-taken-count).
  VPSingleDefRecipe *HeaderMask = nullptr;
  for (auto *Wide : WideCanonicalIVs) {
    for (VPUser *U : SmallVector<VPUser *>(Wide->users())) {
      auto *VPI = dyn_cast<VPInstruction>(U);
      if (!VPI || !vputils::isHeaderMask(VPI, Plan))
        continue;

      assert(VPI->getOperand(0) == Wide &&
             "WidenCanonicalIV must be the first operand of the compare");
      assert(!HeaderMask && "Multiple header masks found?");
      HeaderMask = VPI;
    }
  }
  return HeaderMask;
}

void VPlanTransforms::addActiveLaneMask(
    VPlan &Plan, bool UseActiveLaneMaskForControlFlow,
    bool DataAndControlFlowWithoutRuntimeCheck) {
  assert((!DataAndControlFlowWithoutRuntimeCheck ||
          UseActiveLaneMaskForControlFlow) &&
         "DataAndControlFlowWithoutRuntimeCheck implies "
         "UseActiveLaneMaskForControlFlow");

  auto *FoundWidenCanonicalIVUser = find_if(Plan.getCanonicalIV()->users(),
                                            IsaPred<VPWidenCanonicalIVRecipe>);
  assert(FoundWidenCanonicalIVUser &&
         "Must have widened canonical IV when tail folding!");
  VPSingleDefRecipe *HeaderMask = findHeaderMask(Plan);
  auto *WideCanonicalIV =
      cast<VPWidenCanonicalIVRecipe>(*FoundWidenCanonicalIVUser);
  VPSingleDefRecipe *LaneMask;
  if (UseActiveLaneMaskForControlFlow) {
    LaneMask = addVPLaneMaskPhiAndUpdateExitBranch(
        Plan, DataAndControlFlowWithoutRuntimeCheck);
  } else {
    VPBuilder B = VPBuilder::getToInsertAfter(WideCanonicalIV);
    VPValue *ALMMultiplier = Plan.getOrAddLiveIn(
        ConstantInt::get(Plan.getCanonicalIV()->getScalarType(), 1));
    LaneMask =
        B.createNaryOp(VPInstruction::ActiveLaneMask,
                       {WideCanonicalIV, Plan.getTripCount(), ALMMultiplier},
                       nullptr, "active.lane.mask");
  }

  // Walk users of WideCanonicalIV and replace the header mask of the form
  // (ICMP_ULE, WideCanonicalIV, backedge-taken-count) with an active-lane-mask,
  // removing the old one to ensure there is always only a single header mask.
  HeaderMask->replaceAllUsesWith(LaneMask);
  HeaderMask->eraseFromParent();
}

/// Try to optimize a \p CurRecipe masked by \p HeaderMask to a corresponding
/// EVL-based recipe without the header mask. Returns nullptr if no EVL-based
/// recipe could be created.
/// \p HeaderMask  Header Mask.
/// \p CurRecipe   Recipe to be transform.
/// \p TypeInfo    VPlan-based type analysis.
/// \p AllOneMask  The vector mask parameter of vector-predication intrinsics.
/// \p EVL         The explicit vector length parameter of vector-predication
/// intrinsics.
static VPRecipeBase *optimizeMaskToEVL(VPValue *HeaderMask,
                                       VPRecipeBase &CurRecipe,
                                       VPTypeAnalysis &TypeInfo,
                                       VPValue &AllOneMask, VPValue &EVL) {
  // FIXME: Don't transform recipes to EVL recipes if they're not masked by the
  // header mask.
  auto GetNewMask = [&](VPValue *OrigMask) -> VPValue * {
    assert(OrigMask && "Unmasked recipe when folding tail");
    // HeaderMask will be handled using EVL.
    VPValue *Mask;
    if (match(OrigMask, m_LogicalAnd(m_Specific(HeaderMask), m_VPValue(Mask))))
      return Mask;
    return HeaderMask == OrigMask ? nullptr : OrigMask;
  };

  /// Adjust any end pointers so that they point to the end of EVL lanes not VF.
  auto GetNewAddr = [&CurRecipe, &EVL](VPValue *Addr) -> VPValue * {
    auto *EndPtr = dyn_cast<VPVectorEndPointerRecipe>(Addr);
    if (!EndPtr)
      return Addr;
    assert(EndPtr->getOperand(1) == &EndPtr->getParent()->getPlan()->getVF() &&
           "VPVectorEndPointerRecipe with non-VF VF operand?");
    assert(
        all_of(EndPtr->users(),
               [](VPUser *U) {
                 return cast<VPWidenMemoryRecipe>(U)->isReverse();
               }) &&
        "VPVectorEndPointRecipe not used by reversed widened memory recipe?");
    VPVectorEndPointerRecipe *EVLAddr = EndPtr->clone();
    EVLAddr->insertBefore(&CurRecipe);
    EVLAddr->setOperand(1, &EVL);
    return EVLAddr;
  };

  return TypeSwitch<VPRecipeBase *, VPRecipeBase *>(&CurRecipe)
      .Case<VPWidenLoadRecipe>([&](VPWidenLoadRecipe *L) {
        VPValue *NewMask = GetNewMask(L->getMask());
        VPValue *NewAddr = GetNewAddr(L->getAddr());
        return new VPWidenLoadEVLRecipe(*L, NewAddr, EVL, NewMask);
      })
      .Case<VPWidenStoreRecipe>([&](VPWidenStoreRecipe *S) {
        VPValue *NewMask = GetNewMask(S->getMask());
        VPValue *NewAddr = GetNewAddr(S->getAddr());
        return new VPWidenStoreEVLRecipe(*S, NewAddr, EVL, NewMask);
      })
      .Case<VPReductionRecipe>([&](VPReductionRecipe *Red) {
        VPValue *NewMask = GetNewMask(Red->getCondOp());
        return new VPReductionEVLRecipe(*Red, EVL, NewMask);
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
  VPTypeAnalysis TypeInfo(Plan);
  VPValue *AllOneMask = Plan.getTrue();
  VPRegionBlock *LoopRegion = Plan.getVectorLoopRegion();
  VPBasicBlock *Header = LoopRegion->getEntryBasicBlock();

  assert(all_of(Plan.getVF().users(),
                IsaPred<VPVectorEndPointerRecipe, VPScalarIVStepsRecipe,
                        VPWidenIntOrFpInductionRecipe>) &&
         "User of VF that we can't transform to EVL.");
  Plan.getVF().replaceUsesWithIf(&EVL, [](VPUser &U, unsigned Idx) {
    return isa<VPWidenIntOrFpInductionRecipe, VPScalarIVStepsRecipe>(U);
  });

  assert(all_of(Plan.getVFxUF().users(),
                [&Plan](VPUser *U) {
                  return match(U, m_c_Add(m_Specific(Plan.getCanonicalIV()),
                                          m_Specific(&Plan.getVFxUF()))) ||
                         isa<VPWidenPointerInductionRecipe>(U);
                }) &&
         "Only users of VFxUF should be VPWidenPointerInductionRecipe and the "
         "increment of the canonical induction.");
  Plan.getVFxUF().replaceUsesWithIf(&EVL, [](VPUser &U, unsigned Idx) {
    // Only replace uses in VPWidenPointerInductionRecipe; The increment of the
    // canonical induction must not be updated.
    return isa<VPWidenPointerInductionRecipe>(U);
  });

  // Defer erasing recipes till the end so that we don't invalidate the
  // VPTypeAnalysis cache.
  SmallVector<VPRecipeBase *> ToErase;

  // Create a scalar phi to track the previous EVL if fixed-order recurrence is
  // contained.
  bool ContainsFORs =
      any_of(Header->phis(), IsaPred<VPFirstOrderRecurrencePHIRecipe>);
  if (ContainsFORs) {
    // TODO: Use VPInstruction::ExplicitVectorLength to get maximum EVL.
    VPValue *MaxEVL = &Plan.getVF();
    // Emit VPScalarCastRecipe in preheader if VF is not a 32 bits integer.
    VPBuilder Builder(LoopRegion->getPreheaderVPBB());
    MaxEVL = Builder.createScalarZExtOrTrunc(
        MaxEVL, Type::getInt32Ty(Plan.getContext()),
        TypeInfo.inferScalarType(MaxEVL), DebugLoc());

    Builder.setInsertPoint(Header, Header->getFirstNonPhi());
    VPValue *PrevEVL =
        Builder.createScalarPhi({MaxEVL, &EVL}, DebugLoc(), "prev.evl");

    for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(
             vp_depth_first_deep(Plan.getVectorLoopRegion()->getEntry()))) {
      for (VPRecipeBase &R : *VPBB) {
        VPValue *V1, *V2;
        if (!match(&R,
                   m_VPInstruction<VPInstruction::FirstOrderRecurrenceSplice>(
                       m_VPValue(V1), m_VPValue(V2))))
          continue;
        VPValue *Imm = Plan.getOrAddLiveIn(
            ConstantInt::getSigned(Type::getInt32Ty(Plan.getContext()), -1));
        VPWidenIntrinsicRecipe *VPSplice = new VPWidenIntrinsicRecipe(
            Intrinsic::experimental_vp_splice,
            {V1, V2, Imm, AllOneMask, PrevEVL, &EVL},
            TypeInfo.inferScalarType(R.getVPSingleValue()), R.getDebugLoc());
        VPSplice->insertBefore(&R);
        R.getVPSingleValue()->replaceAllUsesWith(VPSplice);
        ToErase.push_back(&R);
      }
    }
  }

  VPValue *HeaderMask = findHeaderMask(Plan);
  if (!HeaderMask)
    return;

  // Replace header masks with a mask equivalent to predicating by EVL:
  //
  // icmp ule widen-canonical-iv backedge-taken-count
  // ->
  // icmp ult step-vector, EVL
  VPRecipeBase *EVLR = EVL.getDefiningRecipe();
  VPBuilder Builder(EVLR->getParent(), std::next(EVLR->getIterator()));
  Type *EVLType = TypeInfo.inferScalarType(&EVL);
  VPValue *EVLMask = Builder.createICmp(
      CmpInst::ICMP_ULT,
      Builder.createNaryOp(VPInstruction::StepVector, {}, EVLType), &EVL);
  HeaderMask->replaceAllUsesWith(EVLMask);
  ToErase.push_back(HeaderMask->getDefiningRecipe());

  // Try to optimize header mask recipes away to their EVL variants.
  // TODO: Split optimizeMaskToEVL out and move into
  // VPlanTransforms::optimize. transformRecipestoEVLRecipes should be run in
  // tryToBuildVPlanWithVPRecipes beforehand.
  for (VPUser *U : collectUsersRecursively(EVLMask)) {
    auto *CurRecipe = cast<VPRecipeBase>(U);
    VPRecipeBase *EVLRecipe =
        optimizeMaskToEVL(EVLMask, *CurRecipe, TypeInfo, *AllOneMask, EVL);
    if (!EVLRecipe)
      continue;

    [[maybe_unused]] unsigned NumDefVal = EVLRecipe->getNumDefinedValues();
    assert(NumDefVal == CurRecipe->getNumDefinedValues() &&
           "New recipe must define the same number of values as the "
           "original.");
    assert(NumDefVal <= 1 &&
           "Only supports recipes with a single definition or without users.");
    EVLRecipe->insertBefore(CurRecipe);
    if (isa<VPSingleDefRecipe, VPWidenLoadEVLRecipe>(EVLRecipe)) {
      VPValue *CurVPV = CurRecipe->getVPSingleValue();
      CurVPV->replaceAllUsesWith(EVLRecipe->getVPSingleValue());
    }
    ToErase.push_back(CurRecipe);
  }
  // Remove dead EVL mask.
  if (EVLMask->getNumUsers() == 0)
    ToErase.push_back(EVLMask->getDefiningRecipe());

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
/// %AVL = phi [ trip-count, %vector.ph ], [ %NextAVL, %vector.body ]
/// %VPEVL = EXPLICIT-VECTOR-LENGTH %AVL
/// ...
/// %OpEVL = cast i32 %VPEVL to IVSize
/// %NextEVLIV = add IVSize %OpEVL, %EVLPhi
/// %NextAVL = sub IVSize nuw %AVL, %OpEVL
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
/// %AVL = phi [ trip-count, %vector.ph ], [ %NextAVL, %vector.body ]
/// %cmp = cmp ult %AVL, MaxSafeElements
/// %SAFE_AVL = select %cmp, %AVL, MaxSafeElements
/// %VPEVL = EXPLICIT-VECTOR-LENGTH %SAFE_AVL
/// ...
/// %OpEVL = cast i32 %VPEVL to IVSize
/// %NextEVLIV = add IVSize %OpEVL, %EVLPhi
/// %NextAVL = sub IVSize nuw %AVL, %OpEVL
/// ...
///
void VPlanTransforms::addExplicitVectorLength(
    VPlan &Plan, const std::optional<unsigned> &MaxSafeElements) {
  VPBasicBlock *Header = Plan.getVectorLoopRegion()->getEntryBasicBlock();

  auto *CanonicalIVPHI = Plan.getCanonicalIV();
  auto *CanIVTy = CanonicalIVPHI->getScalarType();
  VPValue *StartV = CanonicalIVPHI->getStartValue();

  // Create the ExplicitVectorLengthPhi recipe in the main loop.
  auto *EVLPhi = new VPEVLBasedIVPHIRecipe(StartV, DebugLoc());
  EVLPhi->insertAfter(CanonicalIVPHI);
  VPBuilder Builder(Header, Header->getFirstNonPhi());
  // Create the AVL (application vector length), starting from TC -> 0 in steps
  // of EVL.
  VPPhi *AVLPhi = Builder.createScalarPhi(
      {Plan.getTripCount()}, DebugLoc::getCompilerGenerated(), "avl");
  VPValue *AVL = AVLPhi;

  if (MaxSafeElements) {
    // Support for MaxSafeDist for correct loop emission.
    VPValue *AVLSafe =
        Plan.getOrAddLiveIn(ConstantInt::get(CanIVTy, *MaxSafeElements));
    VPValue *Cmp = Builder.createICmp(ICmpInst::ICMP_ULT, AVL, AVLSafe);
    AVL = Builder.createSelect(Cmp, AVL, AVLSafe, DebugLoc(), "safe_avl");
  }
  auto *VPEVL = Builder.createNaryOp(VPInstruction::ExplicitVectorLength, AVL,
                                     DebugLoc());

  auto *CanonicalIVIncrement =
      cast<VPInstruction>(CanonicalIVPHI->getBackedgeValue());
  Builder.setInsertPoint(CanonicalIVIncrement);
  VPValue *OpVPEVL = VPEVL;

  auto *I32Ty = Type::getInt32Ty(Plan.getContext());
  OpVPEVL = Builder.createScalarZExtOrTrunc(
      OpVPEVL, CanIVTy, I32Ty, CanonicalIVIncrement->getDebugLoc());

  auto *NextEVLIV = Builder.createOverflowingOp(
      Instruction::Add, {OpVPEVL, EVLPhi},
      {CanonicalIVIncrement->hasNoUnsignedWrap(),
       CanonicalIVIncrement->hasNoSignedWrap()},
      CanonicalIVIncrement->getDebugLoc(), "index.evl.next");
  EVLPhi->addOperand(NextEVLIV);

  VPValue *NextAVL = Builder.createOverflowingOp(
      Instruction::Sub, {AVLPhi, OpVPEVL}, {/*hasNUW=*/true, /*hasNSW=*/false},
      DebugLoc::getCompilerGenerated(), "avl.next");
  AVLPhi->addOperand(NextAVL);

  transformRecipestoEVLRecipes(Plan, *VPEVL);

  // Replace all uses of VPCanonicalIVPHIRecipe by
  // VPEVLBasedIVPHIRecipe except for the canonical IV increment.
  CanonicalIVPHI->replaceAllUsesWith(EVLPhi);
  CanonicalIVIncrement->setOperand(0, CanonicalIVPHI);
  // TODO: support unroll factor > 1.
  Plan.setUF(1);
}

void VPlanTransforms::canonicalizeEVLLoops(VPlan &Plan) {
  // Find EVL loop entries by locating VPEVLBasedIVPHIRecipe.
  // There should be only one EVL PHI in the entire plan.
  VPEVLBasedIVPHIRecipe *EVLPhi = nullptr;

  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(
           vp_depth_first_shallow(Plan.getEntry())))
    for (VPRecipeBase &R : VPBB->phis())
      if (auto *PhiR = dyn_cast<VPEVLBasedIVPHIRecipe>(&R)) {
        assert(!EVLPhi && "Found multiple EVL PHIs. Only one expected");
        EVLPhi = PhiR;
      }

  // Early return if no EVL PHI is found.
  if (!EVLPhi)
    return;

  VPBasicBlock *HeaderVPBB = EVLPhi->getParent();
  VPValue *EVLIncrement = EVLPhi->getBackedgeValue();
  VPValue *AVL;
  [[maybe_unused]] bool FoundAVL =
      match(EVLIncrement,
            m_c_Add(m_ZExtOrSelf(m_EVL(m_VPValue(AVL))), m_Specific(EVLPhi)));
  assert(FoundAVL && "Didn't find AVL?");

  // The AVL may be capped to a safe distance.
  VPValue *SafeAVL;
  if (match(AVL, m_Select(m_VPValue(), m_VPValue(SafeAVL), m_VPValue())))
    AVL = SafeAVL;

  VPValue *AVLNext;
  [[maybe_unused]] bool FoundAVLNext =
      match(AVL, m_VPInstruction<Instruction::PHI>(
                     m_Specific(Plan.getTripCount()), m_VPValue(AVLNext)));
  assert(FoundAVLNext && "Didn't find AVL backedge?");

  // Convert EVLPhi to concrete recipe.
  auto *ScalarR =
      VPBuilder(EVLPhi).createScalarPhi({EVLPhi->getStartValue(), EVLIncrement},
                                        EVLPhi->getDebugLoc(), "evl.based.iv");
  EVLPhi->replaceAllUsesWith(ScalarR);
  EVLPhi->eraseFromParent();

  // Replace CanonicalIVInc with EVL-PHI increment.
  auto *CanonicalIV = cast<VPPhi>(&*HeaderVPBB->begin());
  VPValue *Backedge = CanonicalIV->getIncomingValue(1);
  assert(match(Backedge, m_c_Add(m_Specific(CanonicalIV),
                                 m_Specific(&Plan.getVFxUF()))) &&
         "Unexpected canonical iv");
  Backedge->replaceAllUsesWith(EVLIncrement);

  // Remove unused phi and increment.
  VPRecipeBase *CanonicalIVIncrement = Backedge->getDefiningRecipe();
  CanonicalIVIncrement->eraseFromParent();
  CanonicalIV->eraseFromParent();

  // Replace the use of VectorTripCount in the latch-exiting block.
  // Before: (branch-on-count EVLIVInc, VectorTripCount)
  // After: (branch-on-cond eq AVLNext, 0)

  VPBasicBlock *LatchExiting =
      HeaderVPBB->getPredecessors()[1]->getEntryBasicBlock();
  auto *LatchExitingBr = cast<VPInstruction>(LatchExiting->getTerminator());
  // Skip single-iteration loop region
  if (match(LatchExitingBr, m_BranchOnCond(m_True())))
    return;
  assert(LatchExitingBr &&
         match(LatchExitingBr,
               m_BranchOnCount(m_VPValue(EVLIncrement),
                               m_Specific(&Plan.getVectorTripCount()))) &&
         "Unexpected terminator in EVL loop");

  Type *AVLTy = VPTypeAnalysis(Plan).inferScalarType(AVLNext);
  VPBuilder Builder(LatchExitingBr);
  VPValue *Cmp =
      Builder.createICmp(CmpInst::ICMP_EQ, AVLNext,
                         Plan.getOrAddLiveIn(ConstantInt::getNullValue(AVLTy)));
  Builder.createNaryOp(VPInstruction::BranchOnCond, Cmp);
  LatchExitingBr->eraseFromParent();
}

void VPlanTransforms::replaceSymbolicStrides(
    VPlan &Plan, PredicatedScalarEvolution &PSE,
    const DenseMap<Value *, const SCEV *> &StridesMap) {
  // Replace VPValues for known constant strides guaranteed by predicate scalar
  // evolution.
  auto CanUseVersionedStride = [&Plan](VPUser &U, unsigned) {
    auto *R = cast<VPRecipeBase>(&U);
    return R->getParent()->getParent() ||
           R->getParent() == Plan.getVectorLoopRegion()->getSinglePredecessor();
  };
  for (const SCEV *Stride : StridesMap.values()) {
    using namespace SCEVPatternMatch;
    auto *StrideV = cast<SCEVUnknown>(Stride)->getValue();
    const APInt *StrideConst;
    if (!match(PSE.getSCEV(StrideV), m_scev_APInt(StrideConst)))
      // Only handle constant strides for now.
      continue;

    auto *CI =
        Plan.getOrAddLiveIn(ConstantInt::get(Stride->getType(), *StrideConst));
    if (VPValue *StrideVPV = Plan.getLiveIn(StrideV))
      StrideVPV->replaceUsesWithIf(CI, CanUseVersionedStride);

    // The versioned value may not be used in the loop directly but through a
    // sext/zext. Add new live-ins in those cases.
    for (Value *U : StrideV->users()) {
      if (!isa<SExtInst, ZExtInst>(U))
        continue;
      VPValue *StrideVPV = Plan.getLiveIn(U);
      if (!StrideVPV)
        continue;
      unsigned BW = U->getType()->getScalarSizeInBits();
      APInt C =
          isa<SExtInst>(U) ? StrideConst->sext(BW) : StrideConst->zext(BW);
      VPValue *CI = Plan.getOrAddLiveIn(ConstantInt::get(U->getType(), C));
      StrideVPV->replaceUsesWithIf(CI, CanUseVersionedStride);
    }
  }
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
    auto *Start =
        cast<VPWidenMemoryRecipe>(RecipeBuilder.getRecipe(IG->getMember(0)));
    VPIRMetadata InterleaveMD(*Start);
    SmallVector<VPValue *, 4> StoredValues;
    if (auto *StoreR = dyn_cast<VPWidenStoreRecipe>(Start))
      StoredValues.push_back(StoreR->getStoredValue());
    for (unsigned I = 1; I < IG->getFactor(); ++I) {
      Instruction *MemberI = IG->getMember(I);
      if (!MemberI)
        continue;
      VPWidenMemoryRecipe *MemoryR =
          cast<VPWidenMemoryRecipe>(RecipeBuilder.getRecipe(MemberI));
      if (auto *StoreR = dyn_cast<VPWidenStoreRecipe>(MemoryR))
        StoredValues.push_back(StoreR->getStoredValue());
      InterleaveMD.intersect(*MemoryR);
    }

    bool NeedsMaskForGaps =
        (IG->requiresScalarEpilogue() && !ScalarEpilogueAllowed) ||
        (!StoredValues.empty() && !IG->isFull());

    Instruction *IRInsertPos = IG->getInsertPos();
    auto *InsertPos =
        cast<VPWidenMemoryRecipe>(RecipeBuilder.getRecipe(IRInsertPos));

    GEPNoWrapFlags NW = GEPNoWrapFlags::none();
    if (auto *Gep = dyn_cast<GetElementPtrInst>(
            getLoadStorePointerOperand(IRInsertPos)->stripPointerCasts()))
      NW = Gep->getNoWrapFlags().withoutNoUnsignedWrap();

    // Get or create the start address for the interleave group.
    VPValue *Addr = Start->getAddr();
    VPRecipeBase *AddrDef = Addr->getDefiningRecipe();
    if (AddrDef && !VPDT.properlyDominates(AddrDef, InsertPos)) {
      // We cannot re-use the address of member zero because it does not
      // dominate the insert position. Instead, use the address of the insert
      // position and create a PtrAdd adjusting it to the address of member
      // zero.
      // TODO: Hoist Addr's defining recipe (and any operands as needed) to
      // InsertPos or sink loads above zero members to join it.
      assert(IG->getIndex(IRInsertPos) != 0 &&
             "index of insert position shouldn't be zero");
      auto &DL = IRInsertPos->getDataLayout();
      APInt Offset(32,
                   DL.getTypeAllocSize(getLoadStoreType(IRInsertPos)) *
                       IG->getIndex(IRInsertPos),
                   /*IsSigned=*/true);
      VPValue *OffsetVPV =
          Plan.getOrAddLiveIn(ConstantInt::get(Plan.getContext(), -Offset));
      VPBuilder B(InsertPos);
      Addr = B.createNoWrapPtrAdd(InsertPos->getAddr(), OffsetVPV, NW);
    }
    // If the group is reverse, adjust the index to refer to the last vector
    // lane instead of the first. We adjust the index from the first vector
    // lane, rather than directly getting the pointer for lane VF - 1, because
    // the pointer operand of the interleaved access is supposed to be uniform.
    if (IG->isReverse()) {
      auto *ReversePtr = new VPVectorEndPointerRecipe(
          Addr, &Plan.getVF(), getLoadStoreType(IRInsertPos),
          -(int64_t)IG->getFactor(), NW, InsertPos->getDebugLoc());
      ReversePtr->insertBefore(InsertPos);
      Addr = ReversePtr;
    }
    auto *VPIG = new VPInterleaveRecipe(IG, Addr, StoredValues,
                                        InsertPos->getMask(), NeedsMaskForGaps,
                                        InterleaveMD, InsertPos->getDebugLoc());
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

/// Expand a VPWidenIntOrFpInduction into executable recipes, for the initial
/// value, phi and backedge value. In the following example:
///
///  vector.ph:
///  Successor(s): vector loop
///
///  <x1> vector loop: {
///    vector.body:
///      WIDEN-INDUCTION %i = phi %start, %step, %vf
///      ...
///      EMIT branch-on-count ...
///    No successors
///  }
///
/// WIDEN-INDUCTION will get expanded to:
///
///  vector.ph:
///    ...
///    vp<%induction.start> = ...
///    vp<%induction.increment> = ...
///
///  Successor(s): vector loop
///
///  <x1> vector loop: {
///    vector.body:
///      ir<%i> = WIDEN-PHI vp<%induction.start>, vp<%vec.ind.next>
///      ...
///      vp<%vec.ind.next> = add ir<%i>, vp<%induction.increment>
///      EMIT branch-on-count ...
///    No successors
///  }
static void
expandVPWidenIntOrFpInduction(VPWidenIntOrFpInductionRecipe *WidenIVR,
                              VPTypeAnalysis &TypeInfo) {
  VPlan *Plan = WidenIVR->getParent()->getPlan();
  VPValue *Start = WidenIVR->getStartValue();
  VPValue *Step = WidenIVR->getStepValue();
  VPValue *VF = WidenIVR->getVFValue();
  DebugLoc DL = WidenIVR->getDebugLoc();

  // The value from the original loop to which we are mapping the new induction
  // variable.
  Type *Ty = TypeInfo.inferScalarType(WidenIVR);

  const InductionDescriptor &ID = WidenIVR->getInductionDescriptor();
  Instruction::BinaryOps AddOp;
  Instruction::BinaryOps MulOp;
  // FIXME: The newly created binary instructions should contain nsw/nuw
  // flags, which can be found from the original scalar operations.
  VPIRFlags Flags;
  if (ID.getKind() == InductionDescriptor::IK_IntInduction) {
    AddOp = Instruction::Add;
    MulOp = Instruction::Mul;
  } else {
    AddOp = ID.getInductionOpcode();
    MulOp = Instruction::FMul;
    Flags = ID.getInductionBinOp()->getFastMathFlags();
  }

  // If the phi is truncated, truncate the start and step values.
  VPBuilder Builder(Plan->getVectorPreheader());
  Type *StepTy = TypeInfo.inferScalarType(Step);
  if (Ty->getScalarSizeInBits() < StepTy->getScalarSizeInBits()) {
    assert(StepTy->isIntegerTy() && "Truncation requires an integer type");
    Step = Builder.createScalarCast(Instruction::Trunc, Step, Ty, DL);
    Start = Builder.createScalarCast(Instruction::Trunc, Start, Ty, DL);
    StepTy = Ty;
  }

  // Construct the initial value of the vector IV in the vector loop preheader.
  Type *IVIntTy =
      IntegerType::get(Plan->getContext(), StepTy->getScalarSizeInBits());
  VPValue *Init = Builder.createNaryOp(VPInstruction::StepVector, {}, IVIntTy);
  if (StepTy->isFloatingPointTy())
    Init = Builder.createWidenCast(Instruction::UIToFP, Init, StepTy);

  VPValue *SplatStart = Builder.createNaryOp(VPInstruction::Broadcast, Start);
  VPValue *SplatStep = Builder.createNaryOp(VPInstruction::Broadcast, Step);

  Init = Builder.createNaryOp(MulOp, {Init, SplatStep}, Flags);
  Init =
      Builder.createNaryOp(AddOp, {SplatStart, Init}, Flags, {}, "induction");

  // Create the widened phi of the vector IV.
  auto *WidePHI = new VPWidenPHIRecipe(WidenIVR->getPHINode(), nullptr,
                                       WidenIVR->getDebugLoc(), "vec.ind");
  WidePHI->addOperand(Init);
  WidePHI->insertBefore(WidenIVR);

  // Create the backedge value for the vector IV.
  VPValue *Inc;
  VPValue *Prev;
  // If unrolled, use the increment and prev value from the operands.
  if (auto *SplatVF = WidenIVR->getSplatVFValue()) {
    Inc = SplatVF;
    Prev = WidenIVR->getLastUnrolledPartOperand();
  } else {
    if (VPRecipeBase *R = VF->getDefiningRecipe())
      Builder.setInsertPoint(R->getParent(), std::next(R->getIterator()));
    // Multiply the vectorization factor by the step using integer or
    // floating-point arithmetic as appropriate.
    if (StepTy->isFloatingPointTy())
      VF = Builder.createScalarCast(Instruction::CastOps::UIToFP, VF, StepTy,
                                    DL);
    else
      VF = Builder.createScalarZExtOrTrunc(VF, StepTy,
                                           TypeInfo.inferScalarType(VF), DL);

    Inc = Builder.createNaryOp(MulOp, {Step, VF}, Flags);
    Inc = Builder.createNaryOp(VPInstruction::Broadcast, Inc);
    Prev = WidePHI;
  }

  VPBasicBlock *ExitingBB = Plan->getVectorLoopRegion()->getExitingBasicBlock();
  Builder.setInsertPoint(ExitingBB, ExitingBB->getTerminator()->getIterator());
  auto *Next = Builder.createNaryOp(AddOp, {Prev, Inc}, Flags,
                                    WidenIVR->getDebugLoc(), "vec.ind.next");

  WidePHI->addOperand(Next);

  WidenIVR->replaceAllUsesWith(WidePHI);
}

/// Expand a VPWidenPointerInductionRecipe into executable recipes, for the
/// initial value, phi and backedge value. In the following example:
///
///  <x1> vector loop: {
///    vector.body:
///      EMIT ir<%ptr.iv> = WIDEN-POINTER-INDUCTION %start, %step, %vf
///      ...
///      EMIT branch-on-count ...
///  }
///
/// WIDEN-POINTER-INDUCTION will get expanded to:
///
///  <x1> vector loop: {
///    vector.body:
///      EMIT-SCALAR %pointer.phi = phi %start, %ptr.ind
///      EMIT %mul = mul %stepvector, %step
///      EMIT %vector.gep = wide-ptradd %pointer.phi, %mul
///      ...
///      EMIT %ptr.ind = ptradd %pointer.phi, %vf
///      EMIT branch-on-count ...
///  }
static void expandVPWidenPointerInduction(VPWidenPointerInductionRecipe *R,
                                          VPTypeAnalysis &TypeInfo) {
  VPlan *Plan = R->getParent()->getPlan();
  VPValue *Start = R->getStartValue();
  VPValue *Step = R->getStepValue();
  VPValue *VF = R->getVFValue();

  assert(R->getInductionDescriptor().getKind() ==
             InductionDescriptor::IK_PtrInduction &&
         "Not a pointer induction according to InductionDescriptor!");
  assert(TypeInfo.inferScalarType(R)->isPointerTy() && "Unexpected type.");
  assert(!R->onlyScalarsGenerated(Plan->hasScalableVF()) &&
         "Recipe should have been replaced");

  VPBuilder Builder(R);
  DebugLoc DL = R->getDebugLoc();

  // Build a scalar pointer phi.
  VPPhi *ScalarPtrPhi = Builder.createScalarPhi(Start, DL, "pointer.phi");

  // Create actual address geps that use the pointer phi as base and a
  // vectorized version of the step value (<step*0, ..., step*N>) as offset.
  Builder.setInsertPoint(R->getParent(), R->getParent()->getFirstNonPhi());
  Type *StepTy = TypeInfo.inferScalarType(Step);
  VPValue *Offset = Builder.createNaryOp(VPInstruction::StepVector, {}, StepTy);
  Offset = Builder.createNaryOp(Instruction::Mul, {Offset, Step});
  VPValue *PtrAdd = Builder.createNaryOp(
      VPInstruction::WidePtrAdd, {ScalarPtrPhi, Offset}, DL, "vector.gep");
  R->replaceAllUsesWith(PtrAdd);

  // Create the backedge value for the scalar pointer phi.
  VPBasicBlock *ExitingBB = Plan->getVectorLoopRegion()->getExitingBasicBlock();
  Builder.setInsertPoint(ExitingBB, ExitingBB->getTerminator()->getIterator());
  VF = Builder.createScalarZExtOrTrunc(VF, StepTy, TypeInfo.inferScalarType(VF),
                                       DL);
  VPValue *Inc = Builder.createNaryOp(Instruction::Mul, {Step, VF});

  VPValue *InductionGEP =
      Builder.createPtrAdd(ScalarPtrPhi, Inc, DL, "ptr.ind");
  ScalarPtrPhi->addOperand(InductionGEP);
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

void VPlanTransforms::convertToConcreteRecipes(VPlan &Plan) {
  VPTypeAnalysis TypeInfo(Plan);
  SmallVector<VPRecipeBase *> ToRemove;
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(
           vp_depth_first_deep(Plan.getEntry()))) {
    for (VPRecipeBase &R : make_early_inc_range(*VPBB)) {
      if (auto *WidenIVR = dyn_cast<VPWidenIntOrFpInductionRecipe>(&R)) {
        expandVPWidenIntOrFpInduction(WidenIVR, TypeInfo);
        ToRemove.push_back(WidenIVR);
        continue;
      }

      if (auto *WidenIVR = dyn_cast<VPWidenPointerInductionRecipe>(&R)) {
        expandVPWidenPointerInduction(WidenIVR, TypeInfo);
        ToRemove.push_back(WidenIVR);
        continue;
      }

      // Expand VPBlendRecipe into VPInstruction::Select.
      VPBuilder Builder(&R);
      if (auto *Blend = dyn_cast<VPBlendRecipe>(&R)) {
        VPValue *Select = Blend->getIncomingValue(0);
        for (unsigned I = 1; I != Blend->getNumIncomingValues(); ++I)
          Select = Builder.createSelect(Blend->getMask(I),
                                        Blend->getIncomingValue(I), Select,
                                        R.getDebugLoc(), "predphi");
        Blend->replaceAllUsesWith(Select);
        ToRemove.push_back(Blend);
      }

      if (auto *Expr = dyn_cast<VPExpressionRecipe>(&R)) {
        Expr->decompose();
        ToRemove.push_back(Expr);
      }

      VPValue *VectorStep;
      VPValue *ScalarStep;
      if (!match(&R, m_VPInstruction<VPInstruction::WideIVStep>(
                         m_VPValue(VectorStep), m_VPValue(ScalarStep))))
        continue;

      // Expand WideIVStep.
      auto *VPI = cast<VPInstruction>(&R);
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

      VPIRFlags Flags;
      if (IVTy->isFloatingPointTy())
        Flags = {VPI->getFastMathFlags()};

      unsigned MulOpc =
          IVTy->isFloatingPointTy() ? Instruction::FMul : Instruction::Mul;
      VPInstruction *Mul = Builder.createNaryOp(
          MulOpc, {VectorStep, ScalarStep}, Flags, R.getDebugLoc());
      VectorStep = Mul;
      VPI->replaceAllUsesWith(VectorStep);
      ToRemove.push_back(VPI);
    }
  }

  for (VPRecipeBase *R : ToRemove)
    R->eraseFromParent();
}

void VPlanTransforms::handleUncountableEarlyExit(VPBasicBlock *EarlyExitingVPBB,
                                                 VPBasicBlock *EarlyExitVPBB,
                                                 VPlan &Plan,
                                                 VPBasicBlock *HeaderVPBB,
                                                 VPBasicBlock *LatchVPBB) {
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
    if (!IncomingFromEarlyExit->isLiveIn()) {
      // Update the incoming value from the early exit.
      VPValue *FirstActiveLane = EarlyExitB.createNaryOp(
          VPInstruction::FirstActiveLane, {CondToEarlyExit}, nullptr,
          "first.active.lane");
      IncomingFromEarlyExit = EarlyExitB.createNaryOp(
          VPInstruction::ExtractLane, {FirstActiveLane, IncomingFromEarlyExit},
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
/// VPExpressionRecipe and clamp the \p Range if it is beneficial and
/// valid. The created recipe must be decomposed to its constituent
/// recipes before execution.
static VPExpressionRecipe *
tryToMatchAndCreateExtendedReduction(VPReductionRecipe *Red, VPCostContext &Ctx,
                                     VFRange &Range) {
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
    return new VPExpressionRecipe(cast<VPWidenCastRecipe>(VecOp), Red);

  return nullptr;
}

/// This function tries convert extended in-loop reductions to
/// VPExpressionRecipe and clamp the \p Range if it is beneficial
/// and valid. The created VPExpressionRecipe must be decomposed to its
/// constituent recipes before execution. Patterns of the
/// VPExpressionRecipe:
///   reduce.add(mul(...)),
///   reduce.add(mul(ext(A), ext(B))),
///   reduce.add(ext(mul(ext(A), ext(B)))).
static VPExpressionRecipe *
tryToMatchAndCreateMulAccumulateReduction(VPReductionRecipe *Red,
                                          VPCostContext &Ctx, VFRange &Range) {
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
                                   Mul, RecipeA, RecipeB, nullptr)) {
      return new VPExpressionRecipe(RecipeA, RecipeB, Mul, Red);
    }
    // Match reduce.add(mul).
    if (IsMulAccValidAndClampRange(true, Mul, nullptr, nullptr, nullptr))
      return new VPExpressionRecipe(Mul, Red);
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
                                   Mul, Ext0, Ext1, Ext)) {
      auto *NewExt0 = new VPWidenCastRecipe(
          Ext0->getOpcode(), Ext0->getOperand(0), Ext->getResultType(), *Ext0,
          Ext0->getDebugLoc());
      NewExt0->insertBefore(Ext0);

      VPWidenCastRecipe *NewExt1 = NewExt0;
      if (Ext0 != Ext1) {
        NewExt1 = new VPWidenCastRecipe(Ext1->getOpcode(), Ext1->getOperand(0),
                                        Ext->getResultType(), *Ext1,
                                        Ext1->getDebugLoc());
        NewExt1->insertBefore(Ext1);
      }
      Mul->setOperand(0, NewExt0);
      Mul->setOperand(1, NewExt1);
      Red->setOperand(1, Mul);
      return new VPExpressionRecipe(NewExt0, NewExt1, Mul, Red);
    }
  }
  return nullptr;
}

/// This function tries to create abstract recipes from the reduction recipe for
/// following optimizations and cost estimation.
static void tryToCreateAbstractReductionRecipe(VPReductionRecipe *Red,
                                               VPCostContext &Ctx,
                                               VFRange &Range) {
  VPExpressionRecipe *AbstractR = nullptr;
  auto IP = std::next(Red->getIterator());
  auto *VPBB = Red->getParent();
  if (auto *MulAcc = tryToMatchAndCreateMulAccumulateReduction(Red, Ctx, Range))
    AbstractR = MulAcc;
  else if (auto *ExtRed = tryToMatchAndCreateExtendedReduction(Red, Ctx, Range))
    AbstractR = ExtRed;
  // Cannot create abstract inloop reduction recipes.
  if (!AbstractR)
    return;

  AbstractR->insertBefore(*VPBB, IP);
  Red->replaceAllUsesWith(AbstractR);
}

void VPlanTransforms::convertToAbstractRecipes(VPlan &Plan, VPCostContext &Ctx,
                                               VFRange &Range) {
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(
           vp_depth_first_deep(Plan.getVectorLoopRegion()))) {
    for (VPRecipeBase &R : make_early_inc_range(*VPBB)) {
      if (auto *Red = dyn_cast<VPReductionRecipe>(&R))
        tryToCreateAbstractReductionRecipe(Red, Ctx, Range);
    }
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
    if (vputils::onlyScalarValuesUsed(VPV) ||
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

void VPlanTransforms::materializeConstantVectorTripCount(
    VPlan &Plan, ElementCount BestVF, unsigned BestUF,
    PredicatedScalarEvolution &PSE) {
  assert(Plan.hasVF(BestVF) && "BestVF is not available in Plan");
  assert(Plan.hasUF(BestUF) && "BestUF is not available in Plan");

  VPValue *TC = Plan.getTripCount();
  // Skip cases for which the trip count may be non-trivial to materialize.
  // I.e., when a scalar tail is absent - due to tail folding, or when a scalar
  // tail is required.
  if (!Plan.hasScalarTail() ||
      Plan.getMiddleBlock()->getSingleSuccessor() ==
          Plan.getScalarPreheader() ||
      !TC->isLiveIn())
    return;

  // Materialize vector trip counts for constants early if it can simply
  // be computed as (Original TC / VF * UF) * VF * UF.
  // TODO: Compute vector trip counts for loops requiring a scalar epilogue and
  // tail-folded loops.
  ScalarEvolution &SE = *PSE.getSE();
  auto *TCScev = SE.getSCEV(TC->getLiveInIRValue());
  if (!isa<SCEVConstant>(TCScev))
    return;
  const SCEV *VFxUF = SE.getElementCount(TCScev->getType(), BestVF * BestUF);
  auto VecTCScev = SE.getMulExpr(SE.getUDivExpr(TCScev, VFxUF), VFxUF);
  if (auto *ConstVecTC = dyn_cast<SCEVConstant>(VecTCScev))
    Plan.getVectorTripCount().setUnderlyingValue(ConstVecTC->getValue());
}

void VPlanTransforms::materializeBackedgeTakenCount(VPlan &Plan,
                                                    VPBasicBlock *VectorPH) {
  VPValue *BTC = Plan.getOrCreateBackedgeTakenCount();
  if (BTC->getNumUsers() == 0)
    return;

  VPBuilder Builder(VectorPH, VectorPH->begin());
  auto *TCTy = VPTypeAnalysis(Plan).inferScalarType(Plan.getTripCount());
  auto *TCMO = Builder.createNaryOp(
      Instruction::Sub,
      {Plan.getTripCount(), Plan.getOrAddLiveIn(ConstantInt::get(TCTy, 1))},
      DebugLoc::getCompilerGenerated(), "trip.count.minus.1");
  BTC->replaceAllUsesWith(TCMO);
}

void VPlanTransforms::materializeBuildVectors(VPlan &Plan) {
  if (Plan.hasScalarVFOnly())
    return;

  VPTypeAnalysis TypeInfo(Plan);
  VPRegionBlock *LoopRegion = Plan.getVectorLoopRegion();
  auto VPBBsOutsideLoopRegion = VPBlockUtils::blocksOnly<VPBasicBlock>(
      vp_depth_first_shallow(Plan.getEntry()));
  auto VPBBsInsideLoopRegion = VPBlockUtils::blocksOnly<VPBasicBlock>(
      vp_depth_first_shallow(LoopRegion->getEntry()));
  // Materialize Build(Struct)Vector for all replicating VPReplicateRecipes,
  // excluding ones in replicate regions. Those are not materialized explicitly
  // yet. Those vector users are still handled in VPReplicateRegion::execute(),
  // via shouldPack().
  // TODO: materialize build vectors for replicating recipes in replicating
  // regions.
  // TODO: materialize build vectors for VPInstructions.
  for (VPBasicBlock *VPBB :
       concat<VPBasicBlock *>(VPBBsOutsideLoopRegion, VPBBsInsideLoopRegion)) {
    for (VPRecipeBase &R : make_early_inc_range(*VPBB)) {
      auto *RepR = dyn_cast<VPReplicateRecipe>(&R);
      auto UsesVectorOrInsideReplicateRegion = [RepR, LoopRegion](VPUser *U) {
        VPRegionBlock *ParentRegion =
            cast<VPRecipeBase>(U)->getParent()->getParent();
        return !U->usesScalars(RepR) || ParentRegion != LoopRegion;
      };
      if (!RepR || RepR->isSingleScalar() ||
          none_of(RepR->users(), UsesVectorOrInsideReplicateRegion))
        continue;

      Type *ScalarTy = TypeInfo.inferScalarType(RepR);
      unsigned Opcode = ScalarTy->isStructTy()
                            ? VPInstruction::BuildStructVector
                            : VPInstruction::BuildVector;
      auto *BuildVector = new VPInstruction(Opcode, {RepR});
      BuildVector->insertAfter(RepR);

      RepR->replaceUsesWithIf(
          BuildVector, [BuildVector, &UsesVectorOrInsideReplicateRegion](
                           VPUser &U, unsigned) {
            return &U != BuildVector && UsesVectorOrInsideReplicateRegion(&U);
          });
    }
  }
}

void VPlanTransforms::materializeVectorTripCount(VPlan &Plan,
                                                 VPBasicBlock *VectorPHVPBB,
                                                 bool TailByMasking,
                                                 bool RequiresScalarEpilogue) {
  VPValue &VectorTC = Plan.getVectorTripCount();
  assert(VectorTC.isLiveIn() && "vector-trip-count must be a live-in");
  // There's nothing to do if there are no users of the vector trip count or its
  // IR value has already been set.
  if (VectorTC.getNumUsers() == 0 || VectorTC.getLiveInIRValue())
    return;

  VPValue *TC = Plan.getTripCount();
  Type *TCTy = VPTypeAnalysis(Plan).inferScalarType(TC);
  VPBuilder Builder(VectorPHVPBB, VectorPHVPBB->begin());
  VPValue *Step = &Plan.getVFxUF();

  // If the tail is to be folded by masking, round the number of iterations N
  // up to a multiple of Step instead of rounding down. This is done by first
  // adding Step-1 and then rounding down. Note that it's ok if this addition
  // overflows: the vector induction variable will eventually wrap to zero given
  // that it starts at zero and its Step is a power of two; the loop will then
  // exit, with the last early-exit vector comparison also producing all-true.
  // For scalable vectors the VF is not guaranteed to be a power of 2, but this
  // is accounted for in emitIterationCountCheck that adds an overflow check.
  if (TailByMasking) {
    TC = Builder.createNaryOp(
        Instruction::Add,
        {TC, Builder.createNaryOp(
                 Instruction::Sub,
                 {Step, Plan.getOrAddLiveIn(ConstantInt::get(TCTy, 1))})},
        DebugLoc::getCompilerGenerated(), "n.rnd.up");
  }

  // Now we need to generate the expression for the part of the loop that the
  // vectorized body will execute. This is equal to N - (N % Step) if scalar
  // iterations are not required for correctness, or N - Step, otherwise. Step
  // is equal to the vectorization factor (number of SIMD elements) times the
  // unroll factor (number of SIMD instructions).
  VPValue *R =
      Builder.createNaryOp(Instruction::URem, {TC, Step},
                           DebugLoc::getCompilerGenerated(), "n.mod.vf");

  // There are cases where we *must* run at least one iteration in the remainder
  // loop.  See the cost model for when this can happen.  If the step evenly
  // divides the trip count, we set the remainder to be equal to the step. If
  // the step does not evenly divide the trip count, no adjustment is necessary
  // since there will already be scalar iterations. Note that the minimum
  // iterations check ensures that N >= Step.
  if (RequiresScalarEpilogue) {
    assert(!TailByMasking &&
           "requiring scalar epilogue is not supported with fail folding");
    VPValue *IsZero = Builder.createICmp(
        CmpInst::ICMP_EQ, R, Plan.getOrAddLiveIn(ConstantInt::get(TCTy, 0)));
    R = Builder.createSelect(IsZero, Step, R);
  }

  VPValue *Res = Builder.createNaryOp(
      Instruction::Sub, {TC, R}, DebugLoc::getCompilerGenerated(), "n.vec");
  VectorTC.replaceAllUsesWith(Res);
}

void VPlanTransforms::materializeVFAndVFxUF(VPlan &Plan, VPBasicBlock *VectorPH,
                                            ElementCount VFEC) {
  VPBuilder Builder(VectorPH, VectorPH->begin());
  Type *TCTy = VPTypeAnalysis(Plan).inferScalarType(Plan.getTripCount());
  VPValue &VF = Plan.getVF();
  VPValue &VFxUF = Plan.getVFxUF();
  // Note that after the transform, Plan.getVF and Plan.getVFxUF should not be
  // used.
  // TODO: Assert that they aren't used.

  // If there are no users of the runtime VF, compute VFxUF by constant folding
  // the multiplication of VF and UF.
  if (VF.getNumUsers() == 0) {
    VPValue *RuntimeVFxUF =
        Builder.createElementCount(TCTy, VFEC * Plan.getUF());
    VFxUF.replaceAllUsesWith(RuntimeVFxUF);
    return;
  }

  // For users of the runtime VF, compute it as VF * vscale, and VFxUF as (VF *
  // vscale) * UF.
  VPValue *RuntimeVF = Builder.createElementCount(TCTy, VFEC);
  if (!vputils::onlyScalarValuesUsed(&VF)) {
    VPValue *BC = Builder.createNaryOp(VPInstruction::Broadcast, RuntimeVF);
    VF.replaceUsesWithIf(
        BC, [&VF](VPUser &U, unsigned) { return !U.usesScalars(&VF); });
  }
  VF.replaceAllUsesWith(RuntimeVF);

  VPValue *UF = Plan.getOrAddLiveIn(ConstantInt::get(TCTy, Plan.getUF()));
  VPValue *MulByUF = Builder.createNaryOp(Instruction::Mul, {RuntimeVF, UF});
  VFxUF.replaceAllUsesWith(MulByUF);
}

DenseMap<const SCEV *, Value *>
VPlanTransforms::expandSCEVs(VPlan &Plan, ScalarEvolution &SE) {
  const DataLayout &DL = SE.getDataLayout();
  SCEVExpander Expander(SE, DL, "induction", /*PreserveLCSSA=*/true);

  auto *Entry = cast<VPIRBasicBlock>(Plan.getEntry());
  BasicBlock *EntryBB = Entry->getIRBasicBlock();
  DenseMap<const SCEV *, Value *> ExpandedSCEVs;
  for (VPRecipeBase &R : make_early_inc_range(*Entry)) {
    if (isa<VPIRInstruction, VPIRPhi>(&R))
      continue;
    auto *ExpSCEV = dyn_cast<VPExpandSCEVRecipe>(&R);
    if (!ExpSCEV)
      break;
    const SCEV *Expr = ExpSCEV->getSCEV();
    Value *Res =
        Expander.expandCodeFor(Expr, Expr->getType(), EntryBB->getTerminator());
    ExpandedSCEVs[ExpSCEV->getSCEV()] = Res;
    VPValue *Exp = Plan.getOrAddLiveIn(Res);
    ExpSCEV->replaceAllUsesWith(Exp);
    if (Plan.getTripCount() == ExpSCEV)
      Plan.resetTripCount(Exp);
    ExpSCEV->eraseFromParent();
  }
  assert(none_of(*Entry, IsaPred<VPExpandSCEVRecipe>) &&
         "VPExpandSCEVRecipes must be at the beginning of the entry block, "
         "after any VPIRInstructions");
  // Add IR instructions in the entry basic block but not in the VPIRBasicBlock
  // to the VPIRBasicBlock.
  auto EI = Entry->begin();
  for (Instruction &I : drop_end(*EntryBB)) {
    if (EI != Entry->end() && isa<VPIRInstruction>(*EI) &&
        &cast<VPIRInstruction>(&*EI)->getInstruction() == &I) {
      EI++;
      continue;
    }
    VPIRInstruction::create(I)->insertBefore(*Entry, EI);
  }

  return ExpandedSCEVs;
}

/// Returns true if \p V is VPWidenLoadRecipe or VPInterleaveRecipe that can be
/// converted to a narrower recipe. \p V is used by a wide recipe that feeds a
/// store interleave group at index \p Idx, \p WideMember0 is the recipe feeding
/// the same interleave group at index 0. A VPWidenLoadRecipe can be narrowed to
/// an index-independent load if it feeds all wide ops at all indices (\p OpV
/// must be the operand at index \p OpIdx for both the recipe at lane 0, \p
/// WideMember0). A VPInterleaveRecipe can be narrowed to a wide load, if \p V
/// is defined at \p Idx of a load interleave group.
static bool canNarrowLoad(VPWidenRecipe *WideMember0, unsigned OpIdx,
                          VPValue *OpV, unsigned Idx) {
  auto *DefR = OpV->getDefiningRecipe();
  if (!DefR)
    return WideMember0->getOperand(OpIdx) == OpV;
  if (auto *W = dyn_cast<VPWidenLoadRecipe>(DefR))
    return !W->getMask() && WideMember0->getOperand(OpIdx) == OpV;

  if (auto *IR = dyn_cast<VPInterleaveRecipe>(DefR))
    return IR->getInterleaveGroup()->isFull() && IR->getVPValue(Idx) == OpV;
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

/// Returns true if \p VPValue is a narrow VPValue.
static bool isAlreadyNarrow(VPValue *VPV) {
  if (VPV->isLiveIn())
    return true;
  auto *RepR = dyn_cast<VPReplicateRecipe>(VPV);
  return RepR && RepR->isSingleScalar();
}

void VPlanTransforms::narrowInterleaveGroups(VPlan &Plan, ElementCount VF,
                                             unsigned VectorRegWidth) {
  VPRegionBlock *VectorLoop = Plan.getVectorLoopRegion();
  if (!VectorLoop)
    return;

  VPTypeAnalysis TypeInfo(Plan);

  unsigned VFMinVal = VF.getKnownMinValue();
  SmallVector<VPInterleaveRecipe *> StoreGroups;
  for (auto &R : *VectorLoop->getEntryBasicBlock()) {
    if (isa<VPCanonicalIVPHIRecipe>(&R) ||
        match(&R, m_BranchOnCount(m_VPValue(), m_VPValue())))
      continue;

    if (isa<VPDerivedIVRecipe, VPScalarIVStepsRecipe>(&R) &&
        vputils::onlyFirstLaneUsed(cast<VPSingleDefRecipe>(&R)))
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
    if (!isConsecutiveInterleaveGroup(InterleaveR, VFMinVal, TypeInfo,
                                      VectorRegWidth))
      return;

    // Skip read interleave groups.
    if (InterleaveR->getStoredValues().empty())
      continue;

    // Narrow interleave groups, if all operands are already matching narrow
    // ops.
    auto *Member0 = InterleaveR->getStoredValues()[0];
    if (isAlreadyNarrow(Member0) &&
        all_of(InterleaveR->getStoredValues(),
               [Member0](VPValue *VPV) { return Member0 == VPV; })) {
      StoreGroups.push_back(InterleaveR);
      continue;
    }

    // For now, we only support full interleave groups storing load interleave
    // groups.
    if (all_of(enumerate(InterleaveR->getStoredValues()), [](auto Op) {
          VPRecipeBase *DefR = Op.value()->getDefiningRecipe();
          if (!DefR)
            return false;
          auto *IR = dyn_cast<VPInterleaveRecipe>(DefR);
          return IR && IR->getInterleaveGroup()->isFull() &&
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
      auto *R = dyn_cast_or_null<VPWidenRecipe>(V->getDefiningRecipe());
      if (!R || R->getOpcode() != WideMember0->getOpcode() ||
          R->getNumOperands() > 2)
        return;
      if (any_of(enumerate(R->operands()),
                 [WideMember0, Idx = I](const auto &P) {
                   const auto &[OpIdx, OpV] = P;
                   return !canNarrowLoad(WideMember0, OpIdx, OpV, Idx);
                 }))
        return;
    }
    StoreGroups.push_back(InterleaveR);
  }

  if (StoreGroups.empty())
    return;

  // Convert InterleaveGroup \p R to a single VPWidenLoadRecipe.
  auto NarrowOp = [](VPValue *V) -> VPValue * {
    auto *R = V->getDefiningRecipe();
    if (!R)
      return V;
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

    if (auto *RepR = dyn_cast<VPReplicateRecipe>(R)) {
      assert(RepR->isSingleScalar() &&
             isa<LoadInst>(RepR->getUnderlyingInstr()) &&
             "must be a single scalar load");
      return RepR;
    }
    auto *WideLoad = cast<VPWidenLoadRecipe>(R);

    VPValue *PtrOp = WideLoad->getAddr();
    if (auto *VecPtr = dyn_cast<VPVectorPointerRecipe>(PtrOp))
      PtrOp = VecPtr->getOperand(0);
    // Narrow wide load to uniform scalar load, as transformed VPlan will only
    // process one original iteration.
    auto *N = new VPReplicateRecipe(&WideLoad->getIngredient(), {PtrOp},
                                    /*IsUniform*/ true,
                                    /*Mask*/ nullptr, *WideLoad);
    N->insertBefore(WideLoad);
    return N;
  };

  // Narrow operation tree rooted at store groups.
  for (auto *StoreGroup : StoreGroups) {
    VPValue *Res = nullptr;
    VPValue *Member0 = StoreGroup->getStoredValues()[0];
    if (isAlreadyNarrow(Member0)) {
      Res = Member0;
    } else if (auto *WideMember0 =
                   dyn_cast<VPWidenRecipe>(Member0->getDefiningRecipe())) {
      for (unsigned Idx = 0, E = WideMember0->getNumOperands(); Idx != E; ++Idx)
        WideMember0->setOperand(Idx, NarrowOp(WideMember0->getOperand(Idx)));
      Res = WideMember0;
    } else {
      Res = NarrowOp(Member0);
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
  VPBuilder PHBuilder(Plan.getVectorPreheader());

  VPValue *UF = Plan.getOrAddLiveIn(
      ConstantInt::get(CanIV->getScalarType(), 1 * Plan.getUF()));
  if (VF.isScalable()) {
    VPValue *VScale = PHBuilder.createElementCount(
        CanIV->getScalarType(), ElementCount::getScalable(1));
    VPValue *VScaleUF = PHBuilder.createNaryOp(Instruction::Mul, {VScale, UF});
    Inc->setOperand(1, VScaleUF);
    Plan.getVF().replaceAllUsesWith(VScale);
  } else {
    Inc->setOperand(1, UF);
    Plan.getVF().replaceAllUsesWith(
        Plan.getOrAddLiveIn(ConstantInt::get(CanIV->getScalarType(), 1)));
  }
  removeDeadRecipes(Plan);
}

/// Add branch weight metadata, if the \p Plan's middle block is terminated by a
/// BranchOnCond recipe.
void VPlanTransforms::addBranchWeightToMiddleTerminator(
    VPlan &Plan, ElementCount VF, std::optional<unsigned> VScaleForTuning) {
  VPBasicBlock *MiddleVPBB = Plan.getMiddleBlock();
  auto *MiddleTerm =
      dyn_cast_or_null<VPInstruction>(MiddleVPBB->getTerminator());
  // Only add branch metadata if there is a (conditional) terminator.
  if (!MiddleTerm)
    return;

  assert(MiddleTerm->getOpcode() == VPInstruction::BranchOnCond &&
         "must have a BranchOnCond");
  // Assume that `TripCount % VectorStep ` is equally distributed.
  unsigned VectorStep = Plan.getUF() * VF.getKnownMinValue();
  if (VF.isScalable() && VScaleForTuning.has_value())
    VectorStep *= *VScaleForTuning;
  assert(VectorStep > 0 && "trip count should not be zero");
  MDBuilder MDB(Plan.getContext());
  MDNode *BranchWeights =
      MDB.createBranchWeights({1, VectorStep - 1}, /*IsExpected=*/false);
  MiddleTerm->addMetadata(LLVMContext::MD_prof, BranchWeights);
}
