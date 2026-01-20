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
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/Analysis/InstSimplifyFolder.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/ScalarEvolutionPatternMatch.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

using namespace llvm;
using namespace VPlanPatternMatch;
using namespace SCEVPatternMatch;

bool VPlanTransforms::tryToConvertVPInstructionsToVPRecipes(
    VPlan &Plan, const TargetLibraryInfo &TLI) {

  ReversePostOrderTraversal<VPBlockDeepTraversalWrapper<VPBlockBase *>> RPOT(
      Plan.getVectorLoopRegion());
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
        NewRecipe = new VPWidenPHIRecipe(Phi, nullptr, PhiR->getDebugLoc());
        for (VPValue *Op : PhiR->operands())
          NewRecipe->addOperand(Op);
      } else if (auto *VPI = dyn_cast<VPInstruction>(&Ingredient)) {
        assert(!isa<PHINode>(Inst) && "phis should be handled above");
        // Create VPWidenMemoryRecipe for loads and stores.
        if (LoadInst *Load = dyn_cast<LoadInst>(Inst)) {
          NewRecipe = new VPWidenLoadRecipe(
              *Load, Ingredient.getOperand(0), nullptr /*Mask*/,
              false /*Consecutive*/, false /*Reverse*/, *VPI,
              Ingredient.getDebugLoc());
        } else if (StoreInst *Store = dyn_cast<StoreInst>(Inst)) {
          NewRecipe = new VPWidenStoreRecipe(
              *Store, Ingredient.getOperand(1), Ingredient.getOperand(0),
              nullptr /*Mask*/, false /*Consecutive*/, false /*Reverse*/, *VPI,
              Ingredient.getDebugLoc());
        } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Inst)) {
          NewRecipe = new VPWidenGEPRecipe(GEP, Ingredient.operands(), *VPI,
                                           Ingredient.getDebugLoc());
        } else if (CallInst *CI = dyn_cast<CallInst>(Inst)) {
          Intrinsic::ID VectorID = getVectorIntrinsicIDForCall(CI, &TLI);
          if (VectorID == Intrinsic::not_intrinsic)
            return false;
          NewRecipe = new VPWidenIntrinsicRecipe(
              *CI, getVectorIntrinsicIDForCall(CI, &TLI),
              drop_end(Ingredient.operands()), CI->getType(), VPIRFlags(*CI),
              *VPI, CI->getDebugLoc());
        } else if (auto *CI = dyn_cast<CastInst>(Inst)) {
          NewRecipe = new VPWidenCastRecipe(
              CI->getOpcode(), Ingredient.getOperand(0), CI->getType(), CI,
              VPIRFlags(*CI), VPIRMetadata(*CI));
        } else {
          NewRecipe = new VPWidenRecipe(*Inst, Ingredient.operands(), *VPI,
                                        *VPI, Ingredient.getDebugLoc());
        }
      } else {
        assert(isa<VPWidenIntOrFpInductionRecipe>(&Ingredient) &&
               "inductions must be created earlier");
        continue;
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

/// Helper for extra no-alias checks via known-safe recipe and SCEV.
class SinkStoreInfo {
  const SmallPtrSetImpl<VPRecipeBase *> &ExcludeRecipes;
  VPReplicateRecipe &GroupLeader;
  PredicatedScalarEvolution &PSE;
  const Loop &L;
  VPTypeAnalysis &TypeInfo;

  // Return true if \p A and \p B are known to not alias for all VFs in the
  // plan, checked via the distance between the accesses
  bool isNoAliasViaDistance(VPReplicateRecipe *A, VPReplicateRecipe *B) const {
    if (A->getOpcode() != Instruction::Store ||
        B->getOpcode() != Instruction::Store)
      return false;

    VPValue *AddrA = A->getOperand(1);
    const SCEV *SCEVA = vputils::getSCEVExprForVPValue(AddrA, PSE, &L);
    VPValue *AddrB = B->getOperand(1);
    const SCEV *SCEVB = vputils::getSCEVExprForVPValue(AddrB, PSE, &L);
    if (isa<SCEVCouldNotCompute>(SCEVA) || isa<SCEVCouldNotCompute>(SCEVB))
      return false;

    const APInt *Distance;
    ScalarEvolution &SE = *PSE.getSE();
    if (!match(SE.getMinusSCEV(SCEVA, SCEVB), m_scev_APInt(Distance)))
      return false;

    const DataLayout &DL = SE.getDataLayout();
    Type *TyA = TypeInfo.inferScalarType(A->getOperand(0));
    uint64_t SizeA = DL.getTypeStoreSize(TyA);
    Type *TyB = TypeInfo.inferScalarType(B->getOperand(0));
    uint64_t SizeB = DL.getTypeStoreSize(TyB);

    // Use the maximum store size to ensure no overlap from either direction.
    // Currently only handles fixed sizes, as it is only used for
    // replicating VPReplicateRecipes.
    uint64_t MaxStoreSize = std::max(SizeA, SizeB);

    auto VFs = B->getParent()->getPlan()->vectorFactors();
    ElementCount MaxVF = *max_element(VFs, ElementCount::isKnownLT);
    if (MaxVF.isScalable())
      return false;
    return Distance->abs().uge(
        MaxVF.multiplyCoefficientBy(MaxStoreSize).getFixedValue());
  }

public:
  SinkStoreInfo(const SmallPtrSetImpl<VPRecipeBase *> &ExcludeRecipes,
                VPReplicateRecipe &GroupLeader, PredicatedScalarEvolution &PSE,
                const Loop &L, VPTypeAnalysis &TypeInfo)
      : ExcludeRecipes(ExcludeRecipes), GroupLeader(GroupLeader), PSE(PSE),
        L(L), TypeInfo(TypeInfo) {}

  /// Return true if \p R should be skipped during alias checking, either
  /// because it's in the exclude set or because no-alias can be proven via
  /// SCEV.
  bool shouldSkip(VPRecipeBase &R) const {
    auto *Store = dyn_cast<VPReplicateRecipe>(&R);
    return ExcludeRecipes.contains(&R) ||
           (Store && isNoAliasViaDistance(Store, &GroupLeader));
  }
};

/// Check if a memory operation doesn't alias with memory operations in blocks
/// between \p FirstBB and \p LastBB using scoped noalias metadata. If
/// \p SinkInfo is std::nullopt, only recipes that may write to memory are
/// checked (for load hoisting). Otherwise recipes that both read and write
/// memory are checked, and SCEV is used to prove no-alias between the group
/// leader and other replicate recipes (for store sinking).
static bool
canHoistOrSinkWithNoAliasCheck(const MemoryLocation &MemLoc,
                               VPBasicBlock *FirstBB, VPBasicBlock *LastBB,
                               std::optional<SinkStoreInfo> SinkInfo = {}) {
  bool CheckReads = SinkInfo.has_value();
  if (!MemLoc.AATags.Scope)
    return false;

  const AAMDNodes &MemAA = MemLoc.AATags;

  for (VPBlockBase *Block = FirstBB; Block;
       Block = Block->getSingleSuccessor()) {
    assert(Block->getNumSuccessors() <= 1 &&
           "Expected at most one successor in block chain");
    auto *VPBB = cast<VPBasicBlock>(Block);
    for (VPRecipeBase &R : *VPBB) {
      if (SinkInfo && SinkInfo->shouldSkip(R))
        continue;

      // Skip recipes that don't need checking.
      if (!R.mayWriteToMemory() && !(CheckReads && R.mayReadFromMemory()))
        continue;

      auto Loc = vputils::getMemoryLocation(R);
      if (!Loc)
        // Conservatively assume aliasing for memory operations without
        // location.
        return false;

      // For reads, check if they don't alias in the reverse direction and
      // skip if so.
      if (CheckReads && R.mayReadFromMemory() &&
          !ScopedNoAliasAAResult::mayAliasInScopes(Loc->AATags.Scope,
                                                   MemAA.NoAlias))
        continue;

      // Check if the memory operations may alias in the forward direction.
      if (ScopedNoAliasAAResult::mayAliasInScopes(MemAA.Scope,
                                                  Loc->AATags.NoAlias))
        return false;
    }

    if (Block == LastBB)
      break;
  }
  return true;
}

/// Return true if we do not know how to (mechanically) hoist or sink \p R out
/// of a loop region.
static bool cannotHoistOrSinkRecipe(const VPRecipeBase &R) {
  // Assumes don't alias anything or throw; as long as they're guaranteed to
  // execute, they're safe to hoist.
  if (match(&R, m_Intrinsic<Intrinsic::assume>()))
    return false;

  // TODO: Relax checks in the future, e.g. we could also hoist reads, if their
  // memory location is not modified in the vector loop.
  if (R.mayHaveSideEffects() || R.mayReadFromMemory() || R.isPhi())
    return true;

  // Allocas cannot be hoisted.
  auto *RepR = dyn_cast<VPReplicateRecipe>(&R);
  return RepR && RepR->getOpcode() == Instruction::Alloca;
}

static bool sinkScalarOperands(VPlan &Plan) {
  auto Iter = vp_depth_first_deep(Plan.getEntry());
  bool ScalarVFOnly = Plan.hasScalarVFOnly();
  bool Changed = false;

  SetVector<std::pair<VPBasicBlock *, VPSingleDefRecipe *>> WorkList;
  auto InsertIfValidSinkCandidate = [ScalarVFOnly, &WorkList](
                                        VPBasicBlock *SinkTo, VPValue *Op) {
    auto *Candidate =
        dyn_cast_or_null<VPSingleDefRecipe>(Op->getDefiningRecipe());
    if (!Candidate)
      return;

    // We only know how to sink VPReplicateRecipes and VPScalarIVStepsRecipes
    // for now.
    if (!isa<VPReplicateRecipe, VPScalarIVStepsRecipe>(Candidate))
      return;

    if (Candidate->getParent() == SinkTo || cannotHoistOrSinkRecipe(*Candidate))
      return;

    if (auto *RepR = dyn_cast<VPReplicateRecipe>(Candidate))
      if (!ScalarVFOnly && RepR->isSingleScalar())
        return;

    WorkList.insert({SinkTo, Candidate});
  };

  // First, collect the operands of all recipes in replicate blocks as seeds for
  // sinking.
  for (VPRegionBlock *VPR : VPBlockUtils::blocksOnly<VPRegionBlock>(Iter)) {
    VPBasicBlock *EntryVPBB = VPR->getEntryBasicBlock();
    if (!VPR->isReplicator() || EntryVPBB->getSuccessors().size() != 2)
      continue;
    VPBasicBlock *VPBB = cast<VPBasicBlock>(EntryVPBB->getSuccessors().front());
    if (VPBB->getSingleSuccessor() != VPR->getExitingBasicBlock())
      continue;
    for (auto &Recipe : *VPBB)
      for (VPValue *Op : Recipe.operands())
        InsertIfValidSinkCandidate(VPBB, Op);
  }

  // Try to sink each replicate or scalar IV steps recipe in the worklist.
  for (unsigned I = 0; I != WorkList.size(); ++I) {
    VPBasicBlock *SinkTo;
    VPSingleDefRecipe *SinkCandidate;
    std::tie(SinkTo, SinkCandidate) = WorkList[I];

    // All recipe users of SinkCandidate must be in the same block SinkTo or all
    // users outside of SinkTo must only use the first lane of SinkCandidate. In
    // the latter case, we need to duplicate SinkCandidate.
    auto UsersOutsideSinkTo =
        make_filter_range(SinkCandidate->users(), [SinkTo](VPUser *U) {
          return cast<VPRecipeBase>(U)->getParent() != SinkTo;
        });
    if (any_of(UsersOutsideSinkTo, [SinkCandidate](VPUser *U) {
          return !U->usesFirstLaneOnly(SinkCandidate);
        }))
      continue;
    bool NeedsDuplicating = !UsersOutsideSinkTo.empty();

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
                                      nullptr /*Mask*/, *SinkCandidateRepR,
                                      *SinkCandidateRepR);
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
      InsertIfValidSinkCandidate(SinkTo, Op);
    Changed = true;
  }
  return Changed;
}

/// If \p R is a region with a VPBranchOnMaskRecipe in the entry block, return
/// the mask.
static VPValue *getPredicatedMask(VPRegionBlock *R) {
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
      BlockInMask, MaskDef ? MaskDef->getDebugLoc() : DebugLoc::getUnknown());
  auto *Entry =
      Plan.createVPBasicBlock(Twine(RegionName) + ".entry", BOMRecipe);

  // Replace predicated replicate recipe with a replicate recipe without a
  // mask but in the replicate region.
  auto *RecipeWithoutMask = new VPReplicateRecipe(
      PredRecipe->getUnderlyingInstr(), drop_end(PredRecipe->operands()),
      PredRecipe->isSingleScalar(), nullptr /*Mask*/, *PredRecipe, *PredRecipe,
      PredRecipe->getDebugLoc());
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
      Plan.createReplicateRegion(Entry, Exiting, RegionName);

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
    ArrayRef<Instruction *> Casts = IV->getInductionDescriptor().getCastInsts();
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
  VPRegionBlock *LoopRegion = Plan.getVectorLoopRegion();
  VPCanonicalIVPHIRecipe *CanonicalIV = LoopRegion->getCanonicalIV();
  VPWidenCanonicalIVRecipe *WidenNewIV = nullptr;
  for (VPUser *U : CanonicalIV->users()) {
    WidenNewIV = dyn_cast<VPWidenCanonicalIVRecipe>(U);
    if (WidenNewIV)
      break;
  }

  if (!WidenNewIV)
    return;

  VPBasicBlock *HeaderVPBB = LoopRegion->getEntryBasicBlock();
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
      // We are replacing a wide canonical iv with a suitable wide induction.
      // This is used to compute header mask, hence all lanes will be used and
      // we need to drop wrap flags only applying to lanes guranteed to execute
      // in the original scalar loop.
      WidenOriginalIV->dropPoisonGeneratingFlags();
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
      if (!PhiR || PhiR->getNumOperands() != 2)
        continue;
      VPUser *PhiUser = PhiR->getSingleUser();
      if (!PhiUser)
        continue;
      VPValue *Incoming = PhiR->getOperand(1);
      if (PhiUser != Incoming->getDefiningRecipe() ||
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
                    VPIRValue *StartV, VPValue *Step, DebugLoc DL,
                    VPBuilder &Builder) {
  VPRegionBlock *LoopRegion = Plan.getVectorLoopRegion();
  VPBasicBlock *HeaderVPBB = LoopRegion->getEntryBasicBlock();
  VPCanonicalIVPHIRecipe *CanonicalIV = LoopRegion->getCanonicalIV();
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

/// Scalarize a VPWidenPointerInductionRecipe by replacing it with a PtrAdd
/// (IndStart, ScalarIVSteps (0, Step)). This is used when the recipe only
/// generates scalar values.
static VPValue *
scalarizeVPWidenPointerInduction(VPWidenPointerInductionRecipe *PtrIV,
                                 VPlan &Plan, VPBuilder &Builder) {
  const InductionDescriptor &ID = PtrIV->getInductionDescriptor();
  VPIRValue *StartV = Plan.getConstantInt(ID.getStep()->getType(), 0);
  VPValue *StepV = PtrIV->getOperand(1);
  VPScalarIVStepsRecipe *Steps = createScalarIVSteps(
      Plan, InductionDescriptor::IK_IntInduction, Instruction::Add, nullptr,
      nullptr, StartV, StepV, PtrIV->getDebugLoc(), Builder);

  return Builder.createPtrAdd(PtrIV->getStartValue(), Steps,
                              PtrIV->getDebugLoc(), "next.gep");
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
      auto *Def = dyn_cast<VPRecipeWithIRFlags>(U);
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
                                          Def->operands(), /*IsUniform*/ true,
                                          /*Mask*/ nullptr, /*Flags*/ *Def);
      Clone->insertAfter(Def);
      Def->replaceAllUsesWith(Clone);
    }

    // Replace wide pointer inductions which have only their scalars used by
    // PtrAdd(IndStart, ScalarIVSteps (0, Step)).
    if (auto *PtrIV = dyn_cast<VPWidenPointerInductionRecipe>(&Phi)) {
      if (!Plan.hasScalarVFOnly() &&
          !PtrIV->onlyScalarsGenerated(Plan.hasScalableVF()))
        continue;

      VPValue *PtrAdd = scalarizeVPWidenPointerInduction(PtrIV, Plan, Builder);
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
    if (!HasOnlyVectorVFs) {
      assert(!Plan.hasScalableVF() &&
             "plans containing a scalar VF cannot also include scalable VFs");
      WideIV->replaceAllUsesWith(Steps);
    } else {
      bool HasScalableVF = Plan.hasScalableVF();
      WideIV->replaceUsesWithIf(Steps,
                                [WideIV, HasScalableVF](VPUser &U, unsigned) {
                                  if (HasScalableVF)
                                    return U.usesFirstLaneOnly(WideIV);
                                  return U.usesScalars(WideIV);
                                });
    }
  }
}

/// Check if \p VPV is an untruncated wide induction, either before or after the
/// increment. If so return the header IV (before the increment), otherwise
/// return null.
static VPWidenInductionRecipe *
getOptimizableIVOf(VPValue *VPV, PredicatedScalarEvolution &PSE) {
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
      const SCEV *IVStepSCEV = vputils::getSCEVExprForVPValue(IVStep, PSE);
      const SCEV *StepSCEV = vputils::getSCEVExprForVPValue(Step, PSE);
      ScalarEvolution &SE = *PSE.getSE();
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
                                               PredicatedScalarEvolution &PSE) {
  VPValue *Incoming, *Mask;
  if (!match(Op, m_ExtractLane(m_FirstActiveLane(m_VPValue(Mask)),
                               m_VPValue(Incoming))))
    return nullptr;

  auto *WideIV = getOptimizableIVOf(Incoming, PSE);
  if (!WideIV)
    return nullptr;

  auto *WideIntOrFp = dyn_cast<VPWidenIntOrFpInductionRecipe>(WideIV);
  if (WideIntOrFp && WideIntOrFp->getTruncInst())
    return nullptr;

  // Calculate the final index.
  VPRegionBlock *LoopRegion = Plan.getVectorLoopRegion();
  auto *CanonicalIV = LoopRegion->getCanonicalIV();
  Type *CanonicalIVType = LoopRegion->getCanonicalIVType();
  VPBuilder B(cast<VPBasicBlock>(PredVPBB));

  DebugLoc DL = cast<VPInstruction>(Op)->getDebugLoc();
  VPValue *FirstActiveLane =
      B.createNaryOp(VPInstruction::FirstActiveLane, Mask, DL);
  Type *FirstActiveLaneType = TypeInfo.inferScalarType(FirstActiveLane);
  FirstActiveLane = B.createScalarZExtOrTrunc(FirstActiveLane, CanonicalIVType,
                                              FirstActiveLaneType, DL);
  VPValue *EndValue =
      B.createNaryOp(Instruction::Add, {CanonicalIV, FirstActiveLane}, DL);

  // `getOptimizableIVOf()` always returns the pre-incremented IV, so if it
  // changed it means the exit is using the incremented value, so we need to
  // add the step.
  if (Incoming != WideIV) {
    VPValue *One = Plan.getConstantInt(CanonicalIVType, 1);
    EndValue = B.createNaryOp(Instruction::Add, {EndValue, One}, DL);
  }

  if (!WideIntOrFp || !WideIntOrFp->isCanonical()) {
    const InductionDescriptor &ID = WideIV->getInductionDescriptor();
    VPIRValue *Start = WideIV->getStartValue();
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
    DenseMap<VPValue *, VPValue *> &EndValues, PredicatedScalarEvolution &PSE) {
  VPValue *Incoming;
  if (!match(Op, m_ExtractLastLaneOfLastPart(m_VPValue(Incoming))))
    return nullptr;

  auto *WideIV = getOptimizableIVOf(Incoming, PSE);
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
    return B.createNaryOp(Instruction::Sub, {EndValue, Step},
                          DebugLoc::getUnknown(), "ind.escape");
  if (ScalarTy->isPointerTy()) {
    Type *StepTy = TypeInfo.inferScalarType(Step);
    auto *Zero = Plan.getConstantInt(StepTy, 0);
    return B.createPtrAdd(EndValue,
                          B.createNaryOp(Instruction::Sub, {Zero, Step}),
                          DebugLoc::getUnknown(), "ind.escape");
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
    PredicatedScalarEvolution &PSE) {
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
                                                  EndValues, PSE);
        else
          Escape = optimizeEarlyExitInductionUser(
              Plan, TypeInfo, PredVPBB, ExitIRI->getOperand(Idx), PSE);
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
    append_range(WorkList, R->operands());
    R->eraseFromParent();
  }
}

/// Get any instruction opcode or intrinsic ID data embedded in recipe \p R.
/// Returns an optional pair, where the first element indicates whether it is
/// an intrinsic ID.
static std::optional<std::pair<bool, unsigned>>
getOpcodeOrIntrinsicID(const VPSingleDefRecipe *R) {
  return TypeSwitch<const VPSingleDefRecipe *,
                    std::optional<std::pair<bool, unsigned>>>(R)
      .Case<VPInstruction, VPWidenRecipe, VPWidenCastRecipe, VPWidenGEPRecipe,
            VPReplicateRecipe>(
          [](auto *I) { return std::make_pair(false, I->getOpcode()); })
      .Case<VPWidenIntrinsicRecipe>([](auto *I) {
        return std::make_pair(true, I->getVectorIntrinsicID());
      })
      .Case<VPVectorPointerRecipe, VPPredInstPHIRecipe>([](auto *I) {
        // For recipes that do not directly map to LLVM IR instructions,
        // assign opcodes after the last VPInstruction opcode (which is also
        // after the last IR Instruction opcode), based on the VPDefID.
        return std::make_pair(false,
                              VPInstruction::OpsEnd + 1 + I->getVPDefID());
      })
      .Default([](auto *) { return std::nullopt; });
}

/// Try to fold \p R using InstSimplifyFolder. Will succeed and return a
/// non-nullptr VPValue for a handled opcode or intrinsic ID if corresponding \p
/// Operands are foldable live-ins.
static VPValue *tryToFoldLiveIns(VPSingleDefRecipe &R,
                                 ArrayRef<VPValue *> Operands,
                                 const DataLayout &DL,
                                 VPTypeAnalysis &TypeInfo) {
  auto OpcodeOrIID = getOpcodeOrIntrinsicID(&R);
  if (!OpcodeOrIID)
    return nullptr;

  SmallVector<Value *, 4> Ops;
  for (VPValue *Op : Operands) {
    if (!isa<VPIRValue, VPSymbolicValue>(Op))
      return nullptr;
    Value *V = Op->getUnderlyingValue();
    if (!V)
      return nullptr;
    Ops.push_back(V);
  }

  auto FoldToIRValue = [&]() -> Value * {
    InstSimplifyFolder Folder(DL);
    if (OpcodeOrIID->first) {
      if (R.getNumOperands() != 2)
        return nullptr;
      unsigned ID = OpcodeOrIID->second;
      return Folder.FoldBinaryIntrinsic(ID, Ops[0], Ops[1],
                                        TypeInfo.inferScalarType(&R));
    }
    unsigned Opcode = OpcodeOrIID->second;
    if (Instruction::isBinaryOp(Opcode))
      return Folder.FoldBinOp(static_cast<Instruction::BinaryOps>(Opcode),
                              Ops[0], Ops[1]);
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
      return Folder.FoldGEP(GEP->getSourceElementType(), Ops[0],
                            drop_begin(Ops), RFlags.getGEPNoWrapFlags());
    }
    case VPInstruction::PtrAdd:
    case VPInstruction::WidePtrAdd:
      return Folder.FoldGEP(IntegerType::getInt8Ty(TypeInfo.getContext()),
                            Ops[0], Ops[1],
                            cast<VPRecipeWithIRFlags>(R).getGEPNoWrapFlags());
    // An extract of a live-in is an extract of a broadcast, so return the
    // broadcasted element.
    case Instruction::ExtractElement:
      assert(!Ops[0]->getType()->isVectorTy() && "Live-ins should be scalar");
      return Ops[0];
    }
    return nullptr;
  };

  if (Value *V = FoldToIRValue())
    return R.getParent()->getPlan()->getOrAddLiveIn(V);
  return nullptr;
}

/// Try to simplify VPSingleDefRecipe \p Def.
static void simplifyRecipe(VPSingleDefRecipe *Def, VPTypeAnalysis &TypeInfo) {
  VPlan *Plan = Def->getParent()->getPlan();

  // Simplification of live-in IR values for SingleDef recipes using
  // InstSimplifyFolder.
  const DataLayout &DL =
      Plan->getScalarHeader()->getIRBasicBlock()->getDataLayout();
  if (VPValue *V = tryToFoldLiveIns(*Def, Def->operands(), DL, TypeInfo))
    return Def->replaceAllUsesWith(V);

  // Fold PredPHI LiveIn -> LiveIn.
  if (auto *PredPHI = dyn_cast<VPPredInstPHIRecipe>(Def)) {
    VPValue *Op = PredPHI->getOperand(0);
    if (isa<VPIRValue>(Op))
      PredPHI->replaceAllUsesWith(Op);
  }

  VPBuilder Builder(Def);
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

        unsigned ExtOpcode = match(Def->getOperand(0), m_SExt(m_VPValue()))
                                 ? Instruction::SExt
                                 : Instruction::ZExt;
        auto *Ext = Builder.createWidenCast(Instruction::CastOps(ExtOpcode), A,
                                            TruncTy);
        if (auto *UnderlyingExt = Def->getOperand(0)->getUnderlyingValue()) {
          // UnderlyingExt has distinct return type, used to retain legacy cost.
          Ext->setUnderlyingValue(UnderlyingExt);
        }
        Def->replaceAllUsesWith(Ext);
      } else if (ATy->getScalarSizeInBits() > TruncTy->getScalarSizeInBits()) {
        auto *Trunc = Builder.createWidenCast(Instruction::Trunc, A, TruncTy);
        Def->replaceAllUsesWith(Trunc);
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

  // x | 1 -> 1
  if (match(Def, m_c_BinaryOr(m_VPValue(X), m_AllOnes())))
    return Def->replaceAllUsesWith(Def->getOperand(Def->getOperand(0) == X));

  // x | 0 -> x
  if (match(Def, m_c_BinaryOr(m_VPValue(X), m_ZeroInt())))
    return Def->replaceAllUsesWith(X);

  // x & 0 -> 0
  if (match(Def, m_c_BinaryAnd(m_VPValue(X), m_ZeroInt())))
    return Def->replaceAllUsesWith(Def->getOperand(Def->getOperand(0) == X));

  // x && false -> false
  if (match(Def, m_LogicalAnd(m_VPValue(X), m_False())))
    return Def->replaceAllUsesWith(Def->getOperand(1));

  // (x && y) || (x && z) -> x && (y || z)
  if (match(Def, m_c_BinaryOr(m_LogicalAnd(m_VPValue(X), m_VPValue(Y)),
                              m_LogicalAnd(m_Deferred(X), m_VPValue(Z)))) &&
      // Simplify only if one of the operands has one use to avoid creating an
      // extra recipe.
      (!Def->getOperand(0)->hasMoreThanOneUniqueUser() ||
       !Def->getOperand(1)->hasMoreThanOneUniqueUser()))
    return Def->replaceAllUsesWith(
        Builder.createLogicalAnd(X, Builder.createOr(Y, Z)));

  // x && !x -> 0
  if (match(Def, m_LogicalAnd(m_VPValue(X), m_Not(m_Deferred(X)))))
    return Def->replaceAllUsesWith(Plan->getFalse());

  if (match(Def, m_Select(m_VPValue(), m_VPValue(X), m_Deferred(X))))
    return Def->replaceAllUsesWith(X);

  // select c, false, true -> not c
  VPValue *C;
  if (match(Def, m_Select(m_VPValue(C), m_False(), m_True())))
    return Def->replaceAllUsesWith(Builder.createNot(C));

  // select !c, x, y -> select c, y, x
  if (match(Def, m_Select(m_Not(m_VPValue(C)), m_VPValue(X), m_VPValue(Y)))) {
    Def->setOperand(0, C);
    Def->setOperand(1, Y);
    Def->setOperand(2, X);
    return;
  }

  // Reassociate (x && y) && z -> x && (y && z) if x has multiple users. With
  // tail folding it is likely that x is a header mask and can be simplified
  // further.
  if (match(Def, m_LogicalAnd(m_LogicalAnd(m_VPValue(X), m_VPValue(Y)),
                              m_VPValue(Z))) &&
      X->hasMoreThanOneUniqueUser())
    return Def->replaceAllUsesWith(
        Builder.createLogicalAnd(X, Builder.createLogicalAnd(Y, Z)));

  if (match(Def, m_c_Add(m_VPValue(A), m_ZeroInt())))
    return Def->replaceAllUsesWith(A);

  if (match(Def, m_c_Mul(m_VPValue(A), m_One())))
    return Def->replaceAllUsesWith(A);

  if (match(Def, m_c_Mul(m_VPValue(A), m_ZeroInt())))
    return Def->replaceAllUsesWith(
        Def->getOperand(0) == A ? Def->getOperand(1) : Def->getOperand(0));

  const APInt *APC;
  if (match(Def, m_c_Mul(m_VPValue(A), m_APInt(APC))) && APC->isPowerOf2())
    return Def->replaceAllUsesWith(Builder.createNaryOp(
        Instruction::Shl,
        {A, Plan->getConstantInt(APC->getBitWidth(), APC->exactLogBase2())},
        *cast<VPRecipeWithIRFlags>(Def), Def->getDebugLoc()));

  // Don't convert udiv to lshr inside a replicate region, as VPInstructions are
  // not allowed in them.
  const VPRegionBlock *ParentRegion = Def->getParent()->getParent();
  bool IsInReplicateRegion = ParentRegion && ParentRegion->isReplicator();
  if (!IsInReplicateRegion && match(Def, m_UDiv(m_VPValue(A), m_APInt(APC))) &&
      APC->isPowerOf2())
    return Def->replaceAllUsesWith(Builder.createNaryOp(
        Instruction::LShr,
        {A, Plan->getConstantInt(APC->getBitWidth(), APC->exactLogBase2())}, {},
        Def->getDebugLoc()));

  if (match(Def, m_Not(m_VPValue(A)))) {
    if (match(A, m_Not(m_VPValue(A))))
      return Def->replaceAllUsesWith(A);

    // Try to fold Not into compares by adjusting the predicate in-place.
    CmpPredicate Pred;
    if (match(A, m_Cmp(Pred, m_VPValue(), m_VPValue()))) {
      auto *Cmp = cast<VPRecipeWithIRFlags>(A);
      if (all_of(Cmp->users(),
                 match_fn(m_CombineOr(
                     m_Not(m_Specific(Cmp)),
                     m_Select(m_Specific(Cmp), m_VPValue(), m_VPValue()))))) {
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
        if (!Cmp->getDebugLoc() && Def->getDebugLoc())
          Cmp->setDebugLoc(Def->getDebugLoc());
      }
    }
  }

  // Fold any-of (fcmp uno %A, %A), (fcmp uno %B, %B), ... ->
  //      any-of (fcmp uno %A, %B), ...
  if (match(Def, m_AnyOf())) {
    SmallVector<VPValue *, 4> NewOps;
    VPRecipeBase *UnpairedCmp = nullptr;
    for (VPValue *Op : Def->operands()) {
      VPValue *X;
      if (Op->getNumUsers() > 1 ||
          !match(Op, m_SpecificCmp(CmpInst::FCMP_UNO, m_VPValue(X),
                                   m_Deferred(X)))) {
        NewOps.push_back(Op);
      } else if (!UnpairedCmp) {
        UnpairedCmp = Op->getDefiningRecipe();
      } else {
        NewOps.push_back(Builder.createFCmp(CmpInst::FCMP_UNO,
                                            UnpairedCmp->getOperand(0), X));
        UnpairedCmp = nullptr;
      }
    }

    if (UnpairedCmp)
      NewOps.push_back(UnpairedCmp->getVPSingleValue());

    if (NewOps.size() < Def->getNumOperands()) {
      VPValue *NewAnyOf = Builder.createNaryOp(VPInstruction::AnyOf, NewOps);
      return Def->replaceAllUsesWith(NewAnyOf);
    }
  }

  // Fold (fcmp uno %X, %X) or (fcmp uno %Y, %Y) -> fcmp uno %X, %Y
  // This is useful for fmax/fmin without fast-math flags, where we need to
  // check if any operand is NaN.
  if (match(Def, m_BinaryOr(m_SpecificCmp(CmpInst::FCMP_UNO, m_VPValue(X),
                                          m_Deferred(X)),
                            m_SpecificCmp(CmpInst::FCMP_UNO, m_VPValue(Y),
                                          m_Deferred(Y))))) {
    VPValue *NewCmp = Builder.createFCmp(CmpInst::FCMP_UNO, X, Y);
    return Def->replaceAllUsesWith(NewCmp);
  }

  // Remove redundant DerviedIVs, that is 0 + A * 1 -> A and 0 + 0 * x -> 0.
  if ((match(Def, m_DerivedIV(m_ZeroInt(), m_VPValue(A), m_One())) ||
       match(Def, m_DerivedIV(m_ZeroInt(), m_ZeroInt(), m_VPValue()))) &&
      TypeInfo.inferScalarType(Def->getOperand(1)) ==
          TypeInfo.inferScalarType(Def))
    return Def->replaceAllUsesWith(Def->getOperand(1));

  if (match(Def, m_VPInstruction<VPInstruction::WideIVStep>(m_VPValue(X),
                                                            m_One()))) {
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
      TypeInfo.inferScalarType(Def)->isIntegerTy(1)) {
    Def->setOperand(1, Def->getOperand(0));
    Def->setOperand(0, Y);
    return;
  }

  if (auto *Phi = dyn_cast<VPFirstOrderRecurrencePHIRecipe>(Def)) {
    if (Phi->getOperand(0) == Phi->getOperand(1))
      Phi->replaceAllUsesWith(Phi->getOperand(0));
    return;
  }

  // Look through ExtractLastLane.
  if (match(Def, m_ExtractLastLane(m_VPValue(A)))) {
    if (match(A, m_BuildVector())) {
      auto *BuildVector = cast<VPInstruction>(A);
      Def->replaceAllUsesWith(
          BuildVector->getOperand(BuildVector->getNumOperands() - 1));
      return;
    }
    if (Plan->hasScalarVFOnly())
      return Def->replaceAllUsesWith(A);
  }

  // Look through ExtractPenultimateElement (BuildVector ....).
  if (match(Def, m_ExtractPenultimateElement(m_BuildVector()))) {
    auto *BuildVector = cast<VPInstruction>(Def->getOperand(0));
    Def->replaceAllUsesWith(
        BuildVector->getOperand(BuildVector->getNumOperands() - 2));
    return;
  }

  uint64_t Idx;
  if (match(Def, m_ExtractElement(m_BuildVector(), m_ConstantInt(Idx)))) {
    auto *BuildVector = cast<VPInstruction>(Def->getOperand(0));
    Def->replaceAllUsesWith(BuildVector->getOperand(Idx));
    return;
  }

  if (match(Def, m_BuildVector()) && all_equal(Def->operands())) {
    Def->replaceAllUsesWith(
        Builder.createNaryOp(VPInstruction::Broadcast, Def->getOperand(0)));
    return;
  }

  // Look through broadcast of single-scalar when used as select conditions; in
  // that case the scalar condition can be used directly.
  if (match(Def,
            m_Select(m_Broadcast(m_VPValue(C)), m_VPValue(), m_VPValue()))) {
    assert(vputils::isSingleScalar(C) &&
           "broadcast operand must be single-scalar");
    Def->setOperand(0, C);
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

  // After unrolling, extract-lane may be used to extract values from multiple
  // scalar sources. Only simplify when extracting from a single scalar source.
  VPValue *LaneToExtract;
  if (match(Def, m_ExtractLane(m_VPValue(LaneToExtract), m_VPValue(A)))) {
    // Simplify extract-lane(%lane_num, %scalar_val) -> %scalar_val.
    if (vputils::isSingleScalar(A))
      return Def->replaceAllUsesWith(A);

    // Simplify extract-lane with single source to extract-element.
    Def->replaceAllUsesWith(Builder.createNaryOp(
        Instruction::ExtractElement, {A, LaneToExtract}, Def->getDebugLoc()));
    return;
  }

  // Hoist an invariant increment Y of a phi X, by having X start at Y.
  if (match(Def, m_c_Add(m_VPValue(X), m_VPValue(Y))) && isa<VPIRValue>(Y) &&
      isa<VPPhi>(X)) {
    auto *Phi = cast<VPPhi>(X);
    if (Phi->getOperand(1) != Def && match(Phi->getOperand(0), m_ZeroInt()) &&
        Phi->getSingleUser() == Def) {
      Phi->setOperand(0, Y);
      Def->replaceAllUsesWith(Phi);
      return;
    }
  }

  // Simplify unrolled VectorPointer without offset, or with zero offset, to
  // just the pointer operand.
  if (auto *VPR = dyn_cast<VPVectorPointerRecipe>(Def))
    if (!VPR->getOffset() || match(VPR->getOffset(), m_ZeroInt()))
      return VPR->replaceAllUsesWith(VPR->getOperand(0));

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

  if (match(Def, m_ExtractLastLane(m_Broadcast(m_VPValue(A))))) {
    Def->replaceAllUsesWith(A);
    return;
  }

  if (match(Def, m_ExtractLastLane(m_VPValue(A))) &&
      ((isa<VPInstruction>(A) && vputils::isSingleScalar(A)) ||
       (isa<VPReplicateRecipe>(A) &&
        cast<VPReplicateRecipe>(A)->isSingleScalar())) &&
      all_of(A->users(),
             [Def, A](VPUser *U) { return U->usesScalars(A) || Def == U; })) {
    return Def->replaceAllUsesWith(A);
  }

  if (Plan->getUF() == 1 && match(Def, m_ExtractLastPart(m_VPValue(A))))
    return Def->replaceAllUsesWith(A);
}

void VPlanTransforms::simplifyRecipes(VPlan &Plan) {
  ReversePostOrderTraversal<VPBlockDeepTraversalWrapper<VPBlockBase *>> RPOT(
      Plan.getEntry());
  VPTypeAnalysis TypeInfo(Plan);
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(RPOT)) {
    for (VPRecipeBase &R : make_early_inc_range(*VPBB))
      if (auto *Def = dyn_cast<VPSingleDefRecipe>(&R))
        simplifyRecipe(Def, TypeInfo);
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
      if (!isa<VPWidenRecipe, VPWidenGEPRecipe, VPReplicateRecipe,
               VPWidenStoreRecipe>(&R))
        continue;
      auto *RepR = dyn_cast<VPReplicateRecipe>(&R);
      if (RepR && (RepR->isSingleScalar() || RepR->isPredicated()))
        continue;

      // Convert an unmasked scatter with an uniform address into
      // extract-last-lane + scalar store.
      // TODO: Add a profitability check comparing the cost of a scatter vs.
      // extract + scalar store.
      auto *WidenStoreR = dyn_cast<VPWidenStoreRecipe>(&R);
      if (WidenStoreR && vputils::isSingleScalar(WidenStoreR->getAddr()) &&
          !WidenStoreR->isConsecutive()) {
        assert(!WidenStoreR->isReverse() &&
               "Not consecutive memory recipes shouldn't be reversed");
        VPValue *Mask = WidenStoreR->getMask();

        // Only convert the scatter to a scalar store if it is unmasked.
        // TODO: Support converting scatter masked by the header mask to scalar
        // store.
        if (Mask)
          continue;

        auto *Extract = new VPInstruction(VPInstruction::ExtractLastLane,
                                          {WidenStoreR->getOperand(1)});
        Extract->insertBefore(WidenStoreR);

        // TODO: Sink the scalar store recipe to middle block if possible.
        auto *ScalarStore = new VPReplicateRecipe(
            &WidenStoreR->getIngredient(), {Extract, WidenStoreR->getAddr()},
            true /*IsSingleScalar*/, nullptr /*Mask*/, {},
            *WidenStoreR /*Metadata*/);
        ScalarStore->insertBefore(WidenStoreR);
        WidenStoreR->eraseFromParent();
        continue;
      }

      auto *RepOrWidenR = dyn_cast<VPRecipeWithIRFlags>(&R);
      if (RepR && isa<StoreInst>(RepR->getUnderlyingInstr()) &&
          vputils::isSingleScalar(RepR->getOperand(1))) {
        auto *Clone = new VPReplicateRecipe(
            RepOrWidenR->getUnderlyingInstr(), RepOrWidenR->operands(),
            true /*IsSingleScalar*/, nullptr /*Mask*/, *RepR /*Flags*/,
            *RepR /*Metadata*/, RepR->getDebugLoc());
        Clone->insertBefore(RepOrWidenR);
        VPBuilder Builder(Clone);
        VPValue *ExtractOp = Clone->getOperand(0);
        if (vputils::isUniformAcrossVFsAndUFs(RepR->getOperand(1)))
          ExtractOp =
              Builder.createNaryOp(VPInstruction::ExtractLastPart, ExtractOp);
        ExtractOp =
            Builder.createNaryOp(VPInstruction::ExtractLastLane, ExtractOp);
        Clone->setOperand(0, ExtractOp);
        RepR->eraseFromParent();
        continue;
      }

      // Skip recipes that aren't single scalars.
      if (!RepOrWidenR || !vputils::isSingleScalar(RepOrWidenR))
        continue;

      // Skip recipes for which conversion to single-scalar does introduce
      // additional broadcasts. No extra broadcasts are needed, if either only
      // the scalars of the recipe are used, or at least one of the operands
      // would require a broadcast. In the latter case, the single-scalar may
      // need to be broadcasted, but another broadcast is removed.
      if (!all_of(RepOrWidenR->users(),
                  [RepOrWidenR](const VPUser *U) {
                    if (auto *VPI = dyn_cast<VPInstruction>(U)) {
                      unsigned Opcode = VPI->getOpcode();
                      if (Opcode == VPInstruction::ExtractLastLane ||
                          Opcode == VPInstruction::ExtractLastPart ||
                          Opcode == VPInstruction::ExtractPenultimateElement)
                        return true;
                    }

                    return U->usesScalars(RepOrWidenR);
                  }) &&
          none_of(RepOrWidenR->operands(), [RepOrWidenR](VPValue *Op) {
            if (Op->getSingleUser() != RepOrWidenR)
              return false;
            // Non-constant live-ins require broadcasts, while constants do not
            // need explicit broadcasts.
            auto *IRV = dyn_cast<VPIRValue>(Op);
            bool LiveInNeedsBroadcast = IRV && !isa<Constant>(IRV->getValue());
            auto *OpR = dyn_cast<VPReplicateRecipe>(Op);
            return LiveInNeedsBroadcast || (OpR && OpR->isSingleScalar());
          }))
        continue;

      auto *Clone = new VPReplicateRecipe(
          RepOrWidenR->getUnderlyingInstr(), RepOrWidenR->operands(),
          true /*IsSingleScalar*/, nullptr, *RepOrWidenR);
      Clone->insertBefore(RepOrWidenR);
      RepOrWidenR->replaceAllUsesWith(Clone);
      if (isDeadRecipe(*RepOrWidenR))
        RepOrWidenR->eraseFromParent();
    }
  }
}

/// Try to see if all of \p Blend's masks share a common value logically and'ed
/// and remove it from the masks.
static void removeCommonBlendMask(VPBlendRecipe *Blend) {
  if (Blend->isNormalized())
    return;
  VPValue *CommonEdgeMask;
  if (!match(Blend->getMask(0),
             m_LogicalAnd(m_VPValue(CommonEdgeMask), m_VPValue())))
    return;
  for (unsigned I = 0; I < Blend->getNumIncomingValues(); I++)
    if (!match(Blend->getMask(I),
               m_LogicalAnd(m_Specific(CommonEdgeMask), m_VPValue())))
      return;
  for (unsigned I = 0; I < Blend->getNumIncomingValues(); I++)
    Blend->setMask(I, Blend->getMask(I)->getDefiningRecipe()->getOperand(1));
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

      removeCommonBlendMask(Blend);

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

  const APInt *TC;
  if (!BestVF.isFixed() || !match(Plan.getTripCount(), m_APInt(TC)))
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
      ComputeBitWidth(*TC, BestVF.getKnownMinValue() * BestUF);

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
    VPUser *SingleUser = WideIV->getSingleUser();
    if (!SingleUser ||
        !match(SingleUser, m_ICmp(m_Specific(WideIV),
                                  m_Broadcast(m_Specific(
                                      Plan.getOrCreateBackedgeTakenCount())))))
      continue;

    // Update IV operands and comparison bound to use new narrower type.
    auto *NewStart = Plan.getConstantInt(NewIVTy, 0);
    WideIV->setStartValue(NewStart);
    auto *NewStep = Plan.getConstantInt(NewIVTy, 1);
    WideIV->setStepValue(NewStep);

    auto *NewBTC = new VPWidenCastRecipe(
        Instruction::Trunc, Plan.getOrCreateBackedgeTakenCount(), NewIVTy);
    Plan.getVectorPreheader()->appendRecipe(NewBTC);
    auto *Cmp = cast<VPInstruction>(WideIV->getSingleUser());
    Cmp->setOperand(1, NewBTC);

    MadeChange = true;
  }

  return MadeChange;
}

/// Return true if \p Cond is known to be true for given \p BestVF and \p
/// BestUF.
static bool isConditionTrueViaVFAndUF(VPValue *Cond, VPlan &Plan,
                                      ElementCount BestVF, unsigned BestUF,
                                      PredicatedScalarEvolution &PSE) {
  if (match(Cond, m_BinaryOr(m_VPValue(), m_VPValue())))
    return any_of(Cond->getDefiningRecipe()->operands(), [&Plan, BestVF, BestUF,
                                                          &PSE](VPValue *C) {
      return isConditionTrueViaVFAndUF(C, Plan, BestVF, BestUF, PSE);
    });

  auto *CanIV = Plan.getVectorLoopRegion()->getCanonicalIV();
  if (!match(Cond, m_SpecificICmp(CmpInst::ICMP_EQ,
                                  m_Specific(CanIV->getBackedgeValue()),
                                  m_Specific(&Plan.getVectorTripCount()))))
    return false;

  // The compare checks CanIV + VFxUF == vector trip count. The vector trip
  // count is not conveniently available as SCEV so far, so we compare directly
  // against the original trip count. This is stricter than necessary, as we
  // will only return true if the trip count == vector trip count.
  const SCEV *VectorTripCount =
      vputils::getSCEVExprForVPValue(&Plan.getVectorTripCount(), PSE);
  if (isa<SCEVCouldNotCompute>(VectorTripCount))
    VectorTripCount = vputils::getSCEVExprForVPValue(Plan.getTripCount(), PSE);
  assert(!isa<SCEVCouldNotCompute>(VectorTripCount) &&
         "Trip count SCEV must be computable");
  ScalarEvolution &SE = *PSE.getSE();
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
      auto *Ext =
          new VPWidenIntrinsicRecipe(Intrinsic::vector_extract, Ops,
                                     IntegerType::getInt1Ty(Ctx), {}, {}, DL);
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

    uint64_t Part;
    if (match(Index,
              m_VPInstruction<VPInstruction::CanonicalIVIncrementForPart>(
                  m_VPValue(), m_ConstantInt(Part))))
      Phis[Part] = Phi;
    else
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
  VPValue *ALMMultiplier = Plan.getConstantInt(64, UF);
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
  if (match(Term, m_BranchOnCount()) ||
      match(Term, m_BranchOnCond(m_Not(m_ActiveLaneMask(
                      m_VPValue(), m_VPValue(), m_VPValue()))))) {
    // Try to simplify the branch condition if VectorTC <= VF * UF when the
    // latch terminator is BranchOnCount or BranchOnCond(Not(ActiveLaneMask)).
    const SCEV *VectorTripCount =
        vputils::getSCEVExprForVPValue(&Plan.getVectorTripCount(), PSE);
    if (isa<SCEVCouldNotCompute>(VectorTripCount))
      VectorTripCount =
          vputils::getSCEVExprForVPValue(Plan.getTripCount(), PSE);
    assert(!isa<SCEVCouldNotCompute>(VectorTripCount) &&
           "Trip count SCEV must be computable");
    ScalarEvolution &SE = *PSE.getSE();
    ElementCount NumElements = BestVF.multiplyCoefficientBy(BestUF);
    const SCEV *C = SE.getElementCount(VectorTripCount->getType(), NumElements);
    if (!SE.isKnownPredicate(CmpInst::ICMP_ULE, VectorTripCount, C))
      return false;
  } else if (match(Term, m_BranchOnCond(m_VPValue(Cond))) ||
             match(Term, m_BranchOnTwoConds(m_VPValue(), m_VPValue(Cond)))) {
    // For BranchOnCond, check if we can prove the condition to be true using VF
    // and UF.
    if (!isConditionTrueViaVFAndUF(Cond, Plan, BestVF, BestUF, PSE))
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
  // TODO: fold branch-on-constant after dissolving region.
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
    SmallVector<VPBlockBase *> Exits = to_vector(VectorRegion->getSuccessors());
    VPBlockUtils::disconnectBlocks(Preheader, VectorRegion);
    for (VPBlockBase *Exit : Exits)
      VPBlockUtils::disconnectBlocks(VectorRegion, Exit);

    for (VPBlockBase *B : vp_depth_first_shallow(VectorRegion->getEntry()))
      B->setParent(nullptr);

    VPBlockUtils::connectBlocks(Preheader, Header);

    for (VPBlockBase *Exit : Exits)
      VPBlockUtils::connectBlocks(ExitingVPBB, Exit);

    // Replace terminating branch-on-two-conds with branch-on-cond to early
    // exit.
    if (Exits.size() != 1) {
      assert(match(Term, m_BranchOnTwoConds()) && Exits.size() == 2 &&
             "BranchOnTwoConds needs 2 remaining exits");
      VPBuilder(Term).createNaryOp(VPInstruction::BranchOnCond,
                                   Term->getOperand(0));
    }
    VPlanTransforms::simplifyRecipes(Plan);
  } else {
    // The vector region contains header phis for which we cannot remove the
    // loop region yet.

    // For BranchOnTwoConds, set the latch exit condition to true directly.
    if (match(Term, m_BranchOnTwoConds())) {
      Term->setOperand(1, Plan.getTrue());
      return true;
    }

    auto *BOC = new VPInstruction(VPInstruction::BranchOnCond, {Plan.getTrue()},
                                  {}, {}, Term->getDebugLoc());
    ExitingVPBB->appendRecipe(BOC);
  }

  Term->eraseFromParent();

  return true;
}

/// From the definition of llvm.experimental.get.vector.length,
/// VPInstruction::ExplicitVectorLength(%AVL) = %AVL when %AVL <= VF.
static bool simplifyKnownEVL(VPlan &Plan, ElementCount VF,
                             PredicatedScalarEvolution &PSE) {
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(
           vp_depth_first_deep(Plan.getEntry()))) {
    for (VPRecipeBase &R : *VPBB) {
      VPValue *AVL;
      if (!match(&R, m_EVL(m_VPValue(AVL))))
        continue;

      const SCEV *AVLSCEV = vputils::getSCEVExprForVPValue(AVL, PSE);
      if (isa<SCEVCouldNotCompute>(AVLSCEV))
        continue;
      ScalarEvolution &SE = *PSE.getSE();
      const SCEV *VFSCEV = SE.getElementCount(AVLSCEV->getType(), VF);
      if (!SE.isKnownPredicate(CmpInst::ICMP_ULE, AVLSCEV, VFSCEV))
        continue;

      VPValue *Trunc = VPBuilder(&R).createScalarZExtOrTrunc(
          AVL, Type::getInt32Ty(Plan.getContext()), AVLSCEV->getType(),
          R.getDebugLoc());
      R.getVPSingleValue()->replaceAllUsesWith(Trunc);
      return true;
    }
  }
  return false;
}

void VPlanTransforms::optimizeForVFAndUF(VPlan &Plan, ElementCount BestVF,
                                         unsigned BestUF,
                                         PredicatedScalarEvolution &PSE) {
  assert(Plan.hasVF(BestVF) && "BestVF is not available in Plan");
  assert(Plan.hasUF(BestUF) && "BestUF is not available in Plan");

  bool MadeChange = tryToReplaceALMWithWideALM(Plan, BestVF, BestUF);
  MadeChange |= simplifyBranchConditionForVFAndUF(Plan, BestVF, BestUF, PSE);
  MadeChange |= optimizeVectorInductionWidthForTCAndVFUF(Plan, BestVF, BestUF);
  MadeChange |= simplifyKnownEVL(Plan, BestVF, PSE);

  if (MadeChange) {
    Plan.setVF(BestVF);
    assert(Plan.getUF() == BestUF && "BestUF must match the Plan's UF");
  }
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

    if (cannotHoistOrSinkRecipe(*SinkCandidate))
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
  if (cannotHoistOrSinkRecipe(*Previous))
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
    assert((!HoistCandidate->getRegion() ||
            HoistCandidate->getRegion() == EnclosingLoopRegion) &&
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

  if (!NeedsHoisting(Previous->getVPSingleValue()))
    return true;

  // Recursively try to hoist Previous and its operands before all users of FOR.
  HoistCandidates.push_back(Previous);

  for (unsigned I = 0; I != HoistCandidates.size(); ++I) {
    VPRecipeBase *Current = HoistCandidates[I];
    assert(Current->getNumDefinedValues() == 1 &&
           "only recipes with a single defined value expected");
    if (cannotHoistOrSinkRecipe(*Current))
      return false;

    for (VPValue *Op : Current->operands()) {
      // If we reach FOR, it means the original Previous depends on some other
      // recurrence that in turn depends on FOR. If that is the case, we would
      // also need to hoist recipes involving the other FOR, which may break
      // dependencies.
      if (Op == FOR)
        return false;

      if (auto *R = NeedsHoisting(Op)) {
        // Bail out if the recipe defines multiple values.
        // TODO: Hoisting such recipes requires additional handling.
        if (R->getNumDefinedValues() != 1)
          return false;
        HoistCandidates.push_back(R);
      }
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
  VPDominatorTree VPDT(Plan);

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

    // Check for users extracting at the penultimate active lane of the FOR.
    // If only a single lane is active in the current iteration, we need to
    // select the last element from the previous iteration (from the FOR phi
    // directly).
    for (VPUser *U : RecurSplice->users()) {
      if (!match(U, m_ExtractLane(m_LastActiveLane(m_VPValue()),
                                  m_Specific(RecurSplice))))
        continue;

      VPBuilder B(cast<VPInstruction>(U));
      VPValue *LastActiveLane = cast<VPInstruction>(U)->getOperand(0);
      Type *I64Ty = Type::getInt64Ty(Plan.getContext());
      VPValue *Zero = Plan.getOrAddLiveIn(ConstantInt::get(I64Ty, 0));
      VPValue *One = Plan.getOrAddLiveIn(ConstantInt::get(I64Ty, 1));
      VPValue *PenultimateIndex =
          B.createNaryOp(Instruction::Sub, {LastActiveLane, One});
      VPValue *PenultimateLastIter =
          B.createNaryOp(VPInstruction::ExtractLane,
                         {PenultimateIndex, FOR->getBackedgeValue()});
      VPValue *LastPrevIter =
          B.createNaryOp(VPInstruction::ExtractLastLane, FOR);

      VPValue *Cmp = B.createICmp(CmpInst::ICMP_EQ, LastActiveLane, Zero);
      VPValue *Sel = B.createSelect(Cmp, LastPrevIter, PenultimateLastIter);
      cast<VPInstruction>(U)->replaceAllUsesWith(Sel);
    }
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

namespace {
struct VPCSEDenseMapInfo : public DenseMapInfo<VPSingleDefRecipe *> {
  static bool isSentinel(const VPSingleDefRecipe *Def) {
    return Def == getEmptyKey() || Def == getTombstoneKey();
  }

  /// If recipe \p R will lower to a GEP with a non-i8 source element type,
  /// return that source element type.
  static Type *getGEPSourceElementType(const VPSingleDefRecipe *R) {
    // All VPInstructions that lower to GEPs must have the i8 source element
    // type (as they are PtrAdds), so we omit it.
    return TypeSwitch<const VPSingleDefRecipe *, Type *>(R)
        .Case<VPReplicateRecipe>([](auto *I) -> Type * {
          if (auto *GEP = dyn_cast<GetElementPtrInst>(I->getUnderlyingValue()))
            return GEP->getSourceElementType();
          return nullptr;
        })
        .Case<VPVectorPointerRecipe, VPWidenGEPRecipe>(
            [](auto *I) { return I->getSourceElementType(); })
        .Default([](auto *) { return nullptr; });
  }

  /// Returns true if recipe \p Def can be safely handed for CSE.
  static bool canHandle(const VPSingleDefRecipe *Def) {
    // We can extend the list of handled recipes in the future,
    // provided we account for the data embedded in them while checking for
    // equality or hashing.
    auto C = getOpcodeOrIntrinsicID(Def);

    // The issue with (Insert|Extract)Value is that the index of the
    // insert/extract is not a proper operand in LLVM IR, and hence also not in
    // VPlan.
    if (!C || (!C->first && (C->second == Instruction::InsertValue ||
                             C->second == Instruction::ExtractValue)))
      return false;

    // During CSE, we can only handle recipes that don't read from memory: if
    // they read from memory, there could be an intervening write to memory
    // before the next instance is CSE'd, leading to an incorrect result.
    return !Def->mayReadFromMemory();
  }

  /// Hash the underlying data of \p Def.
  static unsigned getHashValue(const VPSingleDefRecipe *Def) {
    const VPlan *Plan = Def->getParent()->getPlan();
    VPTypeAnalysis TypeInfo(*Plan);
    hash_code Result = hash_combine(
        Def->getVPDefID(), getOpcodeOrIntrinsicID(Def),
        getGEPSourceElementType(Def), TypeInfo.inferScalarType(Def),
        vputils::isSingleScalar(Def), hash_combine_range(Def->operands()));
    if (auto *RFlags = dyn_cast<VPRecipeWithIRFlags>(Def))
      if (RFlags->hasPredicate())
        return hash_combine(Result, RFlags->getPredicate());
    return Result;
  }

  /// Check equality of underlying data of \p L and \p R.
  static bool isEqual(const VPSingleDefRecipe *L, const VPSingleDefRecipe *R) {
    if (isSentinel(L) || isSentinel(R))
      return L == R;
    if (L->getVPDefID() != R->getVPDefID() ||
        getOpcodeOrIntrinsicID(L) != getOpcodeOrIntrinsicID(R) ||
        getGEPSourceElementType(L) != getGEPSourceElementType(R) ||
        vputils::isSingleScalar(L) != vputils::isSingleScalar(R) ||
        !equal(L->operands(), R->operands()))
      return false;
    assert(getOpcodeOrIntrinsicID(L) && getOpcodeOrIntrinsicID(R) &&
           "must have valid opcode info for both recipes");
    if (auto *LFlags = dyn_cast<VPRecipeWithIRFlags>(L))
      if (LFlags->hasPredicate() &&
          LFlags->getPredicate() !=
              cast<VPRecipeWithIRFlags>(R)->getPredicate())
        return false;
    // Recipes in replicate regions implicitly depend on predicate. If either
    // recipe is in a replicate region, only consider them equal if both have
    // the same parent.
    const VPRegionBlock *RegionL = L->getRegion();
    const VPRegionBlock *RegionR = R->getRegion();
    if (((RegionL && RegionL->isReplicator()) ||
         (RegionR && RegionR->isReplicator())) &&
        L->getParent() != R->getParent())
      return false;
    const VPlan *Plan = L->getParent()->getPlan();
    VPTypeAnalysis TypeInfo(*Plan);
    return TypeInfo.inferScalarType(L) == TypeInfo.inferScalarType(R);
  }
};
} // end anonymous namespace

/// Perform a common-subexpression-elimination of VPSingleDefRecipes on the \p
/// Plan.
void VPlanTransforms::cse(VPlan &Plan) {
  VPDominatorTree VPDT(Plan);
  DenseMap<VPSingleDefRecipe *, VPSingleDefRecipe *, VPCSEDenseMapInfo> CSEMap;

  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(
           vp_depth_first_deep(Plan.getEntry()))) {
    for (VPRecipeBase &R : *VPBB) {
      auto *Def = dyn_cast<VPSingleDefRecipe>(&R);
      if (!Def || !VPCSEDenseMapInfo::canHandle(Def))
        continue;
      if (VPSingleDefRecipe *V = CSEMap.lookup(Def)) {
        // V must dominate Def for a valid replacement.
        if (!VPDT.dominates(V->getParent(), VPBB))
          continue;
        // Only keep flags present on both V and Def.
        if (auto *RFlags = dyn_cast<VPRecipeWithIRFlags>(V))
          RFlags->intersectFlags(*cast<VPRecipeWithIRFlags>(Def));
        Def->replaceAllUsesWith(V);
        continue;
      }
      CSEMap[Def] = Def;
    }
  }
}

/// Move loop-invariant recipes out of the vector loop region in \p Plan.
static void licm(VPlan &Plan) {
  VPBasicBlock *Preheader = Plan.getVectorPreheader();

  // Hoist any loop invariant recipes from the vector loop region to the
  // preheader. Preform a shallow traversal of the vector loop region, to
  // exclude recipes in replicate regions. Since the top-level blocks in the
  // vector loop region are guaranteed to execute if the vector pre-header is,
  // we don't need to check speculation safety.
  VPRegionBlock *LoopRegion = Plan.getVectorLoopRegion();
  assert(Preheader->getSingleSuccessor() == LoopRegion &&
         "Expected vector prehader's successor to be the vector loop region");
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(
           vp_depth_first_shallow(LoopRegion->getEntry()))) {
    for (VPRecipeBase &R : make_early_inc_range(*VPBB)) {
      if (cannotHoistOrSinkRecipe(R))
        continue;
      if (any_of(R.operands(), [](VPValue *Op) {
            return !Op->isDefinedOutsideLoopRegions();
          }))
        continue;
      R.moveBefore(*Preheader, Preheader->end());
    }
  }
}

void VPlanTransforms::truncateToMinimalBitwidths(
    VPlan &Plan, const MapVector<Instruction *, uint64_t> &MinBWs) {
  if (Plan.hasScalarVFOnly())
    return;
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
               VPWidenLoadRecipe, VPWidenIntrinsicRecipe>(&R))
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
      unsigned StartIdx =
          match(&R, m_Select(m_VPValue(), m_VPValue(), m_VPValue())) ? 1 : 0;
      for (unsigned Idx = StartIdx; Idx != R.getNumOperands(); ++Idx) {
        auto *Op = R.getOperand(Idx);
        unsigned OpSizeInBits =
            TypeInfo.inferScalarType(Op)->getScalarSizeInBits();
        if (OpSizeInBits == NewResSizeInBits)
          continue;
        assert(OpSizeInBits > NewResSizeInBits && "nothing to truncate");
        auto [ProcessedIter, IterIsEmpty] = ProcessedTruncs.try_emplace(Op);
        if (!IterIsEmpty) {
          R.setOperand(Idx, ProcessedIter->second);
          continue;
        }

        VPBuilder Builder;
        if (isa<VPIRValue>(Op))
          Builder.setInsertPoint(PH);
        else
          Builder.setInsertPoint(&R);
        VPWidenCastRecipe *NewOp =
            Builder.createWidenCast(Instruction::Trunc, Op, NewResTy);
        ProcessedIter->second = NewOp;
        R.setOperand(Idx, NewOp);
      }

    }
  }
}

void VPlanTransforms::removeBranchOnConst(VPlan &Plan) {
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(
           vp_depth_first_shallow(Plan.getEntry()))) {
    VPValue *Cond;
    // Skip blocks that are not terminated by BranchOnCond.
    if (VPBB->empty() || !match(&VPBB->back(), m_BranchOnCond(m_VPValue(Cond))))
      continue;

    assert(VPBB->getNumSuccessors() == 2 &&
           "Two successors expected for BranchOnCond");
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
  runPass(removeDeadRecipes, Plan);
  runPass(simplifyBlends, Plan);
  runPass(legalizeAndOptimizeInductions, Plan);
  runPass(narrowToSingleScalarRecipes, Plan);
  runPass(removeRedundantExpandSCEVRecipes, Plan);
  runPass(simplifyRecipes, Plan);
  runPass(removeBranchOnConst, Plan);
  runPass(removeDeadRecipes, Plan);

  runPass(createAndOptimizeReplicateRegions, Plan);
  runPass(hoistInvariantLoads, Plan);
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
  auto *CanonicalIVPHI = TopRegion->getCanonicalIV();
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
  VPValue *ALMMultiplier =
      Plan.getConstantInt(TopRegion->getCanonicalIVType(), 1);
  auto *EntryALM = Builder.createNaryOp(VPInstruction::ActiveLaneMask,
                                        {EntryIncrement, TC, ALMMultiplier}, DL,
                                        "active.lane.mask.entry");

  // Now create the ActiveLaneMaskPhi recipe in the main loop using the
  // preheader ActiveLaneMask instruction.
  auto *LaneMaskPhi =
      new VPActiveLaneMaskPHIRecipe(EntryALM, DebugLoc::getUnknown());
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
  VPRegionBlock *LoopRegion = Plan.getVectorLoopRegion();
  SmallVector<VPValue *> WideCanonicalIVs;
  auto *FoundWidenCanonicalIVUser = find_if(
      LoopRegion->getCanonicalIV()->users(), IsaPred<VPWidenCanonicalIVRecipe>);
  assert(count_if(LoopRegion->getCanonicalIV()->users(),
                  IsaPred<VPWidenCanonicalIVRecipe>) <= 1 &&
         "Must have at most one VPWideCanonicalIVRecipe");
  if (FoundWidenCanonicalIVUser !=
      LoopRegion->getCanonicalIV()->users().end()) {
    auto *WideCanonicalIV =
        cast<VPWidenCanonicalIVRecipe>(*FoundWidenCanonicalIVUser);
    WideCanonicalIVs.push_back(WideCanonicalIV);
  }

  // Also include VPWidenIntOrFpInductionRecipes that represent a widened
  // version of the canonical induction.
  VPBasicBlock *HeaderVPBB = LoopRegion->getEntryBasicBlock();
  for (VPRecipeBase &Phi : HeaderVPBB->phis()) {
    auto *WidenOriginalIV = dyn_cast<VPWidenIntOrFpInductionRecipe>(&Phi);
    if (WidenOriginalIV && WidenOriginalIV->isCanonical())
      WideCanonicalIVs.push_back(WidenOriginalIV);
  }

  // Walk users of wide canonical IVs and find the single compare of the form
  // (ICMP_ULE, WideCanonicalIV, backedge-taken-count).
  VPSingleDefRecipe *HeaderMask = nullptr;
  for (auto *Wide : WideCanonicalIVs) {
    for (VPUser *U : Wide->users()) {
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

  VPRegionBlock *LoopRegion = Plan.getVectorLoopRegion();
  auto *FoundWidenCanonicalIVUser = find_if(
      LoopRegion->getCanonicalIV()->users(), IsaPred<VPWidenCanonicalIVRecipe>);
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
        ConstantInt::get(LoopRegion->getCanonicalIVType(), 1));
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

template <typename Op0_t, typename Op1_t> struct RemoveMask_match {
  Op0_t In;
  Op1_t &Out;

  RemoveMask_match(const Op0_t &In, Op1_t &Out) : In(In), Out(Out) {}

  template <typename OpTy> bool match(OpTy *V) const {
    if (m_Specific(In).match(V)) {
      Out = nullptr;
      return true;
    }
    return m_LogicalAnd(m_Specific(In), m_VPValue(Out)).match(V);
  }
};

/// Match a specific mask \p In, or a combination of it (logical-and In, Out).
/// Returns the remaining part \p Out if so, or nullptr otherwise.
template <typename Op0_t, typename Op1_t>
static inline RemoveMask_match<Op0_t, Op1_t> m_RemoveMask(const Op0_t &In,
                                                          Op1_t &Out) {
  return RemoveMask_match<Op0_t, Op1_t>(In, Out);
}

/// Try to optimize a \p CurRecipe masked by \p HeaderMask to a corresponding
/// EVL-based recipe without the header mask. Returns nullptr if no EVL-based
/// recipe could be created.
/// \p HeaderMask  Header Mask.
/// \p CurRecipe   Recipe to be transform.
/// \p TypeInfo    VPlan-based type analysis.
/// \p EVL         The explicit vector length parameter of vector-predication
/// intrinsics.
static VPRecipeBase *optimizeMaskToEVL(VPValue *HeaderMask,
                                       VPRecipeBase &CurRecipe,
                                       VPTypeAnalysis &TypeInfo, VPValue &EVL) {
  VPlan *Plan = CurRecipe.getParent()->getPlan();
  DebugLoc DL = CurRecipe.getDebugLoc();
  VPValue *Addr, *Mask, *EndPtr;

  /// Adjust any end pointers so that they point to the end of EVL lanes not VF.
  auto AdjustEndPtr = [&CurRecipe, &EVL](VPValue *EndPtr) {
    auto *EVLEndPtr = cast<VPVectorEndPointerRecipe>(EndPtr)->clone();
    EVLEndPtr->insertBefore(&CurRecipe);
    EVLEndPtr->setOperand(1, &EVL);
    return EVLEndPtr;
  };

  if (match(&CurRecipe,
            m_MaskedLoad(m_VPValue(Addr), m_RemoveMask(HeaderMask, Mask))) &&
      !cast<VPWidenLoadRecipe>(CurRecipe).isReverse())
    return new VPWidenLoadEVLRecipe(cast<VPWidenLoadRecipe>(CurRecipe), Addr,
                                    EVL, Mask);

  VPValue *ReversedVal;
  if (match(&CurRecipe, m_Reverse(m_VPValue(ReversedVal))) &&
      match(ReversedVal,
            m_MaskedLoad(m_VPValue(EndPtr), m_RemoveMask(HeaderMask, Mask))) &&
      match(EndPtr, m_VecEndPtr(m_VPValue(Addr), m_Specific(&Plan->getVF()))) &&
      cast<VPWidenLoadRecipe>(ReversedVal)->isReverse()) {
    auto *LoadR = new VPWidenLoadEVLRecipe(
        *cast<VPWidenLoadRecipe>(ReversedVal), AdjustEndPtr(EndPtr), EVL, Mask);
    LoadR->insertBefore(&CurRecipe);
    return new VPWidenIntrinsicRecipe(
        Intrinsic::experimental_vp_reverse, {LoadR, Plan->getTrue(), &EVL},
        TypeInfo.inferScalarType(LoadR), {}, {}, DL);
  }

  VPValue *StoredVal;
  if (match(&CurRecipe, m_MaskedStore(m_VPValue(Addr), m_VPValue(StoredVal),
                                      m_RemoveMask(HeaderMask, Mask))) &&
      !cast<VPWidenStoreRecipe>(CurRecipe).isReverse())
    return new VPWidenStoreEVLRecipe(cast<VPWidenStoreRecipe>(CurRecipe), Addr,
                                     StoredVal, EVL, Mask);

  if (match(&CurRecipe,
            m_MaskedStore(m_VPValue(EndPtr), m_Reverse(m_VPValue(ReversedVal)),
                          m_RemoveMask(HeaderMask, Mask))) &&
      match(EndPtr, m_VecEndPtr(m_VPValue(Addr), m_Specific(&Plan->getVF()))) &&
      cast<VPWidenStoreRecipe>(CurRecipe).isReverse()) {
    auto *NewReverse = new VPWidenIntrinsicRecipe(
        Intrinsic::experimental_vp_reverse,
        {ReversedVal, Plan->getTrue(), &EVL},
        TypeInfo.inferScalarType(ReversedVal), {}, {}, DL);
    NewReverse->insertBefore(&CurRecipe);
    return new VPWidenStoreEVLRecipe(cast<VPWidenStoreRecipe>(CurRecipe),
                                     AdjustEndPtr(EndPtr), NewReverse, EVL,
                                     Mask);
  }

  if (auto *Rdx = dyn_cast<VPReductionRecipe>(&CurRecipe))
    if (Rdx->isConditional() &&
        match(Rdx->getCondOp(), m_RemoveMask(HeaderMask, Mask)))
      return new VPReductionEVLRecipe(*Rdx, EVL, Mask);

  if (auto *Interleave = dyn_cast<VPInterleaveRecipe>(&CurRecipe))
    if (Interleave->getMask() &&
        match(Interleave->getMask(), m_RemoveMask(HeaderMask, Mask)))
      return new VPInterleaveEVLRecipe(*Interleave, EVL, Mask);

  VPValue *LHS, *RHS;
  if (match(&CurRecipe,
            m_Select(m_Specific(HeaderMask), m_VPValue(LHS), m_VPValue(RHS))))
    return new VPWidenIntrinsicRecipe(
        Intrinsic::vp_merge, {Plan->getTrue(), LHS, RHS, &EVL},
        TypeInfo.inferScalarType(LHS), {}, {}, DL);

  if (match(&CurRecipe, m_Select(m_RemoveMask(HeaderMask, Mask), m_VPValue(LHS),
                                 m_VPValue(RHS))))
    return new VPWidenIntrinsicRecipe(
        Intrinsic::vp_merge, {Mask, LHS, RHS, &EVL},
        TypeInfo.inferScalarType(LHS), {}, {}, DL);

  if (match(&CurRecipe, m_LastActiveLane(m_Specific(HeaderMask)))) {
    Type *Ty = TypeInfo.inferScalarType(CurRecipe.getVPSingleValue());
    VPValue *ZExt =
        VPBuilder(&CurRecipe).createScalarCast(Instruction::ZExt, &EVL, Ty, DL);
    return new VPInstruction(Instruction::Sub,
                             {ZExt, Plan->getConstantInt(Ty, 1)}, {}, {}, DL);
  }

  return nullptr;
}

/// Optimize away any EVL-based header masks to VP intrinsic based recipes.
/// The transforms here need to preserve the original semantics.
void VPlanTransforms::optimizeEVLMasks(VPlan &Plan) {
  // Find the EVL-based header mask if it exists: icmp ult step-vector, EVL
  VPValue *HeaderMask = nullptr, *EVL = nullptr;
  for (VPRecipeBase &R : *Plan.getVectorLoopRegion()->getEntryBasicBlock()) {
    if (match(&R, m_SpecificICmp(CmpInst::ICMP_ULT, m_StepVector(),
                                 m_VPValue(EVL))) &&
        match(EVL, m_EVL(m_VPValue()))) {
      HeaderMask = R.getVPSingleValue();
      break;
    }
  }
  if (!HeaderMask)
    return;

  VPTypeAnalysis TypeInfo(Plan);
  SmallVector<VPRecipeBase *> OldRecipes;
  for (VPUser *U : collectUsersRecursively(HeaderMask)) {
    VPRecipeBase *R = cast<VPRecipeBase>(U);
    if (auto *NewR = optimizeMaskToEVL(HeaderMask, *R, TypeInfo, *EVL)) {
      NewR->insertBefore(R);
      for (auto [Old, New] :
           zip_equal(R->definedValues(), NewR->definedValues()))
        Old->replaceAllUsesWith(New);
      OldRecipes.push_back(R);
    }
  }
  // Erase old recipes at the end so we don't invalidate TypeInfo.
  for (VPRecipeBase *R : reverse(OldRecipes)) {
    SmallVector<VPValue *> PossiblyDead(R->operands());
    R->eraseFromParent();
    for (VPValue *Op : PossiblyDead)
      recursivelyDeleteDeadRecipes(Op);
  }
}

/// After replacing the canonical IV with a EVL-based IV, fixup recipes that use
/// VF to use the EVL instead to avoid incorrect updates on the penultimate
/// iteration.
static void fixupVFUsersForEVL(VPlan &Plan, VPValue &EVL) {
  VPTypeAnalysis TypeInfo(Plan);
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
                [&LoopRegion, &Plan](VPUser *U) {
                  return match(U,
                               m_c_Add(m_Specific(LoopRegion->getCanonicalIV()),
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
        TypeInfo.inferScalarType(MaxEVL), DebugLoc::getUnknown());

    Builder.setInsertPoint(Header, Header->getFirstNonPhi());
    VPValue *PrevEVL = Builder.createScalarPhi(
        {MaxEVL, &EVL}, DebugLoc::getUnknown(), "prev.evl");

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
            {V1, V2, Imm, Plan.getTrue(), PrevEVL, &EVL},
            TypeInfo.inferScalarType(R.getVPSingleValue()), {}, {},
            R.getDebugLoc());
        VPSplice->insertBefore(&R);
        R.getVPSingleValue()->replaceAllUsesWith(VPSplice);
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
}

/// Converts a tail folded vector loop region to step by
/// VPInstruction::ExplicitVectorLength elements instead of VF elements each
/// iteration.
///
/// - Add a VPEVLBasedIVPHIRecipe and related recipes to \p Plan and
///   replaces all uses except the canonical IV increment of
///   VPCanonicalIVPHIRecipe with a VPEVLBasedIVPHIRecipe.
///   VPCanonicalIVPHIRecipe is used only for loop iterations counting after
///   this transformation.
///
/// - The header mask is replaced with a header mask based on the EVL.
///
/// - Plans with FORs have a new phi added to keep track of the EVL of the
///   previous iteration, and VPFirstOrderRecurrencePHIRecipes are replaced with
///   @llvm.vp.splice.
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
  if (Plan.hasScalarVFOnly())
    return;
  VPRegionBlock *LoopRegion = Plan.getVectorLoopRegion();
  VPBasicBlock *Header = LoopRegion->getEntryBasicBlock();

  auto *CanonicalIVPHI = LoopRegion->getCanonicalIV();
  auto *CanIVTy = LoopRegion->getCanonicalIVType();
  VPValue *StartV = CanonicalIVPHI->getStartValue();

  // Create the ExplicitVectorLengthPhi recipe in the main loop.
  auto *EVLPhi = new VPEVLBasedIVPHIRecipe(StartV, DebugLoc::getUnknown());
  EVLPhi->insertAfter(CanonicalIVPHI);
  VPBuilder Builder(Header, Header->getFirstNonPhi());
  // Create the AVL (application vector length), starting from TC -> 0 in steps
  // of EVL.
  VPPhi *AVLPhi = Builder.createScalarPhi(
      {Plan.getTripCount()}, DebugLoc::getCompilerGenerated(), "avl");
  VPValue *AVL = AVLPhi;

  if (MaxSafeElements) {
    // Support for MaxSafeDist for correct loop emission.
    VPValue *AVLSafe = Plan.getConstantInt(CanIVTy, *MaxSafeElements);
    VPValue *Cmp = Builder.createICmp(ICmpInst::ICMP_ULT, AVL, AVLSafe);
    AVL = Builder.createSelect(Cmp, AVL, AVLSafe, DebugLoc::getUnknown(),
                               "safe_avl");
  }
  auto *VPEVL = Builder.createNaryOp(VPInstruction::ExplicitVectorLength, AVL,
                                     DebugLoc::getUnknown(), "evl");

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

  fixupVFUsersForEVL(Plan, *VPEVL);
  removeDeadRecipes(Plan);

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
  // Before: (branch-on-cond (icmp eq EVLIVInc, VectorTripCount))
  // After: (branch-on-cond icmp eq AVLNext, 0)
  VPBasicBlock *LatchExiting =
      HeaderVPBB->getPredecessors()[1]->getEntryBasicBlock();
  auto *LatchExitingBr = cast<VPInstruction>(LatchExiting->getTerminator());
  if (match(LatchExitingBr, m_BranchOnCond(m_True())))
    return;

  assert(match(LatchExitingBr, m_BranchOnCond(m_SpecificCmp(
                                   CmpInst::ICMP_EQ, m_VPValue(EVLIncrement),
                                   m_Specific(&Plan.getVectorTripCount())))) &&
         "Expected BranchOnCond with ICmp comparing EVL increment with vector "
         "trip count");

  Type *AVLTy = VPTypeAnalysis(Plan).inferScalarType(AVLNext);
  VPBuilder Builder(LatchExitingBr);
  LatchExitingBr->setOperand(0,
                             Builder.createICmp(CmpInst::ICMP_EQ, AVLNext,
                                                Plan.getConstantInt(AVLTy, 0)));
}

void VPlanTransforms::replaceSymbolicStrides(
    VPlan &Plan, PredicatedScalarEvolution &PSE,
    const DenseMap<Value *, const SCEV *> &StridesMap) {
  // Replace VPValues for known constant strides guaranteed by predicate scalar
  // evolution.
  auto CanUseVersionedStride = [&Plan](VPUser &U, unsigned) {
    auto *R = cast<VPRecipeBase>(&U);
    return R->getRegion() ||
           R->getParent() == Plan.getVectorLoopRegion()->getSinglePredecessor();
  };
  ValueToSCEVMapTy RewriteMap;
  for (const SCEV *Stride : StridesMap.values()) {
    using namespace SCEVPatternMatch;
    auto *StrideV = cast<SCEVUnknown>(Stride)->getValue();
    const APInt *StrideConst;
    if (!match(PSE.getSCEV(StrideV), m_scev_APInt(StrideConst)))
      // Only handle constant strides for now.
      continue;

    auto *CI = Plan.getConstantInt(*StrideConst);
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
      VPValue *CI = Plan.getConstantInt(C);
      StrideVPV->replaceUsesWithIf(CI, CanUseVersionedStride);
    }
    RewriteMap[StrideV] = PSE.getSCEV(StrideV);
  }

  for (VPRecipeBase &R : *Plan.getEntry()) {
    auto *ExpSCEV = dyn_cast<VPExpandSCEVRecipe>(&R);
    if (!ExpSCEV)
      continue;
    const SCEV *ScevExpr = ExpSCEV->getSCEV();
    auto *NewSCEV =
        SCEVParameterRewriter::rewrite(ScevExpr, *PSE.getSE(), RewriteMap);
    if (NewSCEV != ScevExpr) {
      VPValue *NewExp = vputils::getOrCreateVPValueForSCEVExpr(Plan, NewSCEV);
      ExpSCEV->replaceAllUsesWith(NewExp);
      if (Plan.getTripCount() == ExpSCEV)
        Plan.resetTripCount(NewExp);
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
  VPDominatorTree VPDT(Plan);
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
      VPValue *OffsetVPV = Plan.getConstantInt(-Offset);
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
  VPIRFlags Flags = *WidenIVR;
  if (ID.getKind() == InductionDescriptor::IK_IntInduction) {
    AddOp = Instruction::Add;
    MulOp = Instruction::Mul;
  } else {
    AddOp = ID.getInductionOpcode();
    MulOp = Instruction::FMul;
  }

  // If the phi is truncated, truncate the start and step values.
  VPBuilder Builder(Plan->getVectorPreheader());
  Type *StepTy = TypeInfo.inferScalarType(Step);
  if (Ty->getScalarSizeInBits() < StepTy->getScalarSizeInBits()) {
    assert(StepTy->isIntegerTy() && "Truncation requires an integer type");
    Step = Builder.createScalarCast(Instruction::Trunc, Step, Ty, DL);
    Start = Builder.createScalarCast(Instruction::Trunc, Start, Ty, DL);
    // Truncation doesn't preserve WrapFlags.
    Flags.dropPoisonGeneratingFlags();
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
  Init = Builder.createNaryOp(AddOp, {SplatStart, Init}, Flags,
                              DebugLoc::getUnknown(), "induction");

  // Create the widened phi of the vector IV.
  auto *WidePHI = new VPWidenPHIRecipe(WidenIVR->getPHINode(), Init,
                                       WidenIVR->getDebugLoc(), "vec.ind");
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
  Offset = Builder.createOverflowingOp(Instruction::Mul, {Offset, Step});
  VPValue *PtrAdd = Builder.createNaryOp(
      VPInstruction::WidePtrAdd, {ScalarPtrPhi, Offset}, DL, "vector.gep");
  R->replaceAllUsesWith(PtrAdd);

  // Create the backedge value for the scalar pointer phi.
  VPBasicBlock *ExitingBB = Plan->getVectorLoopRegion()->getExitingBasicBlock();
  Builder.setInsertPoint(ExitingBB, ExitingBB->getTerminator()->getIterator());
  VF = Builder.createScalarZExtOrTrunc(VF, StepTy, TypeInfo.inferScalarType(VF),
                                       DL);
  VPValue *Inc = Builder.createOverflowingOp(Instruction::Mul, {Step, VF});

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

void VPlanTransforms::expandBranchOnTwoConds(VPlan &Plan) {
  SmallVector<VPInstruction *> WorkList;
  // The transform runs after dissolving loop regions, so all VPBasicBlocks
  // terminated with BranchOnTwoConds are reached via a shallow traversal.
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(
           vp_depth_first_shallow(Plan.getEntry()))) {
    if (!VPBB->empty() && match(&VPBB->back(), m_BranchOnTwoConds()))
      WorkList.push_back(cast<VPInstruction>(&VPBB->back()));
  }

  // Expand BranchOnTwoConds instructions into explicit CFG with two new
  // single-condition branches:
  // 1. A branch that replaces BranchOnTwoConds, jumps to the first successor if
  //    the first condition is true, and otherwise jumps to a new interim block.
  // 2. A branch that ends the interim block, jumps to the second successor if
  //    the second condition is true, and otherwise jumps to the third
  //    successor.
  for (VPInstruction *Br : WorkList) {
    assert(Br->getNumOperands() == 2 &&
           "BranchOnTwoConds must have exactly 2 conditions");
    DebugLoc DL = Br->getDebugLoc();
    VPBasicBlock *BrOnTwoCondsBB = Br->getParent();
    const auto Successors = to_vector(BrOnTwoCondsBB->getSuccessors());
    assert(Successors.size() == 3 &&
           "BranchOnTwoConds must have exactly 3 successors");

    for (VPBlockBase *Succ : Successors)
      VPBlockUtils::disconnectBlocks(BrOnTwoCondsBB, Succ);

    VPValue *Cond0 = Br->getOperand(0);
    VPValue *Cond1 = Br->getOperand(1);
    VPBlockBase *Succ0 = Successors[0];
    VPBlockBase *Succ1 = Successors[1];
    VPBlockBase *Succ2 = Successors[2];
    assert(!Succ0->getParent() && !Succ1->getParent() && !Succ2->getParent() &&
           !BrOnTwoCondsBB->getParent() && "regions must already be dissolved");

    VPBasicBlock *InterimBB =
        Plan.createVPBasicBlock(BrOnTwoCondsBB->getName() + ".interim");

    VPBuilder(BrOnTwoCondsBB)
        .createNaryOp(VPInstruction::BranchOnCond, {Cond0}, DL);
    VPBlockUtils::connectBlocks(BrOnTwoCondsBB, Succ0);
    VPBlockUtils::connectBlocks(BrOnTwoCondsBB, InterimBB);

    VPBuilder(InterimBB).createNaryOp(VPInstruction::BranchOnCond, {Cond1}, DL);
    VPBlockUtils::connectBlocks(InterimBB, Succ1);
    VPBlockUtils::connectBlocks(InterimBB, Succ2);
    Br->eraseFromParent();
  }
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
        // If the recipe only generates scalars, scalarize it instead of
        // expanding it.
        if (WidenIVR->onlyScalarsGenerated(Plan.hasScalableVF())) {
          VPBuilder Builder(WidenIVR);
          VPValue *PtrAdd =
              scalarizeVPWidenPointerInduction(WidenIVR, Plan, Builder);
          WidenIVR->replaceAllUsesWith(PtrAdd);
          ToRemove.push_back(WidenIVR);
          continue;
        }
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

      // Expand LastActiveLane into Not + FirstActiveLane + Sub.
      auto *LastActiveL = dyn_cast<VPInstruction>(&R);
      if (LastActiveL &&
          LastActiveL->getOpcode() == VPInstruction::LastActiveLane) {
        // Create Not(Mask) for all operands.
        SmallVector<VPValue *, 2> NotMasks;
        for (VPValue *Op : LastActiveL->operands()) {
          VPValue *NotMask = Builder.createNot(Op, LastActiveL->getDebugLoc());
          NotMasks.push_back(NotMask);
        }

        // Create FirstActiveLane on the inverted masks.
        VPValue *FirstInactiveLane = Builder.createNaryOp(
            VPInstruction::FirstActiveLane, NotMasks,
            LastActiveL->getDebugLoc(), "first.inactive.lane");

        // Subtract 1 to get the last active lane.
        VPValue *One = Plan.getOrAddLiveIn(
            ConstantInt::get(Type::getInt64Ty(Plan.getContext()), 1));
        VPValue *LastLane = Builder.createNaryOp(
            Instruction::Sub, {FirstInactiveLane, One},
            LastActiveL->getDebugLoc(), "last.active.lane");

        LastActiveL->replaceAllUsesWith(LastLane);
        ToRemove.push_back(LastActiveL);
        continue;
      }

      // Lower BranchOnCount to ICmp + BranchOnCond.
      VPValue *IV, *TC;
      if (match(&R, m_BranchOnCount(m_VPValue(IV), m_VPValue(TC)))) {
        auto *BranchOnCountInst = cast<VPInstruction>(&R);
        DebugLoc DL = BranchOnCountInst->getDebugLoc();
        VPValue *Cond = Builder.createICmp(CmpInst::ICMP_EQ, IV, TC, DL);
        Builder.createNaryOp(VPInstruction::BranchOnCond, Cond, DL);
        ToRemove.push_back(BranchOnCountInst);
        continue;
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

      assert(!match(ScalarStep, m_One()) && "Expected non-unit scalar-step");
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
  auto *MiddleVPBB = cast<VPBasicBlock>(LatchVPBB->getSuccessors()[0]);
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
  assert(match(EarlyExitingVPBB->getTerminator(), m_BranchOnCond()) &&
         "Terminator must be be BranchOnCond");
  VPValue *CondOfEarlyExitingVPBB =
      EarlyExitingVPBB->getTerminator()->getOperand(0);
  auto *CondToEarlyExit = TrueSucc == EarlyExitVPBB
                              ? CondOfEarlyExitingVPBB
                              : Builder.createNot(CondOfEarlyExitingVPBB);

  // Create a BranchOnTwoConds in the latch that branches to:
  // [0] vector.early.exit, [1] middle block, [2] header (continue looping).
  VPValue *IsEarlyExitTaken =
      Builder.createNaryOp(VPInstruction::AnyOf, {CondToEarlyExit});
  VPBasicBlock *VectorEarlyExitVPBB =
      Plan.createVPBasicBlock("vector.early.exit");
  VectorEarlyExitVPBB->setParent(EarlyExitVPBB->getParent());

  VPBlockUtils::connectBlocks(VectorEarlyExitVPBB, EarlyExitVPBB);

  // Update the exit phis in the early exit block.
  VPBuilder MiddleBuilder(MiddleVPBB);
  VPBuilder EarlyExitB(VectorEarlyExitVPBB);
  for (VPRecipeBase &R : EarlyExitVPBB->phis()) {
    auto *ExitIRI = cast<VPIRPhi>(&R);
    // Early exit operand should always be last, i.e., 0 if EarlyExitVPBB has
    // a single predecessor and 1 if it has two.
    unsigned EarlyExitIdx = ExitIRI->getNumOperands() - 1;
    if (ExitIRI->getNumOperands() != 1) {
      // The first of two operands corresponds to the latch exit, via MiddleVPBB
      // predecessor. Extract its final lane.
      ExitIRI->extractLastLaneOfLastPartOfFirstOperand(MiddleBuilder);
    }

    VPValue *IncomingFromEarlyExit = ExitIRI->getOperand(EarlyExitIdx);
    if (!isa<VPIRValue>(IncomingFromEarlyExit)) {
      // Update the incoming value from the early exit.
      VPValue *FirstActiveLane = EarlyExitB.createNaryOp(
          VPInstruction::FirstActiveLane, {CondToEarlyExit},
          DebugLoc::getUnknown(), "first.active.lane");
      IncomingFromEarlyExit = EarlyExitB.createNaryOp(
          VPInstruction::ExtractLane, {FirstActiveLane, IncomingFromEarlyExit},
          DebugLoc::getUnknown(), "early.exit.value");
      ExitIRI->setOperand(EarlyExitIdx, IncomingFromEarlyExit);
    }
  }

  // Replace the conditional branch controlling the latch exit from the vector
  // loop with a multi-conditional branch exiting to vector early exit if the
  // early exit has been taken, exiting to middle block if the original
  // condition of the vector latch is true, otherwise continuing back to header.
  auto *LatchExitingBranch = cast<VPInstruction>(LatchVPBB->getTerminator());
  assert(LatchExitingBranch->getOpcode() == VPInstruction::BranchOnCount &&
         "Unexpected terminator");
  auto *IsLatchExitTaken =
      Builder.createICmp(CmpInst::ICMP_EQ, LatchExitingBranch->getOperand(0),
                         LatchExitingBranch->getOperand(1));

  DebugLoc LatchDL = LatchExitingBranch->getDebugLoc();
  LatchExitingBranch->eraseFromParent();

  Builder.setInsertPoint(LatchVPBB);
  Builder.createNaryOp(VPInstruction::BranchOnTwoConds,
                       {IsEarlyExitTaken, IsLatchExitTaken}, LatchDL);
  LatchVPBB->clearSuccessors();
  LatchVPBB->setSuccessors({VectorEarlyExitVPBB, MiddleVPBB, HeaderVPBB});
  VectorEarlyExitVPBB->setPredecessors({LatchVPBB});
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
  auto IsExtendedRedValidAndClampRange =
      [&](unsigned Opcode, Instruction::CastOps ExtOpc, Type *SrcTy) -> bool {
    return LoopVectorizationPlanner::getDecisionAndClampRange(
        [&](ElementCount VF) {
          auto *SrcVecTy = cast<VectorType>(toVectorTy(SrcTy, VF));
          TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput;

          InstructionCost ExtRedCost;
          InstructionCost ExtCost =
              cast<VPWidenCastRecipe>(VecOp)->computeCost(VF, Ctx);
          InstructionCost RedCost = Red->computeCost(VF, Ctx);

          if (Red->isPartialReduction()) {
            TargetTransformInfo::PartialReductionExtendKind ExtKind =
                TargetTransformInfo::getPartialReductionExtendKind(ExtOpc);
            // FIXME: Move partial reduction creation, costing and clamping
            // here from LoopVectorize.cpp.
            ExtRedCost = Ctx.TTI.getPartialReductionCost(
                Opcode, SrcTy, nullptr, RedTy, VF, ExtKind,
                llvm::TargetTransformInfo::PR_None, std::nullopt, Ctx.CostKind);
          } else {
            ExtRedCost = Ctx.TTI.getExtendedReductionCost(
                Opcode, ExtOpc == Instruction::CastOps::ZExt, RedTy, SrcVecTy,
                Red->getFastMathFlags(), CostKind);
          }
          return ExtRedCost.isValid() && ExtRedCost < ExtCost + RedCost;
        },
        Range);
  };

  VPValue *A;
  // Match reduce(ext)).
  if (match(VecOp, m_ZExtOrSExt(m_VPValue(A))) &&
      IsExtendedRedValidAndClampRange(
          RecurrenceDescriptor::getOpcode(Red->getRecurrenceKind()),
          cast<VPWidenCastRecipe>(VecOp)->getOpcode(),
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
  if (Opcode != Instruction::Add && Opcode != Instruction::Sub)
    return nullptr;

  Type *RedTy = Ctx.Types.inferScalarType(Red);

  // Clamp the range if using multiply-accumulate-reduction is profitable.
  auto IsMulAccValidAndClampRange =
      [&](VPWidenRecipe *Mul, VPWidenCastRecipe *Ext0, VPWidenCastRecipe *Ext1,
          VPWidenCastRecipe *OuterExt) -> bool {
    return LoopVectorizationPlanner::getDecisionAndClampRange(
        [&](ElementCount VF) {
          TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput;
          Type *SrcTy =
              Ext0 ? Ctx.Types.inferScalarType(Ext0->getOperand(0)) : RedTy;
          InstructionCost MulAccCost;

          if (Red->isPartialReduction()) {
            Type *SrcTy2 =
                Ext1 ? Ctx.Types.inferScalarType(Ext1->getOperand(0)) : nullptr;
            // FIXME: Move partial reduction creation, costing and clamping
            // here from LoopVectorize.cpp.
            MulAccCost = Ctx.TTI.getPartialReductionCost(
                Opcode, SrcTy, SrcTy2, RedTy, VF,
                Ext0 ? TargetTransformInfo::getPartialReductionExtendKind(
                           Ext0->getOpcode())
                     : TargetTransformInfo::PR_None,
                Ext1 ? TargetTransformInfo::getPartialReductionExtendKind(
                           Ext1->getOpcode())
                     : TargetTransformInfo::PR_None,
                Mul->getOpcode(), CostKind);
          } else {
            // Only partial reductions support mixed extends at the moment.
            if (Ext0 && Ext1 && Ext0->getOpcode() != Ext1->getOpcode())
              return false;

            bool IsZExt =
                !Ext0 || Ext0->getOpcode() == Instruction::CastOps::ZExt;
            auto *SrcVecTy = cast<VectorType>(toVectorTy(SrcTy, VF));
            MulAccCost = Ctx.TTI.getMulAccReductionCost(IsZExt, Opcode, RedTy,
                                                        SrcVecTy, CostKind);
          }

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
  VPRecipeBase *Sub = nullptr;
  VPValue *A, *B;
  VPValue *Tmp = nullptr;
  // Sub reductions could have a sub between the add reduction and vec op.
  if (match(VecOp, m_Sub(m_ZeroInt(), m_VPValue(Tmp)))) {
    Sub = VecOp->getDefiningRecipe();
    VecOp = Tmp;
  }

  // If ValB is a constant and can be safely extended, truncate it to the same
  // type as ExtA's operand, then extend it to the same type as ExtA. This
  // creates two uniform extends that can more easily be matched by the rest of
  // the bundling code. The ExtB reference, ValB and operand 1 of Mul are all
  // replaced with the new extend of the constant.
  auto ExtendAndReplaceConstantOp = [&Ctx](VPWidenCastRecipe *ExtA,
                                           VPWidenCastRecipe *&ExtB,
                                           VPValue *&ValB, VPWidenRecipe *Mul) {
    if (!ExtA || ExtB || !isa<VPIRValue>(ValB))
      return;
    Type *NarrowTy = Ctx.Types.inferScalarType(ExtA->getOperand(0));
    Instruction::CastOps ExtOpc = ExtA->getOpcode();
    const APInt *Const;
    if (!match(ValB, m_APInt(Const)) ||
        !llvm::canConstantBeExtended(
            Const, NarrowTy, TTI::getPartialReductionExtendKind(ExtOpc)))
      return;
    // The truncate ensures that the type of each extended operand is the
    // same, and it's been proven that the constant can be extended from
    // NarrowTy safely. Necessary since ExtA's extended operand would be
    // e.g. an i8, while the const will likely be an i32. This will be
    // elided by later optimisations.
    VPBuilder Builder(Mul);
    auto *Trunc =
        Builder.createWidenCast(Instruction::CastOps::Trunc, ValB, NarrowTy);
    Type *WideTy = Ctx.Types.inferScalarType(ExtA);
    ValB = ExtB = Builder.createWidenCast(ExtOpc, Trunc, WideTy);
    Mul->setOperand(1, ExtB);
  };

  // Try to match reduce.add(mul(...)).
  if (match(VecOp, m_Mul(m_VPValue(A), m_VPValue(B)))) {
    auto *RecipeA = dyn_cast_if_present<VPWidenCastRecipe>(A);
    auto *RecipeB = dyn_cast_if_present<VPWidenCastRecipe>(B);
    auto *Mul = cast<VPWidenRecipe>(VecOp);

    // Convert reduce.add(mul(ext, const)) to reduce.add(mul(ext, ext(const)))
    ExtendAndReplaceConstantOp(RecipeA, RecipeB, B, Mul);

    // Match reduce.add/sub(mul(ext, ext)).
    if (RecipeA && RecipeB && match(RecipeA, m_ZExtOrSExt(m_VPValue())) &&
        match(RecipeB, m_ZExtOrSExt(m_VPValue())) &&
        IsMulAccValidAndClampRange(Mul, RecipeA, RecipeB, nullptr)) {
      if (Sub)
        return new VPExpressionRecipe(RecipeA, RecipeB, Mul,
                                      cast<VPWidenRecipe>(Sub), Red);
      return new VPExpressionRecipe(RecipeA, RecipeB, Mul, Red);
    }
    // TODO: Add an expression type for this variant with a negated mul
    if (!Sub && IsMulAccValidAndClampRange(Mul, nullptr, nullptr, nullptr))
      return new VPExpressionRecipe(Mul, Red);
  }
  // TODO: Add an expression type for negated versions of other expression
  // variants.
  if (Sub)
    return nullptr;

  // Match reduce.add(ext(mul(A, B))).
  if (match(VecOp, m_ZExtOrSExt(m_Mul(m_VPValue(A), m_VPValue(B))))) {
    auto *Ext = cast<VPWidenCastRecipe>(VecOp);
    auto *Mul = cast<VPWidenRecipe>(Ext->getOperand(0));
    auto *Ext0 = dyn_cast_if_present<VPWidenCastRecipe>(A);
    auto *Ext1 = dyn_cast_if_present<VPWidenCastRecipe>(B);

    // reduce.add(ext(mul(ext, const)))
    // -> reduce.add(ext(mul(ext, ext(const))))
    ExtendAndReplaceConstantOp(Ext0, Ext1, B, Mul);

    // reduce.add(ext(mul(ext(A), ext(B))))
    // -> reduce.add(mul(wider_ext(A), wider_ext(B)))
    // The inner extends must either have the same opcode as the outer extend or
    // be the same, in which case the multiply can never result in a negative
    // value and the outer extend can be folded away by doing wider
    // extends for the operands of the mul.
    if (Ext0 && Ext1 &&
        (Ext->getOpcode() == Ext0->getOpcode() || Ext0 == Ext1) &&
        Ext0->getOpcode() == Ext1->getOpcode() &&
        IsMulAccValidAndClampRange(Mul, Ext0, Ext1, Ext) && Mul->hasOneUse()) {
      auto *NewExt0 = new VPWidenCastRecipe(
          Ext0->getOpcode(), Ext0->getOperand(0), Ext->getResultType(), nullptr,
          *Ext0, *Ext0, Ext0->getDebugLoc());
      NewExt0->insertBefore(Ext0);

      VPWidenCastRecipe *NewExt1 = NewExt0;
      if (Ext0 != Ext1) {
        NewExt1 = new VPWidenCastRecipe(Ext1->getOpcode(), Ext1->getOperand(0),
                                        Ext->getResultType(), nullptr, *Ext1,
                                        *Ext1, Ext1->getDebugLoc());
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
  VPDominatorTree VPDT(Plan);
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
        (isa<VPIRValue>(VPV) && isa<Constant>(VPV->getLiveInIRValue())))
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

void VPlanTransforms::hoistInvariantLoads(VPlan &Plan) {
  VPRegionBlock *LoopRegion = Plan.getVectorLoopRegion();

  // Collect candidate loads with invariant addresses and noalias scopes
  // metadata and memory-writing recipes with noalias metadata.
  SmallVector<std::pair<VPRecipeBase *, MemoryLocation>> CandidateLoads;
  SmallVector<MemoryLocation> Stores;
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(
           vp_depth_first_shallow(LoopRegion->getEntry()))) {
    for (VPRecipeBase &R : *VPBB) {
      // Only handle single-scalar replicated loads with invariant addresses.
      if (auto *RepR = dyn_cast<VPReplicateRecipe>(&R)) {
        if (RepR->isPredicated() || !RepR->isSingleScalar() ||
            RepR->getOpcode() != Instruction::Load)
          continue;

        VPValue *Addr = RepR->getOperand(0);
        if (Addr->isDefinedOutsideLoopRegions()) {
          MemoryLocation Loc = *vputils::getMemoryLocation(*RepR);
          if (!Loc.AATags.Scope)
            continue;
          CandidateLoads.push_back({RepR, Loc});
        }
      }
      if (R.mayWriteToMemory()) {
        auto Loc = vputils::getMemoryLocation(R);
        if (!Loc || !Loc->AATags.Scope || !Loc->AATags.NoAlias)
          return;
        Stores.push_back(*Loc);
      }
    }
  }

  VPBasicBlock *Preheader = Plan.getVectorPreheader();
  for (auto &[LoadRecipe, LoadLoc] : CandidateLoads) {
    // Hoist the load to the preheader if it doesn't alias with any stores
    // according to the noalias metadata. Other loads should have been hoisted
    // by other passes
    const AAMDNodes &LoadAA = LoadLoc.AATags;
    if (all_of(Stores, [&](const MemoryLocation &StoreLoc) {
          return !ScopedNoAliasAAResult::mayAliasInScopes(
              LoadAA.Scope, StoreLoc.AATags.NoAlias);
        })) {
      LoadRecipe->moveBefore(*Preheader, Preheader->getFirstNonPhi());
    }
  }
}

// Collect common metadata from a group of replicate recipes by intersecting
// metadata from all recipes in the group.
static VPIRMetadata getCommonMetadata(ArrayRef<VPReplicateRecipe *> Recipes) {
  VPIRMetadata CommonMetadata = *Recipes.front();
  for (VPReplicateRecipe *Recipe : drop_begin(Recipes))
    CommonMetadata.intersect(*Recipe);
  return CommonMetadata;
}

template <unsigned Opcode>
static SmallVector<SmallVector<VPReplicateRecipe *, 4>>
collectComplementaryPredicatedMemOps(VPlan &Plan,
                                     PredicatedScalarEvolution &PSE,
                                     const Loop *L) {
  static_assert(Opcode == Instruction::Load || Opcode == Instruction::Store,
                "Only Load and Store opcodes supported");
  constexpr bool IsLoad = (Opcode == Instruction::Load);
  VPRegionBlock *LoopRegion = Plan.getVectorLoopRegion();
  VPTypeAnalysis TypeInfo(Plan);

  // Group predicated operations by their address SCEV.
  DenseMap<const SCEV *, SmallVector<VPReplicateRecipe *>> RecipesByAddress;
  for (VPBlockBase *Block : vp_depth_first_shallow(LoopRegion->getEntry())) {
    auto *VPBB = cast<VPBasicBlock>(Block);
    for (VPRecipeBase &R : *VPBB) {
      auto *RepR = dyn_cast<VPReplicateRecipe>(&R);
      if (!RepR || RepR->getOpcode() != Opcode || !RepR->isPredicated())
        continue;

      // For loads, operand 0 is address; for stores, operand 1 is address.
      VPValue *Addr = RepR->getOperand(IsLoad ? 0 : 1);
      const SCEV *AddrSCEV = vputils::getSCEVExprForVPValue(Addr, PSE, L);
      if (!isa<SCEVCouldNotCompute>(AddrSCEV))
        RecipesByAddress[AddrSCEV].push_back(RepR);
    }
  }

  // For each address, collect operations with the same or complementary masks.
  SmallVector<SmallVector<VPReplicateRecipe *, 4>> AllGroups;
  auto GetLoadStoreValueType = [&](VPReplicateRecipe *Recipe) {
    return TypeInfo.inferScalarType(IsLoad ? Recipe : Recipe->getOperand(0));
  };
  for (auto &[Addr, Recipes] : RecipesByAddress) {
    if (Recipes.size() < 2)
      continue;

    // Collect groups with the same or complementary masks.
    for (VPReplicateRecipe *&RecipeI : Recipes) {
      if (!RecipeI)
        continue;

      VPValue *MaskI = RecipeI->getMask();
      Type *TypeI = GetLoadStoreValueType(RecipeI);
      SmallVector<VPReplicateRecipe *, 4> Group;
      Group.push_back(RecipeI);
      RecipeI = nullptr;

      // Find all operations with the same or complementary masks.
      bool HasComplementaryMask = false;
      for (VPReplicateRecipe *&RecipeJ : Recipes) {
        if (!RecipeJ)
          continue;

        VPValue *MaskJ = RecipeJ->getMask();
        Type *TypeJ = GetLoadStoreValueType(RecipeJ);
        if (TypeI == TypeJ) {
          // Check if any operation in the group has a complementary mask with
          // another, that is M1 == NOT(M2) or M2 == NOT(M1).
          HasComplementaryMask |= match(MaskI, m_Not(m_Specific(MaskJ))) ||
                                  match(MaskJ, m_Not(m_Specific(MaskI)));
          Group.push_back(RecipeJ);
          RecipeJ = nullptr;
        }
      }

      if (HasComplementaryMask) {
        assert(Group.size() >= 2 && "must have at least 2 entries");
        AllGroups.push_back(std::move(Group));
      }
    }
  }

  return AllGroups;
}

// Find the recipe with minimum alignment in the group.
template <typename InstType>
static VPReplicateRecipe *
findRecipeWithMinAlign(ArrayRef<VPReplicateRecipe *> Group) {
  return *min_element(Group, [](VPReplicateRecipe *A, VPReplicateRecipe *B) {
    return cast<InstType>(A->getUnderlyingInstr())->getAlign() <
           cast<InstType>(B->getUnderlyingInstr())->getAlign();
  });
}

void VPlanTransforms::hoistPredicatedLoads(VPlan &Plan,
                                           PredicatedScalarEvolution &PSE,
                                           const Loop *L) {
  auto Groups =
      collectComplementaryPredicatedMemOps<Instruction::Load>(Plan, PSE, L);
  if (Groups.empty())
    return;

  VPDominatorTree VPDT(Plan);

  // Process each group of loads.
  for (auto &Group : Groups) {
    // Sort loads by dominance order, with earliest (most dominating) first.
    sort(Group, [&VPDT](VPReplicateRecipe *A, VPReplicateRecipe *B) {
      return VPDT.properlyDominates(A, B);
    });

    // Try to use the earliest (most dominating) load to replace all others.
    VPReplicateRecipe *EarliestLoad = Group[0];
    VPBasicBlock *FirstBB = EarliestLoad->getParent();
    VPBasicBlock *LastBB = Group.back()->getParent();

    // Check that the load doesn't alias with stores between first and last.
    auto LoadLoc = vputils::getMemoryLocation(*EarliestLoad);
    if (!LoadLoc || !canHoistOrSinkWithNoAliasCheck(*LoadLoc, FirstBB, LastBB))
      continue;

    // Collect common metadata from all loads in the group.
    VPIRMetadata CommonMetadata = getCommonMetadata(Group);

    // Find the load with minimum alignment to use.
    auto *LoadWithMinAlign = findRecipeWithMinAlign<LoadInst>(Group);

    // Create an unpredicated version of the earliest load with common
    // metadata.
    auto *UnpredicatedLoad = new VPReplicateRecipe(
        LoadWithMinAlign->getUnderlyingInstr(), {EarliestLoad->getOperand(0)},
        /*IsSingleScalar=*/false, /*Mask=*/nullptr, *EarliestLoad,
        CommonMetadata);

    UnpredicatedLoad->insertBefore(EarliestLoad);

    // Replace all loads in the group with the unpredicated load.
    for (VPReplicateRecipe *Load : Group) {
      Load->replaceAllUsesWith(UnpredicatedLoad);
      Load->eraseFromParent();
    }
  }
}

static bool
canSinkStoreWithNoAliasCheck(ArrayRef<VPReplicateRecipe *> StoresToSink,
                             PredicatedScalarEvolution &PSE, const Loop &L,
                             VPTypeAnalysis &TypeInfo) {
  auto StoreLoc = vputils::getMemoryLocation(*StoresToSink.front());
  if (!StoreLoc || !StoreLoc->AATags.Scope)
    return false;

  // When sinking a group of stores, all members of the group alias each other.
  // Skip them during the alias checks.
  SmallPtrSet<VPRecipeBase *, 4> StoresToSinkSet(StoresToSink.begin(),
                                                 StoresToSink.end());

  VPBasicBlock *FirstBB = StoresToSink.front()->getParent();
  VPBasicBlock *LastBB = StoresToSink.back()->getParent();
  SinkStoreInfo SinkInfo(StoresToSinkSet, *StoresToSink[0], PSE, L, TypeInfo);
  return canHoistOrSinkWithNoAliasCheck(*StoreLoc, FirstBB, LastBB, SinkInfo);
}

void VPlanTransforms::sinkPredicatedStores(VPlan &Plan,
                                           PredicatedScalarEvolution &PSE,
                                           const Loop *L) {
  auto Groups =
      collectComplementaryPredicatedMemOps<Instruction::Store>(Plan, PSE, L);
  if (Groups.empty())
    return;

  VPDominatorTree VPDT(Plan);
  VPTypeAnalysis TypeInfo(Plan);

  for (auto &Group : Groups) {
    sort(Group, [&VPDT](VPReplicateRecipe *A, VPReplicateRecipe *B) {
      return VPDT.properlyDominates(A, B);
    });

    if (!canSinkStoreWithNoAliasCheck(Group, PSE, *L, TypeInfo))
      continue;

    // Use the last (most dominated) store's location for the unconditional
    // store.
    VPReplicateRecipe *LastStore = Group.back();
    VPBasicBlock *InsertBB = LastStore->getParent();

    // Collect common alias metadata from all stores in the group.
    VPIRMetadata CommonMetadata = getCommonMetadata(Group);

    // Build select chain for stored values.
    VPValue *SelectedValue = Group[0]->getOperand(0);
    VPBuilder Builder(InsertBB, LastStore->getIterator());

    for (unsigned I = 1; I < Group.size(); ++I) {
      VPValue *Mask = Group[I]->getMask();
      VPValue *Value = Group[I]->getOperand(0);
      SelectedValue = Builder.createSelect(Mask, Value, SelectedValue,
                                           Group[I]->getDebugLoc());
    }

    // Find the store with minimum alignment to use.
    auto *StoreWithMinAlign = findRecipeWithMinAlign<StoreInst>(Group);

    // Create unconditional store with selected value and common metadata.
    auto *UnpredicatedStore =
        new VPReplicateRecipe(StoreWithMinAlign->getUnderlyingInstr(),
                              {SelectedValue, LastStore->getOperand(1)},
                              /*IsSingleScalar=*/false,
                              /*Mask=*/nullptr, *LastStore, CommonMetadata);
    UnpredicatedStore->insertBefore(*InsertBB, LastStore->getIterator());

    // Remove all predicated stores from the group.
    for (VPReplicateRecipe *Store : Group)
      Store->eraseFromParent();
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
      !isa<VPIRValue>(TC))
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
      Instruction::Sub, {Plan.getTripCount(), Plan.getConstantInt(TCTy, 1)},
      DebugLoc::getCompilerGenerated(), "trip.count.minus.1");
  BTC->replaceAllUsesWith(TCMO);
}

void VPlanTransforms::materializePacksAndUnpacks(VPlan &Plan) {
  if (Plan.hasScalarVFOnly())
    return;

  VPTypeAnalysis TypeInfo(Plan);
  VPRegionBlock *LoopRegion = Plan.getVectorLoopRegion();
  auto VPBBsOutsideLoopRegion = VPBlockUtils::blocksOnly<VPBasicBlock>(
      vp_depth_first_shallow(Plan.getEntry()));
  auto VPBBsInsideLoopRegion = VPBlockUtils::blocksOnly<VPBasicBlock>(
      vp_depth_first_shallow(LoopRegion->getEntry()));
  // Materialize Build(Struct)Vector for all replicating VPReplicateRecipes and
  // VPInstructions, excluding ones in replicate regions. Those are not
  // materialized explicitly yet. Those vector users are still handled in
  // VPReplicateRegion::execute(), via shouldPack().
  // TODO: materialize build vectors for replicating recipes in replicating
  // regions.
  for (VPBasicBlock *VPBB :
       concat<VPBasicBlock *>(VPBBsOutsideLoopRegion, VPBBsInsideLoopRegion)) {
    for (VPRecipeBase &R : make_early_inc_range(*VPBB)) {
      if (!isa<VPReplicateRecipe, VPInstruction>(&R))
        continue;
      auto *DefR = cast<VPRecipeWithIRFlags>(&R);
      auto UsesVectorOrInsideReplicateRegion = [DefR, LoopRegion](VPUser *U) {
        VPRegionBlock *ParentRegion = cast<VPRecipeBase>(U)->getRegion();
        return !U->usesScalars(DefR) || ParentRegion != LoopRegion;
      };
      if ((isa<VPReplicateRecipe>(DefR) &&
           cast<VPReplicateRecipe>(DefR)->isSingleScalar()) ||
          (isa<VPInstruction>(DefR) &&
           (vputils::onlyFirstLaneUsed(DefR) ||
            !cast<VPInstruction>(DefR)->doesGeneratePerAllLanes())) ||
          none_of(DefR->users(), UsesVectorOrInsideReplicateRegion))
        continue;

      Type *ScalarTy = TypeInfo.inferScalarType(DefR);
      unsigned Opcode = ScalarTy->isStructTy()
                            ? VPInstruction::BuildStructVector
                            : VPInstruction::BuildVector;
      auto *BuildVector = new VPInstruction(Opcode, {DefR});
      BuildVector->insertAfter(DefR);

      DefR->replaceUsesWithIf(
          BuildVector, [BuildVector, &UsesVectorOrInsideReplicateRegion](
                           VPUser &U, unsigned) {
            return &U != BuildVector && UsesVectorOrInsideReplicateRegion(&U);
          });
    }
  }

  // Create explicit VPInstructions to convert vectors to scalars. The current
  // implementation is conservative - it may miss some cases that may or may not
  // be vector values. TODO: introduce Unpacks speculatively - remove them later
  // if they are known to operate on scalar values.
  for (VPBasicBlock *VPBB : VPBBsInsideLoopRegion) {
    for (VPRecipeBase &R : make_early_inc_range(*VPBB)) {
      if (isa<VPReplicateRecipe, VPInstruction, VPScalarIVStepsRecipe,
              VPDerivedIVRecipe, VPCanonicalIVPHIRecipe>(&R))
        continue;
      for (VPValue *Def : R.definedValues()) {
        // Skip recipes that are single-scalar or only have their first lane
        // used.
        // TODO: The Defs skipped here may or may not be vector values.
        // Introduce Unpacks, and remove them later, if they are guaranteed to
        // produce scalar values.
        if (vputils::isSingleScalar(Def) || vputils::onlyFirstLaneUsed(Def))
          continue;

        // At the moment, we create unpacks only for scalar users outside
        // replicate regions. Recipes inside replicate regions still extract the
        // required lanes implicitly.
        // TODO: Remove once replicate regions are unrolled completely.
        auto IsCandidateUnpackUser = [Def](VPUser *U) {
          VPRegionBlock *ParentRegion = cast<VPRecipeBase>(U)->getRegion();
          return U->usesScalars(Def) &&
                 (!ParentRegion || !ParentRegion->isReplicator());
        };
        if (none_of(Def->users(), IsCandidateUnpackUser))
          continue;

        auto *Unpack = new VPInstruction(VPInstruction::Unpack, {Def});
        if (R.isPhi())
          Unpack->insertBefore(*VPBB, VPBB->getFirstNonPhi());
        else
          Unpack->insertAfter(&R);
        Def->replaceUsesWithIf(Unpack,
                               [&IsCandidateUnpackUser](VPUser &U, unsigned) {
                                 return IsCandidateUnpackUser(&U);
                               });
      }
    }
  }
}

void VPlanTransforms::materializeVectorTripCount(VPlan &Plan,
                                                 VPBasicBlock *VectorPHVPBB,
                                                 bool TailByMasking,
                                                 bool RequiresScalarEpilogue) {
  VPSymbolicValue &VectorTC = Plan.getVectorTripCount();
  // There's nothing to do if there are no users of the vector trip count or its
  // IR value has already been set.
  if (VectorTC.getNumUsers() == 0 || VectorTC.getUnderlyingValue())
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
        {TC, Builder.createNaryOp(Instruction::Sub,
                                  {Step, Plan.getConstantInt(TCTy, 1)})},
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
    VPValue *IsZero =
        Builder.createICmp(CmpInst::ICMP_EQ, R, Plan.getConstantInt(TCTy, 0));
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

  VPValue *UF = Plan.getConstantInt(TCTy, Plan.getUF());
  VPValue *MulByUF = Builder.createOverflowingOp(
      Instruction::Mul, {RuntimeVF, UF}, {true, false});
  VFxUF.replaceAllUsesWith(MulByUF);
}

DenseMap<const SCEV *, Value *>
VPlanTransforms::expandSCEVs(VPlan &Plan, ScalarEvolution &SE) {
  SCEVExpander Expander(SE, "induction", /*PreserveLCSSA=*/false);

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
  VPValue *Member0Op = WideMember0->getOperand(OpIdx);
  VPRecipeBase *Member0OpR = Member0Op->getDefiningRecipe();
  if (!Member0OpR)
    return Member0Op == OpV;
  if (auto *W = dyn_cast<VPWidenLoadRecipe>(Member0OpR))
    return !W->getMask() && Member0Op == OpV;
  if (auto *IR = dyn_cast<VPInterleaveRecipe>(Member0OpR))
    return IR->getInterleaveGroup()->isFull() && IR->getVPValue(Idx) == OpV;
  return false;
}

/// Returns true if \p IR is a full interleave group with factor and number of
/// members both equal to \p VF. The interleave group must also access the full
/// vector width \p VectorRegWidth.
static bool isConsecutiveInterleaveGroup(VPInterleaveRecipe *InterleaveR,
                                         ElementCount VF,
                                         VPTypeAnalysis &TypeInfo,
                                         TypeSize VectorRegWidth) {
  if (!InterleaveR || InterleaveR->getMask())
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

  unsigned VFMin = VF.getKnownMinValue();
  TypeSize GroupSize = TypeSize::get(
      GroupElementTy->getScalarSizeInBits() * VFMin, VF.isScalable());
  const auto *IG = InterleaveR->getInterleaveGroup();
  return IG->getFactor() == VFMin && IG->getNumMembers() == VFMin &&
         GroupSize == VectorRegWidth;
}

/// Returns true if \p VPValue is a narrow VPValue.
static bool isAlreadyNarrow(VPValue *VPV) {
  if (isa<VPIRValue>(VPV))
    return true;
  auto *RepR = dyn_cast<VPReplicateRecipe>(VPV);
  return RepR && RepR->isSingleScalar();
}

// Convert a wide recipe defining a VPValue \p V feeding an interleave group to
// a narrow variant.
static VPValue *
narrowInterleaveGroupOp(VPValue *V, SmallPtrSetImpl<VPValue *> &NarrowedOps) {
  auto *R = V->getDefiningRecipe();
  if (!R || NarrowedOps.contains(V))
    return V;

  if (isAlreadyNarrow(V))
    return V;

  if (auto *WideMember0 = dyn_cast<VPWidenRecipe>(R)) {
    for (unsigned Idx = 0, E = WideMember0->getNumOperands(); Idx != E; ++Idx)
      WideMember0->setOperand(
          Idx,
          narrowInterleaveGroupOp(WideMember0->getOperand(Idx), NarrowedOps));
    return V;
  }

  if (auto *LoadGroup = dyn_cast<VPInterleaveRecipe>(R)) {
    // Narrow interleave group to wide load, as transformed VPlan will only
    // process one original iteration.
    auto *LI = cast<LoadInst>(LoadGroup->getInterleaveGroup()->getInsertPos());
    auto *L = new VPWidenLoadRecipe(
        *LI, LoadGroup->getAddr(), LoadGroup->getMask(), /*Consecutive=*/true,
        /*Reverse=*/false, {}, LoadGroup->getDebugLoc());
    L->insertBefore(LoadGroup);
    NarrowedOps.insert(L);
    return L;
  }

  if (auto *RepR = dyn_cast<VPReplicateRecipe>(R)) {
    assert(RepR->isSingleScalar() &&
           isa<LoadInst>(RepR->getUnderlyingInstr()) &&
           "must be a single scalar load");
    NarrowedOps.insert(RepR);
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
                                  /*Mask*/ nullptr, {}, *WideLoad);
  N->insertBefore(WideLoad);
  NarrowedOps.insert(N);
  return N;
}

void VPlanTransforms::narrowInterleaveGroups(VPlan &Plan, ElementCount VF,
                                             TypeSize VectorRegWidth) {
  VPRegionBlock *VectorLoop = Plan.getVectorLoopRegion();
  if (!VectorLoop || VectorLoop->getEntry()->getNumSuccessors() != 0)
    return;

  VPTypeAnalysis TypeInfo(Plan);

  SmallVector<VPInterleaveRecipe *> StoreGroups;
  for (auto &R : *VectorLoop->getEntryBasicBlock()) {
    if (isa<VPCanonicalIVPHIRecipe>(&R))
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
    if (!isConsecutiveInterleaveGroup(InterleaveR, VF, TypeInfo,
                                      VectorRegWidth))
      return;

    // Skip read interleave groups.
    if (InterleaveR->getStoredValues().empty())
      continue;

    // Narrow interleave groups, if all operands are already matching narrow
    // ops.
    auto *Member0 = InterleaveR->getStoredValues()[0];
    if (isAlreadyNarrow(Member0) &&
        all_of(InterleaveR->getStoredValues(), equal_to(Member0))) {
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
    auto *WideMember0 =
        dyn_cast_or_null<VPWidenRecipe>(InterleaveR->getStoredValues()[0]);
    if (!WideMember0)
      return;
    for (const auto &[I, V] : enumerate(InterleaveR->getStoredValues())) {
      auto *R = dyn_cast_or_null<VPWidenRecipe>(V);
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
  SmallPtrSet<VPValue *, 4> NarrowedOps;
  // Narrow operation tree rooted at store groups.
  for (auto *StoreGroup : StoreGroups) {
    VPValue *Res =
        narrowInterleaveGroupOp(StoreGroup->getStoredValues()[0], NarrowedOps);
    auto *SI =
        cast<StoreInst>(StoreGroup->getInterleaveGroup()->getInsertPos());
    auto *S = new VPWidenStoreRecipe(
        *SI, StoreGroup->getAddr(), Res, nullptr, /*Consecutive=*/true,
        /*Reverse=*/false, {}, StoreGroup->getDebugLoc());
    S->insertBefore(StoreGroup);
    StoreGroup->eraseFromParent();
  }

  // Adjust induction to reflect that the transformed plan only processes one
  // original iteration.
  auto *CanIV = VectorLoop->getCanonicalIV();
  auto *Inc = cast<VPInstruction>(CanIV->getBackedgeValue());
  VPBuilder PHBuilder(Plan.getVectorPreheader());

  VPValue *UF = Plan.getOrAddLiveIn(
      ConstantInt::get(VectorLoop->getCanonicalIVType(), 1 * Plan.getUF()));
  if (VF.isScalable()) {
    VPValue *VScale = PHBuilder.createElementCount(
        VectorLoop->getCanonicalIVType(), ElementCount::getScalable(1));
    VPValue *VScaleUF = PHBuilder.createOverflowingOp(
        Instruction::Mul, {VScale, UF}, {true, false});
    Inc->setOperand(1, VScaleUF);
    Plan.getVF().replaceAllUsesWith(VScale);
  } else {
    Inc->setOperand(1, UF);
    Plan.getVF().replaceAllUsesWith(
        Plan.getConstantInt(CanIV->getScalarType(), 1));
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
  MiddleTerm->setMetadata(LLVMContext::MD_prof, BranchWeights);
}

/// Compute and return the end value for \p WideIV, unless it is truncated. If
/// the induction recipe is not canonical, creates a VPDerivedIVRecipe to
/// compute the end value of the induction.
static VPValue *tryToComputeEndValueForInduction(VPWidenInductionRecipe *WideIV,
                                                 VPBuilder &VectorPHBuilder,
                                                 VPTypeAnalysis &TypeInfo,
                                                 VPValue *VectorTC) {
  auto *WideIntOrFp = dyn_cast<VPWidenIntOrFpInductionRecipe>(WideIV);
  // Truncated wide inductions resume from the last lane of their vector value
  // in the last vector iteration which is handled elsewhere.
  if (WideIntOrFp && WideIntOrFp->getTruncInst())
    return nullptr;

  VPIRValue *Start = WideIV->getStartValue();
  VPValue *Step = WideIV->getStepValue();
  const InductionDescriptor &ID = WideIV->getInductionDescriptor();
  VPValue *EndValue = VectorTC;
  if (!WideIntOrFp || !WideIntOrFp->isCanonical()) {
    EndValue = VectorPHBuilder.createDerivedIV(
        ID.getKind(), dyn_cast_or_null<FPMathOperator>(ID.getInductionBinOp()),
        Start, VectorTC, Step);
  }

  // EndValue is derived from the vector trip count (which has the same type as
  // the widest induction) and thus may be wider than the induction here.
  Type *ScalarTypeOfWideIV = TypeInfo.inferScalarType(WideIV);
  if (ScalarTypeOfWideIV != TypeInfo.inferScalarType(EndValue)) {
    EndValue = VectorPHBuilder.createScalarCast(Instruction::Trunc, EndValue,
                                                ScalarTypeOfWideIV,
                                                WideIV->getDebugLoc());
  }

  return EndValue;
}

void VPlanTransforms::updateScalarResumePhis(
    VPlan &Plan, DenseMap<VPValue *, VPValue *> &IVEndValues) {
  VPTypeAnalysis TypeInfo(Plan);
  auto *ScalarPH = Plan.getScalarPreheader();
  auto *MiddleVPBB = cast<VPBasicBlock>(ScalarPH->getPredecessors()[0]);
  VPRegionBlock *VectorRegion = Plan.getVectorLoopRegion();
  VPBuilder VectorPHBuilder(
      cast<VPBasicBlock>(VectorRegion->getSinglePredecessor()));
  VPBuilder MiddleBuilder(MiddleVPBB, MiddleVPBB->getFirstNonPhi());
  for (VPRecipeBase &PhiR : Plan.getScalarPreheader()->phis()) {
    auto *ResumePhiR = cast<VPPhi>(&PhiR);

    // TODO: Extract final value from induction recipe initially, optimize to
    // pre-computed end value together in optimizeInductionExitUsers.
    auto *VectorPhiR = cast<VPHeaderPHIRecipe>(ResumePhiR->getOperand(0));
    if (auto *WideIVR = dyn_cast<VPWidenInductionRecipe>(VectorPhiR)) {
      if (VPValue *EndValue = tryToComputeEndValueForInduction(
              WideIVR, VectorPHBuilder, TypeInfo, &Plan.getVectorTripCount())) {
        IVEndValues[WideIVR] = EndValue;
        ResumePhiR->setOperand(0, EndValue);
        ResumePhiR->setName("bc.resume.val");
        continue;
      }
      // TODO: Also handle truncated inductions here. Computing end-values
      // separately should be done as VPlan-to-VPlan optimization, after
      // legalizing all resume values to use the last lane from the loop.
      assert(cast<VPWidenIntOrFpInductionRecipe>(VectorPhiR)->getTruncInst() &&
             "should only skip truncated wide inductions");
      continue;
    }

    // The backedge value provides the value to resume coming out of a loop,
    // which for FORs is a vector whose last element needs to be extracted. The
    // start value provides the value if the loop is bypassed.
    bool IsFOR = isa<VPFirstOrderRecurrencePHIRecipe>(VectorPhiR);
    auto *ResumeFromVectorLoop = VectorPhiR->getBackedgeValue();
    assert(VectorRegion->getSingleSuccessor() == Plan.getMiddleBlock() &&
           "Cannot handle loops with uncountable early exits");
    if (IsFOR) {
      auto *ExtractPart = MiddleBuilder.createNaryOp(
          VPInstruction::ExtractLastPart, ResumeFromVectorLoop);
      ResumeFromVectorLoop = MiddleBuilder.createNaryOp(
          VPInstruction::ExtractLastLane, ExtractPart, DebugLoc::getUnknown(),
          "vector.recur.extract");
    }
    ResumePhiR->setName(IsFOR ? "scalar.recur.init" : "bc.merge.rdx");
    ResumePhiR->setOperand(0, ResumeFromVectorLoop);
  }
}

void VPlanTransforms::addExitUsersForFirstOrderRecurrences(VPlan &Plan,
                                                           VFRange &Range) {
  VPRegionBlock *VectorRegion = Plan.getVectorLoopRegion();
  auto *ScalarPHVPBB = Plan.getScalarPreheader();
  auto *MiddleVPBB = Plan.getMiddleBlock();
  VPBuilder ScalarPHBuilder(ScalarPHVPBB);
  VPBuilder MiddleBuilder(MiddleVPBB, MiddleVPBB->getFirstNonPhi());

  auto IsScalableOne = [](ElementCount VF) -> bool {
    return VF == ElementCount::getScalable(1);
  };

  for (auto &HeaderPhi : VectorRegion->getEntryBasicBlock()->phis()) {
    auto *FOR = dyn_cast<VPFirstOrderRecurrencePHIRecipe>(&HeaderPhi);
    if (!FOR)
      continue;

    assert(VectorRegion->getSingleSuccessor() == Plan.getMiddleBlock() &&
           "Cannot handle loops with uncountable early exits");

    // This is the second phase of vectorizing first-order recurrences, creating
    // extract for users outside the loop. An overview of the transformation is
    // described below. Suppose we have the following loop with some use after
    // the loop of the last a[i-1],
    //
    //   for (int i = 0; i < n; ++i) {
    //     t = a[i - 1];
    //     b[i] = a[i] - t;
    //   }
    //   use t;
    //
    // There is a first-order recurrence on "a". For this loop, the shorthand
    // scalar IR looks like:
    //
    //   scalar.ph:
    //     s.init = a[-1]
    //     br scalar.body
    //
    //   scalar.body:
    //     i = phi [0, scalar.ph], [i+1, scalar.body]
    //     s1 = phi [s.init, scalar.ph], [s2, scalar.body]
    //     s2 = a[i]
    //     b[i] = s2 - s1
    //     br cond, scalar.body, exit.block
    //
    //   exit.block:
    //     use = lcssa.phi [s1, scalar.body]
    //
    // In this example, s1 is a recurrence because it's value depends on the
    // previous iteration. In the first phase of vectorization, we created a
    // VPFirstOrderRecurrencePHIRecipe v1 for s1. Now we create the extracts
    // for users in the scalar preheader and exit block.
    //
    //   vector.ph:
    //     v_init = vector(..., ..., ..., a[-1])
    //     br vector.body
    //
    //   vector.body
    //     i = phi [0, vector.ph], [i+4, vector.body]
    //     v1 = phi [v_init, vector.ph], [v2, vector.body]
    //     v2 = a[i, i+1, i+2, i+3]
    //     b[i] = v2 - v1
    //     // Next, third phase will introduce v1' = splice(v1(3), v2(0, 1, 2))
    //     b[i, i+1, i+2, i+3] = v2 - v1
    //     br cond, vector.body, middle.block
    //
    //   middle.block:
    //     vector.recur.extract.for.phi = v2(2)
    //     vector.recur.extract = v2(3)
    //     br cond, scalar.ph, exit.block
    //
    //   scalar.ph:
    //     scalar.recur.init = phi [vector.recur.extract, middle.block],
    //                             [s.init, otherwise]
    //     br scalar.body
    //
    //   scalar.body:
    //     i = phi [0, scalar.ph], [i+1, scalar.body]
    //     s1 = phi [scalar.recur.init, scalar.ph], [s2, scalar.body]
    //     s2 = a[i]
    //     b[i] = s2 - s1
    //     br cond, scalar.body, exit.block
    //
    //   exit.block:
    //     lo = lcssa.phi [s1, scalar.body],
    //                    [vector.recur.extract.for.phi, middle.block]
    //
    // Now update VPIRInstructions modeling LCSSA phis in the exit block.
    // Extract the penultimate value of the recurrence and use it as operand for
    // the VPIRInstruction modeling the phi.
    for (VPRecipeBase &R : make_early_inc_range(
             make_range(MiddleVPBB->getFirstNonPhi(), MiddleVPBB->end()))) {
      if (!match(&R, m_ExtractLastLaneOfLastPart(m_Specific(FOR))))
        continue;

      // For VF vscale x 1, if vscale = 1, we are unable to extract the
      // penultimate value of the recurrence. Instead we rely on the existing
      // extract of the last element from the result of
      // VPInstruction::FirstOrderRecurrenceSplice.
      // TODO: Consider vscale_range info and UF.
      if (LoopVectorizationPlanner::getDecisionAndClampRange(IsScalableOne,
                                                             Range))
        return;
      VPValue *PenultimateElement = MiddleBuilder.createNaryOp(
          VPInstruction::ExtractPenultimateElement, FOR->getBackedgeValue(), {},
          "vector.recur.extract.for.phi");
      cast<VPInstruction>(&R)->replaceAllUsesWith(PenultimateElement);
    }
  }
}
