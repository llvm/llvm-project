//===- VPlanAnalysis.cpp - Various Analyses working on VPlan ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VPlanAnalysis.h"
#include "VPlan.h"
#include "VPlanCFG.h"
#include "VPlanDominatorTree.h"
#include "VPlanHelpers.h"
#include "VPlanPatternMatch.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Analysis/TargetTransformInfo.h"

using namespace llvm;
using namespace VPlanPatternMatch;

#define DEBUG_TYPE "vplan"

void llvm::collectEphemeralRecipesForVPlan(
    VPlan &Plan, DenseSet<VPRecipeBase *> &EphRecipes) {
  // First, collect seed recipes which are operands of assumes.
  SmallVector<VPRecipeBase *> Worklist;
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(
           vp_depth_first_deep(Plan.getVectorLoopRegion()->getEntry()))) {
    for (VPRecipeBase &R : *VPBB) {
      auto *RepR = dyn_cast<VPReplicateRecipe>(&R);
      if (!RepR || !match(RepR, m_Intrinsic<Intrinsic::assume>()))
        continue;
      Worklist.push_back(RepR);
      EphRecipes.insert(RepR);
    }
  }

  // Process operands of candidates in worklist and add them to the set of
  // ephemeral recipes, if they don't have side-effects and are only used by
  // other ephemeral recipes.
  while (!Worklist.empty()) {
    VPRecipeBase *Cur = Worklist.pop_back_val();
    for (VPValue *Op : Cur->operands()) {
      auto *OpR = Op->getDefiningRecipe();
      if (!OpR || OpR->mayHaveSideEffects() || EphRecipes.contains(OpR))
        continue;
      if (any_of(Op->users(), [EphRecipes](VPUser *U) {
            auto *UR = dyn_cast<VPRecipeBase>(U);
            return !UR || !EphRecipes.contains(UR);
          }))
        continue;
      EphRecipes.insert(OpR);
      Worklist.push_back(OpR);
    }
  }
}

template void DomTreeBuilder::Calculate<DominatorTreeBase<VPBlockBase, false>>(
    DominatorTreeBase<VPBlockBase, false> &DT);

bool VPDominatorTree::properlyDominates(const VPRecipeBase *A,
                                        const VPRecipeBase *B) {
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

  return Base::properlyDominates(ParentA, ParentB);
}

InstructionCost
VPRegisterUsage::spillCost(const TargetTransformInfo &TTI,
                           TargetTransformInfo::TargetCostKind CostKind,
                           unsigned OverrideMaxNumRegs) const {
  InstructionCost Cost;
  for (const auto &[RegClass, MaxUsers] : MaxLocalUsers) {
    unsigned AvailableRegs = OverrideMaxNumRegs > 0
                                 ? OverrideMaxNumRegs
                                 : TTI.getNumberOfRegisters(RegClass);
    if (MaxUsers > AvailableRegs) {
      // Assume that for each register used past what's available we get one
      // spill and reload.
      unsigned Spills = MaxUsers - AvailableRegs;
      InstructionCost SpillCost =
          TTI.getRegisterClassSpillCost(RegClass, CostKind) +
          TTI.getRegisterClassReloadCost(RegClass, CostKind);
      InstructionCost TotalCost = Spills * SpillCost;
      LLVM_DEBUG(dbgs() << "LV(REG): Cost of " << TotalCost << " from "
                        << Spills << " spills of "
                        << TTI.getRegisterClassName(RegClass) << "\n");
      Cost += TotalCost;
    }
  }
  return Cost;
}

SmallVector<VPRegisterUsage, 8> llvm::calculateRegisterUsageForPlan(
    VPlan &Plan, ArrayRef<ElementCount> VFs, const TargetTransformInfo &TTI,
    const SmallPtrSetImpl<const Value *> &ValuesToIgnore) {
  // Each 'key' in the map opens a new interval. The values
  // of the map are the index of the 'last seen' usage of the
  // VPValue that is the key.
  using IntervalMap = SmallDenseMap<VPValue *, unsigned, 16>;

  // Maps indices to recipes.
  SmallVector<VPRecipeBase *, 64> Idx2Recipe;
  // Marks the end of each interval.
  IntervalMap EndPoint;
  // Saves the list of VPValues that are used in the loop.
  SmallPtrSet<VPValue *, 8> Ends;
  // Saves the list of values that are used in the loop but are defined outside
  // the loop (not including non-recipe values such as arguments and
  // constants).
  SmallSetVector<VPValue *, 8> LoopInvariants;
  if (Plan.getVectorTripCount().getNumUsers() > 0)
    LoopInvariants.insert(&Plan.getVectorTripCount());

  // We scan the loop in a topological order in order and assign a number to
  // each recipe. We use RPO to ensure that defs are met before their users. We
  // assume that each recipe that has in-loop users starts an interval. We
  // record every time that an in-loop value is used, so we have a list of the
  // first occurences of each recipe and last occurrence of each VPValue.
  VPRegionBlock *LoopRegion = Plan.getVectorLoopRegion();
  ReversePostOrderTraversal<VPBlockDeepTraversalWrapper<VPBlockBase *>> RPOT(
      LoopRegion);
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(RPOT)) {
    if (!VPBB->getParent())
      break;
    for (VPRecipeBase &R : *VPBB) {
      Idx2Recipe.push_back(&R);

      // Save the end location of each USE.
      for (VPValue *U : R.operands()) {
        if (isa<VPRecipeValue>(U)) {
          // Overwrite previous end points.
          EndPoint[U] = Idx2Recipe.size();
          Ends.insert(U);
        } else if (auto *IRV = dyn_cast<VPIRValue>(U)) {
          // Ignore non-recipe values such as arguments, constants, etc.
          // FIXME: Might need some motivation why these values are ignored. If
          // for example an argument is used inside the loop it will increase
          // the register pressure (so shouldn't we add it to LoopInvariants).
          if (!isa<Instruction>(IRV->getValue()))
            continue;
          // This recipe is outside the loop, record it and continue.
          LoopInvariants.insert(U);
        }
        // Other types of VPValue are currently not tracked.
      }
    }
    if (VPBB == LoopRegion->getExiting()) {
      // VPWidenIntOrFpInductionRecipes are used implicitly at the end of the
      // exiting block, where their increment will get materialized eventually.
      for (auto &R : LoopRegion->getEntryBasicBlock()->phis()) {
        if (auto *WideIV = dyn_cast<VPWidenIntOrFpInductionRecipe>(&R)) {
          EndPoint[WideIV] = Idx2Recipe.size();
          Ends.insert(WideIV);
        }
      }
    }
  }

  // Saves the list of intervals that end with the index in 'key'.
  using VPValueList = SmallVector<VPValue *, 2>;
  SmallDenseMap<unsigned, VPValueList, 16> TransposeEnds;

  // Next, we transpose the EndPoints into a multi map that holds the list of
  // intervals that *end* at a specific location.
  for (auto &Interval : EndPoint)
    TransposeEnds[Interval.second].push_back(Interval.first);

  SmallPtrSet<VPValue *, 8> OpenIntervals;
  SmallVector<VPRegisterUsage, 8> RUs(VFs.size());
  SmallVector<SmallMapVector<unsigned, unsigned, 4>, 8> MaxUsages(VFs.size());

  LLVM_DEBUG(dbgs() << "LV(REG): Calculating max register usage:\n");

  const auto &TTICapture = TTI;
  auto GetRegUsage = [&TTICapture](Type *Ty, ElementCount VF) -> unsigned {
    if (Ty->isTokenTy() || !VectorType::isValidElementType(Ty) ||
        (VF.isScalable() &&
         !TTICapture.isElementTypeLegalForScalableVector(Ty)))
      return 0;
    return TTICapture.getRegUsageForType(VectorType::get(Ty, VF));
  };

  VPValue *CanIV = LoopRegion->getCanonicalIV();
  // Note: canonical IVs are retained even if they have no users.
  if (CanIV->getNumUsers() != 0)
    OpenIntervals.insert(CanIV);

  // We scan the instructions linearly and record each time that a new interval
  // starts, by placing it in a set. If we find this value in TransposEnds then
  // we remove it from the set. The max register usage is the maximum register
  // usage of the recipes of the set.
  for (unsigned int Idx = 0, Sz = Idx2Recipe.size(); Idx < Sz; ++Idx) {
    VPRecipeBase *R = Idx2Recipe[Idx];

    // Remove all of the VPValues that end at this location.
    VPValueList &List = TransposeEnds[Idx];
    for (VPValue *ToRemove : List)
      OpenIntervals.erase(ToRemove);

    // Ignore recipes that are never used within the loop and do not have side
    // effects.
    if (none_of(R->definedValues(),
                [&Ends](VPValue *Def) { return Ends.count(Def); }) &&
        !R->mayHaveSideEffects())
      continue;

    // Skip recipes for ignored values.
    // TODO: Should mark recipes for ephemeral values that cannot be removed
    // explictly in VPlan.
    if (isa<VPSingleDefRecipe>(R) &&
        ValuesToIgnore.contains(
            cast<VPSingleDefRecipe>(R)->getUnderlyingValue()))
      continue;

    // For each VF find the maximum usage of registers.
    for (unsigned J = 0, E = VFs.size(); J < E; ++J) {
      // Count the number of registers used, per register class, given all open
      // intervals.
      // Note that elements in this SmallMapVector will be default constructed
      // as 0. So we can use "RegUsage[ClassID] += n" in the code below even if
      // there is no previous entry for ClassID.
      SmallMapVector<unsigned, unsigned, 4> RegUsage;

      for (auto *VPV : OpenIntervals) {
        // Skip artificial values or values that weren't present in the original
        // loop.
        // TODO: Remove skipping values that weren't present in the original
        // loop after removing the legacy
        // LoopVectorizationCostModel::calculateRegisterUsage
        if (isa<VPVectorPointerRecipe, VPVectorEndPointerRecipe,
                VPBranchOnMaskRecipe>(VPV) ||
            match(VPV, m_ExtractLastPart(m_VPValue())))
          continue;

        if (VFs[J].isScalar() ||
            isa<VPRegionValue, VPReplicateRecipe, VPDerivedIVRecipe,
                VPCurrentIterationPHIRecipe, VPScalarIVStepsRecipe>(VPV) ||
            (isa<VPInstruction>(VPV) && vputils::onlyScalarValuesUsed(VPV)) ||
            (isa<VPReductionPHIRecipe>(VPV) &&
             (cast<VPReductionPHIRecipe>(VPV))->isInLoop())) {
          unsigned ClassID =
              TTI.getRegisterClassForType(false, VPV->getScalarType());
          // FIXME: The target might use more than one register for the type
          // even in the scalar case.
          RegUsage[ClassID] += 1;
        } else {
          // The output from scaled phis and scaled reductions actually has
          // fewer lanes than the VF.
          unsigned ScaleFactor =
              vputils::getVFScaleFactor(VPV->getDefiningRecipe());
          ElementCount VF = VFs[J];
          if (ScaleFactor > 1) {
            VF = VFs[J].divideCoefficientBy(ScaleFactor);
            LLVM_DEBUG(dbgs() << "LV(REG): Scaled down VF from " << VFs[J]
                              << " to " << VF << " for " << *R << "\n";);
          }

          Type *ScalarTy = VPV->getScalarType();
          unsigned ClassID = TTI.getRegisterClassForType(true, ScalarTy);
          RegUsage[ClassID] += GetRegUsage(ScalarTy, VF);
        }
      }

      for (const auto &Pair : RegUsage) {
        auto &Entry = MaxUsages[J][Pair.first];
        Entry = std::max(Entry, Pair.second);
      }
    }

    LLVM_DEBUG(dbgs() << "LV(REG): At #" << Idx << " Interval # "
                      << OpenIntervals.size() << '\n');

    // Add used VPValues defined by the current recipe to the list of open
    // intervals.
    for (VPValue *DefV : R->definedValues())
      if (Ends.contains(DefV))
        OpenIntervals.insert(DefV);
  }

  // We also search for instructions that are defined outside the loop, but are
  // used inside the loop. We need this number separately from the max-interval
  // usage number because when we unroll, loop-invariant values do not take
  // more register.
  VPRegisterUsage RU;
  for (unsigned Idx = 0, End = VFs.size(); Idx < End; ++Idx) {
    // Note that elements in this SmallMapVector will be default constructed
    // as 0. So we can use "Invariant[ClassID] += n" in the code below even if
    // there is no previous entry for ClassID.
    SmallMapVector<unsigned, unsigned, 4> Invariant;

    for (auto *In : LoopInvariants) {
      // FIXME: The target might use more than one register for the type
      // even in the scalar case.
      bool IsScalar = vputils::onlyScalarValuesUsed(In);

      ElementCount VF = IsScalar ? ElementCount::getFixed(1) : VFs[Idx];
      unsigned ClassID =
          TTI.getRegisterClassForType(VF.isVector(), In->getScalarType());
      Invariant[ClassID] += GetRegUsage(In->getScalarType(), VF);
    }

    LLVM_DEBUG({
      dbgs() << "LV(REG): VF = " << VFs[Idx] << '\n';
      dbgs() << "LV(REG): Found max usage: " << MaxUsages[Idx].size()
             << " item\n";
      for (const auto &pair : MaxUsages[Idx]) {
        dbgs() << "LV(REG): RegisterClass: "
               << TTI.getRegisterClassName(pair.first) << ", " << pair.second
               << " registers\n";
      }
      dbgs() << "LV(REG): Found invariant usage: " << Invariant.size()
             << " item\n";
      for (const auto &pair : Invariant) {
        dbgs() << "LV(REG): RegisterClass: "
               << TTI.getRegisterClassName(pair.first) << ", " << pair.second
               << " registers\n";
      }
    });

    RU.LoopInvariantRegs = Invariant;
    RU.MaxLocalUsers = MaxUsages[Idx];
    RUs[Idx] = RU;
  }

  return RUs;
}
