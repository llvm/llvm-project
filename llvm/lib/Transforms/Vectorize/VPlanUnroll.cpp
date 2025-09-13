//===-- VPlanUnroll.cpp - VPlan unroller ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements explicit unrolling for VPlans.
///
//===----------------------------------------------------------------------===//

#include "VPRecipeBuilder.h"
#include "VPlan.h"
#include "VPlanAnalysis.h"
#include "VPlanCFG.h"
#include "VPlanHelpers.h"
#include "VPlanPatternMatch.h"
#include "VPlanTransforms.h"
#include "VPlanUtils.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/IR/Intrinsics.h"

using namespace llvm;
using namespace llvm::VPlanPatternMatch;

namespace {

/// Helper to hold state needed for unrolling. It holds the Plan to unroll by
/// UF. It also holds copies of VPValues across UF-1 unroll parts to facilitate
/// the unrolling transformation, where the original VPValues are retained for
/// part zero.
class UnrollState {
  /// Plan to unroll.
  VPlan &Plan;
  /// Unroll factor to unroll by.
  const unsigned UF;
  /// Analysis for types.
  VPTypeAnalysis TypeInfo;

  /// Unrolling may create recipes that should not be unrolled themselves.
  /// Those are tracked in ToSkip.
  SmallPtrSet<VPRecipeBase *, 8> ToSkip;

  // Associate with each VPValue of part 0 its unrolled instances of parts 1,
  // ..., UF-1.
  DenseMap<VPValue *, SmallVector<VPValue *>> VPV2Parts;

  /// Unroll replicate region \p VPR by cloning the region UF - 1 times.
  void unrollReplicateRegionByUF(VPRegionBlock *VPR);

  /// Unroll recipe \p R by cloning it UF - 1 times, unless it is uniform across
  /// all parts.
  void unrollRecipeByUF(VPRecipeBase &R);

  /// Unroll header phi recipe \p R. How exactly the recipe gets unrolled
  /// depends on the concrete header phi. Inserts newly created recipes at \p
  /// InsertPtForPhi.
  void unrollHeaderPHIByUF(VPHeaderPHIRecipe *R,
                           VPBasicBlock::iterator InsertPtForPhi);

  /// Unroll a widen induction recipe \p IV. This introduces recipes to compute
  /// the induction steps for each part.
  void unrollWidenInductionByUF(VPWidenInductionRecipe *IV,
                                VPBasicBlock::iterator InsertPtForPhi);

  VPValue *getConstantVPV(unsigned Part) {
    Type *CanIVIntTy = Plan.getCanonicalIV()->getScalarType();
    return Plan.getOrAddLiveIn(ConstantInt::get(CanIVIntTy, Part));
  }

public:
  UnrollState(VPlan &Plan, unsigned UF) : Plan(Plan), UF(UF), TypeInfo(Plan) {}

  void unrollBlock(VPBlockBase *VPB);

  VPValue *getValueForPart(VPValue *V, unsigned Part) {
    if (Part == 0 || V->isLiveIn())
      return V;
    assert((VPV2Parts.contains(V) && VPV2Parts[V].size() >= Part) &&
           "accessed value does not exist");
    return VPV2Parts[V][Part - 1];
  }

  /// Given a single original recipe \p OrigR (of part zero), and its copy \p
  /// CopyR for part \p Part, map every VPValue defined by \p OrigR to its
  /// corresponding VPValue defined by \p CopyR.
  void addRecipeForPart(VPRecipeBase *OrigR, VPRecipeBase *CopyR,
                        unsigned Part) {
    for (const auto &[Idx, VPV] : enumerate(OrigR->definedValues())) {
      const auto &[V, _] = VPV2Parts.try_emplace(VPV);
      assert(V->second.size() == Part - 1 && "earlier parts not set");
      V->second.push_back(CopyR->getVPValue(Idx));
    }
  }

  /// Given a uniform recipe \p R, add it for all parts.
  void addUniformForAllParts(VPSingleDefRecipe *R) {
    const auto &[V, Inserted] = VPV2Parts.try_emplace(R);
    assert(Inserted && "uniform value already added");
    for (unsigned Part = 0; Part != UF; ++Part)
      V->second.push_back(R);
  }

  bool contains(VPValue *VPV) const { return VPV2Parts.contains(VPV); }

  /// Update \p R's operand at \p OpIdx with its corresponding VPValue for part
  /// \p P.
  void remapOperand(VPRecipeBase *R, unsigned OpIdx, unsigned Part) {
    auto *Op = R->getOperand(OpIdx);
    R->setOperand(OpIdx, getValueForPart(Op, Part));
  }

  /// Update \p R's operands with their corresponding VPValues for part \p P.
  void remapOperands(VPRecipeBase *R, unsigned Part) {
    for (const auto &[OpIdx, Op] : enumerate(R->operands()))
      R->setOperand(OpIdx, getValueForPart(Op, Part));
  }
};
} // namespace

void UnrollState::unrollReplicateRegionByUF(VPRegionBlock *VPR) {
  VPBlockBase *InsertPt = VPR->getSingleSuccessor();
  for (unsigned Part = 1; Part != UF; ++Part) {
    auto *Copy = VPR->clone();
    VPBlockUtils::insertBlockBefore(Copy, InsertPt);

    auto PartI = vp_depth_first_shallow(Copy->getEntry());
    auto Part0 = vp_depth_first_shallow(VPR->getEntry());
    for (const auto &[PartIVPBB, Part0VPBB] :
         zip(VPBlockUtils::blocksOnly<VPBasicBlock>(PartI),
             VPBlockUtils::blocksOnly<VPBasicBlock>(Part0))) {
      for (const auto &[PartIR, Part0R] : zip(*PartIVPBB, *Part0VPBB)) {
        remapOperands(&PartIR, Part);
        if (auto *ScalarIVSteps = dyn_cast<VPScalarIVStepsRecipe>(&PartIR)) {
          ScalarIVSteps->addOperand(getConstantVPV(Part));
        }

        addRecipeForPart(&Part0R, &PartIR, Part);
      }
    }
  }
}

void UnrollState::unrollWidenInductionByUF(
    VPWidenInductionRecipe *IV, VPBasicBlock::iterator InsertPtForPhi) {
  VPBasicBlock *PH = cast<VPBasicBlock>(
      IV->getParent()->getEnclosingLoopRegion()->getSinglePredecessor());
  Type *IVTy = TypeInfo.inferScalarType(IV);
  auto &ID = IV->getInductionDescriptor();
  VPIRFlags Flags;
  if (isa_and_present<FPMathOperator>(ID.getInductionBinOp()))
    Flags = ID.getInductionBinOp()->getFastMathFlags();

  VPValue *ScalarStep = IV->getStepValue();
  VPBuilder Builder(PH);
  Type *VectorStepTy =
      IVTy->isPointerTy() ? TypeInfo.inferScalarType(ScalarStep) : IVTy;
  VPInstruction *VectorStep = Builder.createNaryOp(
      VPInstruction::WideIVStep, {&Plan.getVF(), ScalarStep}, VectorStepTy,
      Flags, IV->getDebugLoc());

  ToSkip.insert(VectorStep);

  // Now create recipes to compute the induction steps for part 1 .. UF. Part 0
  // remains the header phi. Parts > 0 are computed by adding Step to the
  // previous part. The header phi recipe will get 2 new operands: the step
  // value for a single part and the last part, used to compute the backedge
  // value during VPWidenInductionRecipe::execute.
  // %Part.0 = VPWidenInductionRecipe %Start, %ScalarStep, %VectorStep, %Part.3
  // %Part.1 = %Part.0 + %VectorStep
  // %Part.2 = %Part.1 + %VectorStep
  // %Part.3 = %Part.2 + %VectorStep
  //
  // The newly added recipes are added to ToSkip to avoid interleaving them
  // again.
  VPValue *Prev = IV;
  Builder.setInsertPoint(IV->getParent(), InsertPtForPhi);
  unsigned AddOpc;
  if (IVTy->isPointerTy())
    AddOpc = VPInstruction::WidePtrAdd;
  else if (IVTy->isFloatingPointTy())
    AddOpc = ID.getInductionOpcode();
  else
    AddOpc = Instruction::Add;
  for (unsigned Part = 1; Part != UF; ++Part) {
    std::string Name =
        Part > 1 ? "step.add." + std::to_string(Part) : "step.add";

    VPInstruction *Add = Builder.createNaryOp(AddOpc,
                                              {
                                                  Prev,
                                                  VectorStep,
                                              },
                                              Flags, IV->getDebugLoc(), Name);
    ToSkip.insert(Add);
    addRecipeForPart(IV, Add, Part);
    Prev = Add;
  }
  IV->addOperand(VectorStep);
  IV->addOperand(Prev);
}

void UnrollState::unrollHeaderPHIByUF(VPHeaderPHIRecipe *R,
                                      VPBasicBlock::iterator InsertPtForPhi) {
  // First-order recurrences pass a single vector or scalar through their header
  // phis, irrespective of interleaving.
  if (isa<VPFirstOrderRecurrencePHIRecipe>(R))
    return;

  // Generate step vectors for each unrolled part.
  if (auto *IV = dyn_cast<VPWidenInductionRecipe>(R)) {
    unrollWidenInductionByUF(IV, InsertPtForPhi);
    return;
  }

  auto *RdxPhi = dyn_cast<VPReductionPHIRecipe>(R);
  if (RdxPhi && RdxPhi->isOrdered())
    return;

  auto InsertPt = std::next(R->getIterator());
  for (unsigned Part = 1; Part != UF; ++Part) {
    VPRecipeBase *Copy = R->clone();
    Copy->insertBefore(*R->getParent(), InsertPt);
    addRecipeForPart(R, Copy, Part);
    if (RdxPhi) {
      // If the start value is a ReductionStartVector, use the identity value
      // (second operand) for unrolled parts. If the scaling factor is > 1,
      // create a new ReductionStartVector with the scale factor and both
      // operands set to the identity value.
      if (auto *VPI = dyn_cast<VPInstruction>(RdxPhi->getStartValue())) {
        assert(VPI->getOpcode() == VPInstruction::ReductionStartVector &&
               "unexpected start VPInstruction");
        if (Part != 1)
          continue;
        VPValue *StartV;
        if (match(VPI->getOperand(2), m_SpecificInt(1))) {
          StartV = VPI->getOperand(1);
        } else {
          auto *C = VPI->clone();
          C->setOperand(0, C->getOperand(1));
          C->insertAfter(VPI);
          StartV = C;
        }
        for (unsigned Part = 1; Part != UF; ++Part)
          VPV2Parts[VPI][Part - 1] = StartV;
      }
      Copy->addOperand(getConstantVPV(Part));
    } else {
      assert(isa<VPActiveLaneMaskPHIRecipe>(R) &&
             "unexpected header phi recipe not needing unrolled part");
    }
  }
}

/// Handle non-header-phi recipes.
void UnrollState::unrollRecipeByUF(VPRecipeBase &R) {
  if (match(&R, m_BranchOnCond(m_VPValue())) ||
      match(&R, m_BranchOnCount(m_VPValue(), m_VPValue())))
    return;

  if (auto *VPI = dyn_cast<VPInstruction>(&R)) {
    if (vputils::onlyFirstPartUsed(VPI)) {
      addUniformForAllParts(VPI);
      return;
    }
  }
  if (auto *RepR = dyn_cast<VPReplicateRecipe>(&R)) {
    if (isa<StoreInst>(RepR->getUnderlyingValue()) &&
        RepR->getOperand(1)->isDefinedOutsideLoopRegions()) {
      // Stores to an invariant address only need to store the last part.
      remapOperands(&R, UF - 1);
      return;
    }
    if (auto *II = dyn_cast<IntrinsicInst>(RepR->getUnderlyingValue())) {
      if (II->getIntrinsicID() == Intrinsic::experimental_noalias_scope_decl) {
        addUniformForAllParts(RepR);
        return;
      }
    }
  }

  // Unroll non-uniform recipes.
  auto InsertPt = std::next(R.getIterator());
  VPBasicBlock &VPBB = *R.getParent();
  for (unsigned Part = 1; Part != UF; ++Part) {
    VPRecipeBase *Copy = R.clone();
    Copy->insertBefore(VPBB, InsertPt);
    addRecipeForPart(&R, Copy, Part);

    VPValue *Op;
    if (match(&R, m_VPInstruction<VPInstruction::FirstOrderRecurrenceSplice>(
                      m_VPValue(), m_VPValue(Op)))) {
      Copy->setOperand(0, getValueForPart(Op, Part - 1));
      Copy->setOperand(1, getValueForPart(Op, Part));
      continue;
    }
    if (auto *Red = dyn_cast<VPReductionRecipe>(&R)) {
      auto *Phi = dyn_cast<VPReductionPHIRecipe>(R.getOperand(0));
      if (Phi && Phi->isOrdered()) {
        auto &Parts = VPV2Parts[Phi];
        if (Part == 1) {
          Parts.clear();
          Parts.push_back(Red);
        }
        Parts.push_back(Copy->getVPSingleValue());
        Phi->setOperand(1, Copy->getVPSingleValue());
      }
    }
    remapOperands(Copy, Part);

    // Add operand indicating the part to generate code for, to recipes still
    // requiring it.
    if (isa<VPScalarIVStepsRecipe, VPWidenCanonicalIVRecipe,
            VPVectorPointerRecipe, VPVectorEndPointerRecipe>(Copy) ||
        match(Copy, m_VPInstruction<VPInstruction::CanonicalIVIncrementForPart>(
                        m_VPValue())))
      Copy->addOperand(getConstantVPV(Part));

    if (isa<VPVectorPointerRecipe, VPVectorEndPointerRecipe>(R))
      Copy->setOperand(0, R.getOperand(0));
  }
}

void UnrollState::unrollBlock(VPBlockBase *VPB) {
  auto *VPR = dyn_cast<VPRegionBlock>(VPB);
  if (VPR) {
    if (VPR->isReplicator())
      return unrollReplicateRegionByUF(VPR);

    // Traverse blocks in region in RPO to ensure defs are visited before uses
    // across blocks.
    ReversePostOrderTraversal<VPBlockShallowTraversalWrapper<VPBlockBase *>>
        RPOT(VPR->getEntry());
    for (VPBlockBase *VPB : RPOT)
      unrollBlock(VPB);
    return;
  }

  // VPB is a VPBasicBlock; unroll it, i.e., unroll its recipes.
  auto *VPBB = cast<VPBasicBlock>(VPB);
  auto InsertPtForPhi = VPBB->getFirstNonPhi();
  for (VPRecipeBase &R : make_early_inc_range(*VPBB)) {
    if (ToSkip.contains(&R) || isa<VPIRInstruction>(&R))
      continue;

    // Add all VPValues for all parts to AnyOf, FirstActiveLaneMask and
    // Compute*Result which combine all parts to compute the final value.
    VPValue *Op1;
    if (match(&R, m_VPInstruction<VPInstruction::AnyOf>(m_VPValue(Op1))) ||
        match(&R, m_VPInstruction<VPInstruction::FirstActiveLane>(
                      m_VPValue(Op1))) ||
        match(&R, m_VPInstruction<VPInstruction::ComputeAnyOfResult>(
                      m_VPValue(), m_VPValue(), m_VPValue(Op1))) ||
        match(&R, m_VPInstruction<VPInstruction::ComputeReductionResult>(
                      m_VPValue(), m_VPValue(Op1))) ||
        match(&R, m_VPInstruction<VPInstruction::ComputeFindIVResult>(
                      m_VPValue(), m_VPValue(), m_VPValue(), m_VPValue(Op1)))) {
      addUniformForAllParts(cast<VPInstruction>(&R));
      for (unsigned Part = 1; Part != UF; ++Part)
        R.addOperand(getValueForPart(Op1, Part));
      continue;
    }
    VPValue *Op0;
    if (match(&R, m_VPInstruction<VPInstruction::ExtractLane>(
                      m_VPValue(Op0), m_VPValue(Op1)))) {
      addUniformForAllParts(cast<VPInstruction>(&R));
      for (unsigned Part = 1; Part != UF; ++Part)
        R.addOperand(getValueForPart(Op1, Part));
      continue;
    }
    if (match(&R, m_ExtractLastElement(m_VPValue(Op0))) ||
        match(&R, m_VPInstruction<VPInstruction::ExtractPenultimateElement>(
                      m_VPValue(Op0)))) {
      addUniformForAllParts(cast<VPSingleDefRecipe>(&R));
      if (Plan.hasScalarVFOnly()) {
        auto *I = cast<VPInstruction>(&R);
        // Extracting from end with VF = 1 implies retrieving the last or
        // penultimate scalar part (UF-1 or UF-2).
        unsigned Offset =
            I->getOpcode() == VPInstruction::ExtractLastElement ? 1 : 2;
        I->replaceAllUsesWith(getValueForPart(Op0, UF - Offset));
        R.eraseFromParent();
      } else {
        // Otherwise we extract from the last part.
        remapOperands(&R, UF - 1);
      }
      continue;
    }

    auto *SingleDef = dyn_cast<VPSingleDefRecipe>(&R);
    if (SingleDef && vputils::isUniformAcrossVFsAndUFs(SingleDef)) {
      addUniformForAllParts(SingleDef);
      continue;
    }

    if (auto *H = dyn_cast<VPHeaderPHIRecipe>(&R)) {
      unrollHeaderPHIByUF(H, InsertPtForPhi);
      continue;
    }

    unrollRecipeByUF(R);
  }
}

void VPlanTransforms::unrollByUF(VPlan &Plan, unsigned UF) {
  assert(UF > 0 && "Unroll factor must be positive");
  Plan.setUF(UF);
  auto Cleanup = make_scope_exit([&Plan]() {
    auto Iter = vp_depth_first_deep(Plan.getEntry());
    // Remove recipes that are redundant after unrolling.
    for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(Iter)) {
      for (VPRecipeBase &R : make_early_inc_range(*VPBB)) {
        auto *VPI = dyn_cast<VPInstruction>(&R);
        if (VPI &&
            VPI->getOpcode() == VPInstruction::CanonicalIVIncrementForPart &&
            VPI->getNumOperands() == 1) {
          VPI->replaceAllUsesWith(VPI->getOperand(0));
          VPI->eraseFromParent();
        }
      }
    }
  });
  if (UF == 1) {
    return;
  }

  UnrollState Unroller(Plan, UF);

  // Iterate over all blocks in the plan starting from Entry, and unroll
  // recipes inside them. This includes the vector preheader and middle blocks,
  // which may set up or post-process per-part values.
  ReversePostOrderTraversal<VPBlockShallowTraversalWrapper<VPBlockBase *>> RPOT(
      Plan.getEntry());
  for (VPBlockBase *VPB : RPOT)
    Unroller.unrollBlock(VPB);

  unsigned Part = 1;
  // Remap operands of cloned header phis to update backedge values. The header
  // phis cloned during unrolling are just after the header phi for part 0.
  // Reset Part to 1 when reaching the first (part 0) recipe of a block.
  for (VPRecipeBase &H :
       Plan.getVectorLoopRegion()->getEntryBasicBlock()->phis()) {
    // The second operand of Fixed Order Recurrence phi's, feeding the spliced
    // value across the backedge, needs to remap to the last part of the spliced
    // value.
    if (isa<VPFirstOrderRecurrencePHIRecipe>(&H)) {
      Unroller.remapOperand(&H, 1, UF - 1);
      continue;
    }
    if (Unroller.contains(H.getVPSingleValue())) {
      Part = 1;
      continue;
    }
    Unroller.remapOperands(&H, Part);
    Part++;
  }

  VPlanTransforms::removeDeadRecipes(Plan);
}

/// Create a single-scalar clone of \p DefR (must be a VPReplicateRecipe or
/// VPInstruction) for lane \p Lane. Use \p Def2LaneDefs to look up scalar
/// definitions for operands of \DefR.
static VPRecipeWithIRFlags *
cloneForLane(VPlan &Plan, VPBuilder &Builder, Type *IdxTy,
             VPRecipeWithIRFlags *DefR, VPLane Lane,
             const DenseMap<VPValue *, SmallVector<VPValue *>> &Def2LaneDefs) {
  // Collect the operands at Lane, creating extracts as needed.
  SmallVector<VPValue *> NewOps;
  for (VPValue *Op : DefR->operands()) {
    // If Op is a definition that has been unrolled, directly use the clone for
    // the corresponding lane.
    auto LaneDefs = Def2LaneDefs.find(Op);
    if (LaneDefs != Def2LaneDefs.end()) {
      NewOps.push_back(LaneDefs->second[Lane.getKnownLane()]);
      continue;
    }
    if (Lane.getKind() == VPLane::Kind::ScalableLast) {
      NewOps.push_back(
          Builder.createNaryOp(VPInstruction::ExtractLastElement, {Op}));
      continue;
    }
    if (vputils::isSingleScalar(Op)) {
      NewOps.push_back(Op);
      continue;
    }

    // Look through buildvector to avoid unnecessary extracts.
    if (match(Op, m_BuildVector())) {
      NewOps.push_back(
          cast<VPInstruction>(Op)->getOperand(Lane.getKnownLane()));
      continue;
    }
    VPValue *Idx =
        Plan.getOrAddLiveIn(ConstantInt::get(IdxTy, Lane.getKnownLane()));
    VPValue *Ext = Builder.createNaryOp(Instruction::ExtractElement, {Op, Idx});
    NewOps.push_back(Ext);
  }

  VPRecipeWithIRFlags *New;
  if (auto *RepR = dyn_cast<VPReplicateRecipe>(DefR)) {
    // TODO: have cloning of replicate recipes also provide the desired result
    // coupled with setting its operands to NewOps (deriving IsSingleScalar and
    // Mask from the operands?)
    New =
        new VPReplicateRecipe(RepR->getUnderlyingInstr(), NewOps,
                              /*IsSingleScalar=*/true, /*Mask=*/nullptr, *RepR);
  } else {
    assert(isa<VPInstruction>(DefR) &&
           "DefR must be a VPReplicateRecipe or VPInstruction");
    New = DefR->clone();
    for (const auto &[Idx, Op] : enumerate(NewOps)) {
      New->setOperand(Idx, Op);
    }
  }
  New->transferFlags(*DefR);
  New->insertBefore(DefR);
  return New;
}

void VPlanTransforms::replicateByVF(VPlan &Plan, ElementCount VF) {
  Type *IdxTy = IntegerType::get(
      Plan.getScalarHeader()->getIRBasicBlock()->getContext(), 32);

  // Visit all VPBBs outside the loop region and directly inside the top-level
  // loop region.
  auto VPBBsOutsideLoopRegion = VPBlockUtils::blocksOnly<VPBasicBlock>(
      vp_depth_first_shallow(Plan.getEntry()));
  auto VPBBsInsideLoopRegion = VPBlockUtils::blocksOnly<VPBasicBlock>(
      vp_depth_first_shallow(Plan.getVectorLoopRegion()->getEntry()));
  auto VPBBsToUnroll =
      concat<VPBasicBlock *>(VPBBsOutsideLoopRegion, VPBBsInsideLoopRegion);
  // A mapping of current VPValue definitions to collections of new VPValues
  // defined per lane. Serves to hook-up potential users of current VPValue
  // definition that are replicated-per-VF later.
  DenseMap<VPValue *, SmallVector<VPValue *>> Def2LaneDefs;
  // The removal of current recipes being replaced by new ones needs to be
  // delayed after Def2LaneDefs is no longer in use.
  SmallVector<VPRecipeBase *> ToRemove;
  for (VPBasicBlock *VPBB : VPBBsToUnroll) {
    for (VPRecipeBase &R : make_early_inc_range(*VPBB)) {
      if (!isa<VPInstruction, VPReplicateRecipe>(&R) ||
          (isa<VPReplicateRecipe>(&R) &&
           cast<VPReplicateRecipe>(&R)->isSingleScalar()) ||
          (isa<VPInstruction>(&R) &&
           !cast<VPInstruction>(&R)->doesGeneratePerAllLanes()))
        continue;

      auto *DefR = cast<VPRecipeWithIRFlags>(&R);
      VPBuilder Builder(DefR);
      if (DefR->getNumUsers() == 0) {
        // Create single-scalar version of DefR for all lanes.
        for (unsigned I = 0; I != VF.getKnownMinValue(); ++I)
          cloneForLane(Plan, Builder, IdxTy, DefR, VPLane(I), Def2LaneDefs);
        DefR->eraseFromParent();
        continue;
      }
      /// Create single-scalar version of DefR for all lanes.
      SmallVector<VPValue *> LaneDefs;
      for (unsigned I = 0; I != VF.getKnownMinValue(); ++I)
        LaneDefs.push_back(
            cloneForLane(Plan, Builder, IdxTy, DefR, VPLane(I), Def2LaneDefs));

      Def2LaneDefs[DefR] = LaneDefs;
      /// Users that only demand the first lane can use the definition for lane
      /// 0.
      DefR->replaceUsesWithIf(LaneDefs[0], [DefR](VPUser &U, unsigned) {
        return U.onlyFirstLaneUsed(DefR);
      });

      // Update each build vector user that currently has DefR as its only
      // operand, to have all LaneDefs as its operands.
      for (VPUser *U : to_vector(DefR->users())) {
        auto *VPI = dyn_cast<VPInstruction>(U);
        if (!VPI || (VPI->getOpcode() != VPInstruction::BuildVector &&
                     VPI->getOpcode() != VPInstruction::BuildStructVector))
          continue;
        assert(VPI->getNumOperands() == 1 &&
               "Build(Struct)Vector must have a single operand before "
               "replicating by VF");
        VPI->setOperand(0, LaneDefs[0]);
        for (VPValue *LaneDef : drop_begin(LaneDefs))
          VPI->addOperand(LaneDef);
      }
      ToRemove.push_back(DefR);
    }
  }
  for (auto *R : reverse(ToRemove))
    R->eraseFromParent();
}
