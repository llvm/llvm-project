//===- VPlanWideningDecisions.cpp - VPlan-based widening decisions --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements VPlan-based widening decisions, which convert
/// initial recipes into widened, scalarized, interleaved or otherwise
/// specialized recipes.
///
//===----------------------------------------------------------------------===//

#include "VPRecipeBuilder.h"
#include "VPlan.h"
#include "VPlanAnalysis.h"
#include "VPlanCFG.h"
#include "VPlanDominatorTree.h"
#include "VPlanHelpers.h"
#include "VPlanPatternMatch.h"
#include "VPlanTransforms.h"
#include "VPlanUtils.h"
#include "VPlanVerifier.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/Analysis/InstSimplifyFolder.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
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
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

using namespace llvm;
using namespace VPlanPatternMatch;
using namespace SCEVPatternMatch;

/// If the pointer operand \p Addr of a memory access is an affine AddRec
/// w.r.t. \p L with a constant stride, return the stride in units of
/// \p AccessTy. Otherwise return std::nullopt.
static std::optional<int64_t> getConstantStride(VPValue *Addr, Type *AccessTy,
                                                PredicatedScalarEvolution &PSE,
                                                const Loop *L) {
  const SCEV *AddrSCEV = vputils::getSCEVExprForVPValue(Addr, PSE, L);
  auto *AddRec = dyn_cast<SCEVAddRecExpr>(AddrSCEV);
  if (!AddRec)
    return {};

  return getStrideFromAddRec(AddRec, L, AccessTy, /*Ptr=*/nullptr, PSE);
}

bool VPlanTransforms::tryToConvertVPInstructionsToVPRecipes(
    VPlan &Plan, const TargetLibraryInfo &TLI, PredicatedScalarEvolution &PSE,
    Loop *OuterLoop) {

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

      // Atomic accesses and fences have ordering/atomicity semantics that
      // cannot be preserved by lane-wise widening.
      if (isa<AtomicRMWInst, AtomicCmpXchgInst, FenceInst>(Inst))
        return false;

      VPRecipeBase *NewRecipe = nullptr;
      if (auto *PhiR = dyn_cast<VPPhi>(&Ingredient)) {
        auto *Phi = cast<PHINode>(PhiR->getUnderlyingValue());
        NewRecipe = new VPWidenPHIRecipe(PhiR->operands(), PhiR->getDebugLoc(),
                                         Phi->getName());
      } else if (auto *VPI = dyn_cast<VPInstruction>(&Ingredient)) {
        assert(!isa<PHINode>(Inst) && "phis should be handled above");
        // Create VPWidenMemoryRecipe for loads and stores.
        if (LoadInst *Load = dyn_cast<LoadInst>(Inst)) {
          bool IsConsecutive =
              getConstantStride(VPI->getOperand(0), VPI->getScalarType(), PSE,
                                OuterLoop) == 1;
          NewRecipe = new VPWidenLoadRecipe(*Load, Ingredient.getOperand(0),
                                            nullptr /*Mask*/, IsConsecutive,
                                            *VPI, Ingredient.getDebugLoc());
        } else if (StoreInst *Store = dyn_cast<StoreInst>(Inst)) {
          bool IsConsecutive =
              getConstantStride(VPI->getOperand(1),
                                VPI->getOperand(0)->getScalarType(), PSE,
                                OuterLoop) == 1;
          NewRecipe = new VPWidenStoreRecipe(
              *Store, Ingredient.getOperand(1), Ingredient.getOperand(0),
              nullptr /*Mask*/, IsConsecutive, *VPI, Ingredient.getDebugLoc());
        } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Inst)) {
          NewRecipe = new VPWidenGEPRecipe(GEP->getSourceElementType(),
                                           Ingredient.operands(), *VPI,
                                           Ingredient.getDebugLoc(), GEP);
        } else if (CallInst *CI = dyn_cast<CallInst>(Inst)) {
          Intrinsic::ID VectorID = getVectorIntrinsicIDForCall(CI, &TLI);
          if (VectorID == Intrinsic::not_intrinsic)
            return false;

          // The noalias.scope.decl intrinsic declares a noalias scope that
          // is valid for a single iteration. Emitting it as a single-scalar
          // replicate would incorrectly extend the scope across multiple
          // original iterations packed into one vector iteration.
          // FIXME: If we want to vectorize this loop, then we have to drop
          // all the associated !alias.scope and !noalias.
          if (VectorID == Intrinsic::experimental_noalias_scope_decl)
            return false;

          // These intrinsics are recognized by getVectorIntrinsicIDForCall
          // but are not widenable. Emit them as replicate instead of widening.
          if (VectorID == Intrinsic::assume ||
              VectorID == Intrinsic::lifetime_end ||
              VectorID == Intrinsic::lifetime_start ||
              VectorID == Intrinsic::sideeffect ||
              VectorID == Intrinsic::pseudoprobe) {
            // If the operand of llvm.assume holds before vectorization, it will
            // also hold per lane.
            // llvm.pseudoprobe requires to be duplicated per lane for accurate
            // sample count.
            const bool IsSingleScalar = VectorID != Intrinsic::assume &&
                                        VectorID != Intrinsic::pseudoprobe;
            NewRecipe = new VPReplicateRecipe(CI, Ingredient.operands(),
                                              /*IsSingleScalar=*/IsSingleScalar,
                                              /*Mask=*/nullptr, *VPI, *VPI,
                                              Ingredient.getDebugLoc());
          } else {
            NewRecipe = new VPWidenIntrinsicRecipe(
                *CI, VectorID, drop_end(Ingredient.operands()), CI->getType(),
                VPIRFlags(*CI), *VPI, CI->getDebugLoc());
          }
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

/// This function tries convert extended in-loop reductions to
/// VPExpressionRecipe and clamp the \p Range if it is beneficial and
/// valid. The created recipe must be decomposed to its constituent
/// recipes before execution.
static VPExpressionRecipe *
tryToMatchAndCreateExtendedReduction(VPReductionRecipe *Red, VPCostContext &Ctx,
                                     VFRange &Range) {
  Type *RedTy = Red->getScalarType();
  VPValue *VecOp = Red->getVecOp();

  assert(!Red->isPartialReduction() &&
         "This path does not support partial reductions");

  // Clamp the range if using extended-reduction is profitable.
  auto IsExtendedRedValidAndClampRange =
      [&](unsigned Opcode, Instruction::CastOps ExtOpc, Type *SrcTy) -> bool {
    return LoopVectorizationPlanner::getDecisionAndClampRange(
        [&](ElementCount VF) {
          auto *SrcVecTy = cast<VectorType>(toVectorTy(SrcTy, VF));
          TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput;

          InstructionCost ExtRedCost = InstructionCost::getInvalid();
          InstructionCost ExtCost =
              cast<VPWidenCastRecipe>(VecOp)->computeCost(VF, Ctx);
          InstructionCost RedCost = Red->computeCost(VF, Ctx);

          assert(!RedTy->isFloatingPointTy() &&
                 "getExtendedReductionCost only supports integer types");
          ExtRedCost = Ctx.TTI.getExtendedReductionCost(
              Opcode, ExtOpc == Instruction::CastOps::ZExt, RedTy, SrcVecTy,
              Red->getFastMathFlagsOrNone(), CostKind);
          return ExtRedCost.isValid() && ExtRedCost < ExtCost + RedCost;
        },
        Range);
  };

  VPValue *A;
  // Match reduce(ext)).
  if (match(VecOp, m_Isa<VPWidenCastRecipe>(m_ZExtOrSExt(m_VPValue(A)))) &&
      IsExtendedRedValidAndClampRange(
          RecurrenceDescriptor::getOpcode(Red->getRecurrenceKind()),
          cast<VPWidenCastRecipe>(VecOp)->getOpcode(), A->getScalarType()))
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
///   reduce.fadd(fmul(ext(A), ext(B)))
static VPExpressionRecipe *
tryToMatchAndCreateMulAccumulateReduction(VPReductionRecipe *Red,
                                          VPCostContext &Ctx, VFRange &Range) {
  unsigned Opcode = RecurrenceDescriptor::getOpcode(Red->getRecurrenceKind());
  if (Opcode != Instruction::Add && Opcode != Instruction::Sub &&
      Opcode != Instruction::FAdd)
    return nullptr;

  assert(!Red->isPartialReduction() &&
         "This path does not support partial reductions");
  Type *RedTy = Red->getScalarType();

  // Clamp the range if using multiply-accumulate-reduction is profitable.
  auto IsMulAccValidAndClampRange =
      [&](VPWidenRecipe *Mul, VPWidenCastRecipe *Ext0, VPWidenCastRecipe *Ext1,
          VPWidenCastRecipe *OuterExt) -> bool {
    return LoopVectorizationPlanner::getDecisionAndClampRange(
        [&](ElementCount VF) {
          TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput;
          Type *SrcTy = Ext0 ? Ext0->getOperand(0)->getScalarType() : RedTy;
          InstructionCost MulAccCost;

          // getMulAccReductionCost for in-loop reductions does not support
          // mixed or floating-point extends.
          if (Ext0 && Ext1 &&
              (Ext0->getOpcode() != Ext1->getOpcode() ||
               Ext0->getOpcode() == Instruction::CastOps::FPExt))
            return false;

          bool IsZExt =
              !Ext0 || Ext0->getOpcode() == Instruction::CastOps::ZExt;
          auto *SrcVecTy = cast<VectorType>(toVectorTy(SrcTy, VF));
          MulAccCost = Ctx.TTI.getMulAccReductionCost(IsZExt, Opcode, RedTy,
                                                      SrcVecTy, CostKind);

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

  if (RedTy->isFloatingPointTy())
    return nullptr;

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
  auto ExtendAndReplaceConstantOp = [](VPWidenCastRecipe *ExtA,
                                       VPWidenCastRecipe *&ExtB, VPValue *&ValB,
                                       VPWidenRecipe *Mul) {
    if (!ExtA || ExtB || !isa<VPIRValue>(ValB))
      return;
    Type *NarrowTy = ExtA->getOperand(0)->getScalarType();
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
    Type *WideTy = ExtA->getScalarType();
    ValB = ExtB = Builder.createWidenCast(ExtOpc, Trunc, WideTy);
    Mul->setOperand(1, ExtB);
  };

  // Try to match reduce.add(mul(...)).
  if (match(VecOp, m_Mul(m_VPValue(A), m_VPValue(B)))) {
    auto *RecipeA = dyn_cast<VPWidenCastRecipe>(A);
    auto *RecipeB = dyn_cast<VPWidenCastRecipe>(B);
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
    auto *Ext0 = dyn_cast<VPWidenCastRecipe>(A);
    auto *Ext1 = dyn_cast<VPWidenCastRecipe>(B);

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
          Ext0->getOpcode(), Ext0->getOperand(0), Ext->getScalarType(), nullptr,
          *Ext0, *Ext0, Ext0->getDebugLoc());
      NewExt0->insertBefore(Ext0);

      VPWidenCastRecipe *NewExt1 = NewExt0;
      if (Ext0 != Ext1) {
        NewExt1 = new VPWidenCastRecipe(Ext1->getOpcode(), Ext1->getOperand(0),
                                        Ext->getScalarType(), nullptr, *Ext1,
                                        *Ext1, Ext1->getDebugLoc());
        NewExt1->insertBefore(Ext1);
      }
      auto *NewMul = Mul->cloneWithOperands({NewExt0, NewExt1});
      NewMul->insertBefore(Mul);
      Ext->replaceAllUsesWith(NewMul);
      Ext->eraseFromParent();
      Mul->eraseFromParent();
      return new VPExpressionRecipe(NewExt0, NewExt1, NewMul, Red);
    }
  }
  return nullptr;
}

/// This function tries to create abstract recipes from the reduction recipe for
/// following optimizations and cost estimation.
static void tryToCreateAbstractReductionRecipe(VPReductionRecipe *Red,
                                               VPCostContext &Ctx,
                                               VFRange &Range) {
  // Creation of VPExpressions for partial reductions is entirely handled in
  // transformToPartialReduction.
  assert(!Red->isPartialReduction() &&
         "This path does not support partial reductions");

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

namespace {

using ExtendKind = TTI::PartialReductionExtendKind;
struct ReductionExtend {
  Type *SrcType = nullptr;
  ExtendKind Kind = ExtendKind::PR_None;
};

/// Describes the extends used to compute the extended reduction operand.
/// ExtendB is optional. If ExtendB is present, ExtendsUser is a binary
/// operation.
struct ExtendedReductionOperand {
  /// The recipe that consumes the extends.
  VPWidenRecipe *ExtendsUser = nullptr;
  /// Extend descriptions (inputs to getPartialReductionCost).
  ReductionExtend ExtendA, ExtendB;
};

/// A chain of recipes that form a partial reduction. Matches either
///   reduction_bin_op (extended op, accumulator), or
///   reduction_bin_op (accumulator, extended op).
/// The possible forms of the "extended op" are listed in
/// matchExtendedReductionOperand.
struct VPPartialReductionChain {
  /// The top-level binary operation that forms the reduction to a scalar
  /// after the loop body.
  VPWidenRecipe *ReductionBinOp = nullptr;
  /// The user of the extends that is then reduced.
  ExtendedReductionOperand ExtendedOp;
  /// The recurrence kind for the entire partial reduction chain.
  /// This allows distinguishing between Sub and AddWithSub recurrences,
  /// when the ReductionBinOp is a Instruction::Sub.
  RecurKind RK;
  /// The index of the accumulator operand of ReductionBinOp. The extended op
  /// is `1 - AccumulatorOpIdx`.
  unsigned AccumulatorOpIdx;
  unsigned ScaleFactor;
  /// Optional blend to represent predication for the block that updates the
  /// reduction.
  VPBlendRecipe *Blend = nullptr;
};

// Return the incoming index of the single-use value in the blend, which is
// expected to be the predicated reduction update.
static std::optional<unsigned>
getBlendReductionUpdateValueIdx(VPBlendRecipe *Blend) {
  assert(Blend && !Blend->isNormalized() &&
         Blend->getNumIncomingValues() == 2 &&
         "Expected a non-normalized blend with two incoming values");
  bool FirstIncomingHasOneUse = Blend->getIncomingValue(0)->hasOneUse();

  // Only the update value should have one use (the blend). The previous
  // value should always have at least two uses, the blend and the reduction.
  if (FirstIncomingHasOneUse == Blend->getIncomingValue(1)->hasOneUse())
    return std::nullopt;
  return FirstIncomingHasOneUse ? 0 : 1;
}

static VPSingleDefRecipe *
optimizeExtendsForPartialReduction(VPSingleDefRecipe *Op) {
  // reduce.add(mul(ext(A), C))
  // -> reduce.add(mul(ext(A), ext(trunc(C))))
  const APInt *Const;
  if (match(Op, m_Mul(m_ZExtOrSExt(m_VPValue()), m_APInt(Const)))) {
    auto *ExtA = cast<VPWidenCastRecipe>(Op->getOperand(0));
    Instruction::CastOps ExtOpc = ExtA->getOpcode();
    Type *NarrowTy = ExtA->getOperand(0)->getScalarType();
    if (!Op->hasOneUse() ||
        !llvm::canConstantBeExtended(
            Const, NarrowTy, TTI::getPartialReductionExtendKind(ExtOpc)))
      return Op;

    VPBuilder Builder(Op);
    auto *Trunc = Builder.createWidenCast(Instruction::CastOps::Trunc,
                                          Op->getOperand(1), NarrowTy);
    Type *WideTy = ExtA->getScalarType();
    Op->setOperand(1, Builder.createWidenCast(ExtOpc, Trunc, WideTy));
    return Op;
  }

  // reduce.add(abs(sub(ext(A), ext(B))))
  // -> reduce.add(ext(absolute-difference(A, B)))
  VPValue *X, *Y;
  if (match(Op, m_WidenIntrinsic<Intrinsic::abs>(m_Sub(
                    m_ZExtOrSExt(m_VPValue(X)), m_ZExtOrSExt(m_VPValue(Y)))))) {
    auto *Sub = Op->getOperand(0)->getDefiningRecipe();
    auto *Ext = cast<VPWidenCastRecipe>(Sub->getOperand(0));
    assert(Ext->getOpcode() ==
               cast<VPWidenCastRecipe>(Sub->getOperand(1))->getOpcode() &&
           "Expected both the LHS and RHS extends to be the same");
    bool IsSigned = Ext->getOpcode() == Instruction::SExt;
    VPBuilder Builder(Op);
    Type *SrcTy = X->getScalarType();
    auto *FreezeX = Builder.insert(new VPWidenRecipe(Instruction::Freeze, {X}));
    auto *FreezeY = Builder.insert(new VPWidenRecipe(Instruction::Freeze, {Y}));
    auto *Max = Builder.insert(
        new VPWidenIntrinsicRecipe(IsSigned ? Intrinsic::smax : Intrinsic::umax,
                                   {FreezeX, FreezeY}, SrcTy));
    auto *Min = Builder.insert(
        new VPWidenIntrinsicRecipe(IsSigned ? Intrinsic::smin : Intrinsic::umin,
                                   {FreezeX, FreezeY}, SrcTy));
    auto *AbsDiff =
        Builder.insert(new VPWidenRecipe(Instruction::Sub, {Max, Min}));
    return Builder.createWidenCast(Instruction::CastOps::ZExt, AbsDiff,
                                   Op->getScalarType());
  }

  // reduce.add(ext(mul(ext(A), ext(B))))
  // -> reduce.add(mul(wider_ext(A), wider_ext(B)))
  // TODO: Support this optimization for float types.
  if (match(Op, m_ZExtOrSExt(m_Mul(m_ZExtOrSExt(m_VPValue()),
                                   m_ZExtOrSExt(m_VPValue()))))) {
    auto *Ext = cast<VPWidenCastRecipe>(Op);
    auto *Mul = cast<VPWidenRecipe>(Ext->getOperand(0));
    auto *MulLHS = cast<VPWidenCastRecipe>(Mul->getOperand(0));
    auto *MulRHS = cast<VPWidenCastRecipe>(Mul->getOperand(1));
    if (!Mul->hasOneUse() ||
        (Ext->getOpcode() != MulLHS->getOpcode() && MulLHS != MulRHS) ||
        MulLHS->getOpcode() != MulRHS->getOpcode())
      return Op;
    VPBuilder Builder(Mul);
    auto *NewLHS = Builder.createWidenCast(
        MulLHS->getOpcode(), MulLHS->getOperand(0), Ext->getScalarType());
    auto *NewRHS = MulLHS == MulRHS
                       ? NewLHS
                       : Builder.createWidenCast(MulRHS->getOpcode(),
                                                 MulRHS->getOperand(0),
                                                 Ext->getScalarType());
    auto *NewMul = Mul->cloneWithOperands({NewLHS, NewRHS});
    Builder.insert(NewMul);
    Op->replaceAllUsesWith(NewMul);
    Op->eraseFromParent();
    Mul->eraseFromParent();
    return NewMul;
  }

  return Op;
}

static VPExpressionRecipe *
createPartialReductionExpression(VPReductionRecipe *Red) {
  VPValue *VecOp = Red->getVecOp();

  // reduce.[f]add(ext(op))
  //  -> VPExpressionRecipe(op, red)
  if (match(VecOp, m_WidenAnyExtend(m_VPValue())))
    return new VPExpressionRecipe(cast<VPWidenCastRecipe>(VecOp), Red);

  // reduce.[f]add(neg(ext(op)))
  // -> VPExpressionRecipe(op, sub/neg, red)
  if (match(VecOp, m_AnyNeg(m_WidenAnyExtend(m_VPValue())))) {
    auto *Neg = cast<VPWidenRecipe>(VecOp);
    auto *Ext =
        cast<VPWidenCastRecipe>(Neg->getOperand(Neg->getNumOperands() - 1));
    return new VPExpressionRecipe(Ext, Neg, Red);
  }

  // reduce.[f]add([f]mul(ext(a), ext(b)))
  //  -> VPExpressionRecipe(a, b, mul, red)
  if (match(VecOp, m_FMul(m_FPExt(m_VPValue()), m_FPExt(m_VPValue()))) ||
      match(VecOp,
            m_Mul(m_ZExtOrSExt(m_VPValue()), m_ZExtOrSExt(m_VPValue())))) {
    auto *Mul = cast<VPWidenRecipe>(VecOp);
    auto *ExtA = cast<VPWidenCastRecipe>(Mul->getOperand(0));
    auto *ExtB = cast<VPWidenCastRecipe>(Mul->getOperand(1));
    return new VPExpressionRecipe(ExtA, ExtB, Mul, Red);
  }

  // reduce.fadd(fneg(fmul(fpext(a), fpext(b))))
  //  -> VPExpressionRecipe(a, b, fmul, fsub, red)
  if (match(VecOp,
            m_FNeg(m_FMul(m_FPExt(m_VPValue()), m_FPExt(m_VPValue()))))) {
    auto *FNeg = cast<VPWidenRecipe>(VecOp);
    auto *FMul = cast<VPWidenRecipe>(FNeg->getOperand(0));
    auto *ExtA = cast<VPWidenCastRecipe>(FMul->getOperand(0));
    auto *ExtB = cast<VPWidenCastRecipe>(FMul->getOperand(1));
    return new VPExpressionRecipe(ExtA, ExtB, FMul, FNeg, Red);
  }

  // reduce.add(neg(mul(ext(a), ext(b))))
  //  -> VPExpressionRecipe(a, b, mul, sub, red)
  if (match(VecOp, m_Sub(m_ZeroInt(), m_Mul(m_ZExtOrSExt(m_VPValue()),
                                            m_ZExtOrSExt(m_VPValue()))))) {
    auto *Sub = cast<VPWidenRecipe>(VecOp);
    auto *Mul = cast<VPWidenRecipe>(Sub->getOperand(1));
    auto *ExtA = cast<VPWidenCastRecipe>(Mul->getOperand(0));
    auto *ExtB = cast<VPWidenCastRecipe>(Mul->getOperand(1));
    return new VPExpressionRecipe(ExtA, ExtB, Mul, Sub, Red);
  }

  llvm_unreachable("Unsupported expression");
}

// Helper to transform a partial reduction chain into a partial reduction
// recipe. Assumes profitability has been checked.
static void transformToPartialReduction(const VPPartialReductionChain &Chain,
                                        VPlan &Plan,
                                        VPReductionPHIRecipe *RdxPhi) {
  VPWidenRecipe *WidenRecipe = Chain.ReductionBinOp;
  assert(WidenRecipe->getNumOperands() == 2 && "Expected binary operation");

  VPValue *Accumulator = WidenRecipe->getOperand(Chain.AccumulatorOpIdx);
  auto *ExtendedOp = cast<VPSingleDefRecipe>(
      WidenRecipe->getOperand(1 - Chain.AccumulatorOpIdx));

  // FIXME: Do these transforms before invoking the cost-model.
  ExtendedOp = optimizeExtendsForPartialReduction(ExtendedOp);

  // Sub-reductions can be implemented in two ways:
  // (1) negate the operand in the vector loop (the default way).
  // (2) subtract the reduced value from the init value in the middle block.
  // Both ways keep the reduction itself as an 'add' reduction.
  //
  // The ISD nodes for partial reductions don't support folding the
  // sub/negation into its operands because the following is not a valid
  // transformation:
  //      sub(0, mul(ext(a), ext(b)))
  //   -> mul(ext(a), ext(sub(0, b)))
  //
  // It's therefore better to choose option (2) such that the partial
  // reduction is always positive (starting at '0') and to do a final
  // subtract in the middle block.
  if ((WidenRecipe->getOpcode() == Instruction::Sub &&
       Chain.RK != RecurKind::Sub) ||
      (WidenRecipe->getOpcode() == Instruction::FSub &&
       Chain.RK != RecurKind::FSub)) {
    VPBuilder Builder(WidenRecipe);
    Type *ElemTy = ExtendedOp->getScalarType();
    VPWidenRecipe *NegRecipe;
    if (WidenRecipe->getOpcode() == Instruction::FSub) {
      NegRecipe =
          new VPWidenRecipe(Instruction::FNeg, {ExtendedOp}, VPIRFlags(),
                            VPIRMetadata(), DebugLoc::getUnknown());
    } else {
      auto *Zero = Plan.getZero(ElemTy);
      NegRecipe =
          new VPWidenRecipe(Instruction::Sub, {Zero, ExtendedOp}, VPIRFlags(),
                            VPIRMetadata(), DebugLoc::getUnknown());
    }
    Builder.insert(NegRecipe);
    ExtendedOp = NegRecipe;
  }

  // Check if WidenRecipe is the final result of the reduction. If so, look
  // through the Select recipe introduced by tail-folding, otherwise look
  // through any Blend recipe introduced by predication for the block.
  VPValue *ExitSearch =
      Chain.Blend ? cast<VPValue>(Chain.Blend) : cast<VPValue>(WidenRecipe);

  VPValue *Cond = nullptr;
  VPValue *ExitValue = cast_or_null<VPInstruction>(
      findUserOf(ExitSearch, m_Select(m_VPValue(Cond), m_Specific(ExitSearch),
                                      m_Specific(RdxPhi))));

  if (Chain.Blend) {
    std::optional<unsigned> BlendReductionIdx =
        getBlendReductionUpdateValueIdx(Chain.Blend);
    assert(BlendReductionIdx &&
           Chain.Blend->getIncomingValue(*BlendReductionIdx) == WidenRecipe &&
           "Expected blend to contain the reduction update");
    VPValue *BlendCond = Chain.Blend->getMask(*BlendReductionIdx);
    Cond = ExitValue ? VPBuilder(WidenRecipe)
                           .createLogicalAnd(Cond, BlendCond,
                                             WidenRecipe->getDebugLoc())
                     : BlendCond;
  }

  bool IsLastInChain = RdxPhi->getBackedgeValue() == WidenRecipe ||
                       RdxPhi->getBackedgeValue() == ExitValue ||
                       RdxPhi->getBackedgeValue() == Chain.Blend;
  assert((!ExitValue || IsLastInChain) &&
         "if we found ExitValue, it must match RdxPhi's backedge value");

  Type *PhiType = RdxPhi->getScalarType();
  RecurKind RdxKind =
      PhiType->isFloatingPointTy() ? RecurKind::FAdd : RecurKind::Add;
  auto *PartialRed = new VPReductionRecipe(
      RdxKind,
      RdxKind == RecurKind::FAdd ? WidenRecipe->getFastMathFlagsOrNone()
                                 : FastMathFlags(),
      WidenRecipe->getUnderlyingInstr(), Accumulator, ExtendedOp, Cond,
      RdxUnordered{/*VFScaleFactor=*/Chain.ScaleFactor});
  PartialRed->insertBefore(WidenRecipe);

  if (ExitValue)
    ExitValue->replaceAllUsesWith(PartialRed);
  if (Chain.Blend)
    Chain.Blend->replaceAllUsesWith(PartialRed);
  WidenRecipe->replaceAllUsesWith(PartialRed);

  // For cost-model purposes, fold this into a VPExpression.
  VPExpressionRecipe *E = createPartialReductionExpression(PartialRed);
  E->insertBefore(WidenRecipe);
  PartialRed->replaceAllUsesWith(E);

  // We only need to update the PHI node once, which is when we find the
  // last reduction in the chain.
  if (!IsLastInChain)
    return;

  // Scale the PHI and ReductionStartVector by the VFScaleFactor
  assert(RdxPhi->getVFScaleFactor() == 1 && "scale factor must not be set");
  RdxPhi->setVFScaleFactor(Chain.ScaleFactor);

  auto *StartInst = cast<VPInstruction>(RdxPhi->getStartValue());
  assert(StartInst->getOpcode() == VPInstruction::ReductionStartVector);
  auto *NewScaleFactor = Plan.getConstantInt(32, Chain.ScaleFactor);
  StartInst->setOperand(2, NewScaleFactor);

  // If this is the last value in a sub-reduction chain, then update the PHI
  // node to start at `0` and update the reduction-result to subtract from
  // the PHI's start value.
  if (Chain.RK != RecurKind::Sub && Chain.RK != RecurKind::FSub)
    return;

  VPValue *OldStartValue = StartInst->getOperand(0);
  StartInst->setOperand(0, StartInst->getOperand(1));

  // Replace reduction_result by 'sub (startval, reductionresult)'.
  VPInstruction *RdxResult = vputils::findComputeReductionResult(RdxPhi);
  assert(RdxResult && "Could not find reduction result");

  VPBuilder Builder = VPBuilder::getToInsertAfter(RdxResult);
  unsigned SubOpc = Chain.RK == RecurKind::FSub ? Instruction::BinaryOps::FSub
                                                : Instruction::BinaryOps::Sub;
  VPInstruction *NewResult = Builder.createNaryOp(
      SubOpc, {OldStartValue, RdxResult}, VPIRFlags::getDefaultFlags(SubOpc),
      RdxPhi->getDebugLoc());
  RdxResult->replaceUsesWithIf(
      NewResult,
      [&NewResult](VPUser &U, unsigned Idx) { return &U != NewResult; });
}

/// Returns the cost of a link in a partial-reduction chain for a given VF.
static InstructionCost
getPartialReductionLinkCost(VPCostContext &CostCtx,
                            const VPPartialReductionChain &Link,
                            ElementCount VF) {
  Type *RdxType = Link.ReductionBinOp->getScalarType();
  const ExtendedReductionOperand &ExtendedOp = Link.ExtendedOp;
  std::optional<unsigned> BinOpc = std::nullopt;
  // If ExtendB is not none, then the "ExtendsUser" is the binary operation.
  if (ExtendedOp.ExtendB.Kind != ExtendKind::PR_None)
    BinOpc = ExtendedOp.ExtendsUser->getOpcode();

  std::optional<llvm::FastMathFlags> Flags;
  if (RdxType->isFloatingPointTy())
    Flags = Link.ReductionBinOp->getFastMathFlagsOrNone();

  auto GetLinkOpcode = [&Link]() -> unsigned {
    switch (Link.RK) {
    case RecurKind::Sub:
      return Instruction::Add;
    case RecurKind::FSub:
      return Instruction::FAdd;
    default:
      return Link.ReductionBinOp->getOpcode();
    }
  };

  return CostCtx.TTI.getPartialReductionCost(
      GetLinkOpcode(), ExtendedOp.ExtendA.SrcType, ExtendedOp.ExtendB.SrcType,
      RdxType, VF, ExtendedOp.ExtendA.Kind, ExtendedOp.ExtendB.Kind, BinOpc,
      CostCtx.CostKind, Flags);
}

static ExtendKind getPartialReductionExtendKind(VPWidenCastRecipe *Cast) {
  return TTI::getPartialReductionExtendKind(Cast->getOpcode());
}

/// Checks if \p Op (which is an operand of \p UpdateR) is an extended reduction
/// operand. This is an operand where the source of the value (e.g. a load) has
/// been extended (sext, zext, or fpext) before it is used in the reduction.
///
/// Possible forms matched by this function:
///  - UpdateR(PrevValue, ext(...))
///  - UpdateR(PrevValue, mul(ext(...), ext(...)))
///  - UpdateR(PrevValue, mul(ext(...), Constant))
///  - UpdateR(PrevValue, ext(mul(ext(...), ext(...))))
///  - UpdateR(PrevValue, ext(mul(ext(...), Constant)))
///  - UpdateR(PrevValue, abs(sub(ext(...), ext(...)))
///
/// Note: The second operand of UpdateR corresponds to \p Op in the examples.
static std::optional<ExtendedReductionOperand>
matchExtendedReductionOperand(VPWidenRecipe *UpdateR, VPValue *Op) {
  assert(is_contained(UpdateR->operands(), Op) &&
         "Op should be operand of UpdateR");

  // Try matching an absolute difference operand of the form
  // `abs(sub(ext(A), ext(B)))`. This will be later transformed into
  // `ext(absolute-difference(A, B))`. This allows us to perform the absolute
  // difference on a wider type and get the extend for "free" from the partial
  // reduction.
  VPValue *X, *Y;
  if (Op->hasOneUse() &&
      match(Op, m_WidenIntrinsic<Intrinsic::abs>(
                    m_OneUse(m_Sub(m_WidenAnyExtend(m_VPValue(X)),
                                   m_WidenAnyExtend(m_VPValue(Y))))))) {
    auto *Abs = cast<VPWidenIntrinsicRecipe>(Op);
    auto *Sub = cast<VPWidenRecipe>(Abs->getOperand(0));
    auto *LHSExt = cast<VPWidenCastRecipe>(Sub->getOperand(0));
    auto *RHSExt = cast<VPWidenCastRecipe>(Sub->getOperand(1));
    Type *LHSInputType = X->getScalarType();
    Type *RHSInputType = Y->getScalarType();
    if (LHSInputType != RHSInputType ||
        LHSExt->getOpcode() != RHSExt->getOpcode())
      return std::nullopt;
    // Note: This is essentially the same as matching ext(...) as we will
    // rewrite this operand to ext(absolute-difference(A, B)).
    return ExtendedReductionOperand{
        Sub,
        /*ExtendA=*/{LHSInputType, getPartialReductionExtendKind(LHSExt)},
        /*ExtendB=*/{}};
  }

  std::optional<TTI::PartialReductionExtendKind> OuterExtKind;
  if (match(Op, m_WidenAnyExtend(m_VPValue()))) {
    auto *CastRecipe = cast<VPWidenCastRecipe>(Op);
    VPValue *CastSource = CastRecipe->getOperand(0);
    OuterExtKind = getPartialReductionExtendKind(CastRecipe);
    if (match(CastSource, m_Mul(m_VPValue(), m_VPValue())) ||
        match(CastSource, m_FMul(m_VPValue(), m_VPValue()))) {
      // Match: ext(mul(...))
      // Record the outer extend kind and set `Op` to the mul. We can then match
      // this as a binary operation. Note: We can optimize out the outer extend
      // by widening the inner extends to match it. See
      // optimizeExtendsForPartialReduction.
      Op = CastSource;
    } else {
      return ExtendedReductionOperand{
          UpdateR,
          /*ExtendA=*/{CastSource->getScalarType(), *OuterExtKind},
          /*ExtendB=*/{}};
    }
  }

  if (!Op->hasOneUse())
    return std::nullopt;

  VPWidenRecipe *MulOp = dyn_cast<VPWidenRecipe>(Op);
  if (!MulOp ||
      !is_contained({Instruction::Mul, Instruction::FMul}, MulOp->getOpcode()))
    return std::nullopt;

  // The rest of the matching assumes `Op` is a (possibly extended) mul
  // operation.

  VPValue *LHS = MulOp->getOperand(0);
  VPValue *RHS = MulOp->getOperand(1);

  // The LHS of the operation must always be an extend.
  if (!match(LHS, m_WidenAnyExtend(m_VPValue())))
    return std::nullopt;

  auto *LHSCast = cast<VPWidenCastRecipe>(LHS);
  Type *LHSInputType = LHSCast->getOperand(0)->getScalarType();
  ExtendKind LHSExtendKind = getPartialReductionExtendKind(LHSCast);

  // The RHS of the operation can be an extend or a constant integer.
  const APInt *RHSConst = nullptr;
  VPWidenCastRecipe *RHSCast = nullptr;
  if (match(RHS, m_WidenAnyExtend(m_VPValue())))
    RHSCast = cast<VPWidenCastRecipe>(RHS);
  else if (!match(RHS, m_APInt(RHSConst)) ||
           !canConstantBeExtended(RHSConst, LHSInputType, LHSExtendKind))
    return std::nullopt;

  // The outer extend kind must match the inner extends for folding.
  for (VPWidenCastRecipe *Cast : {LHSCast, RHSCast})
    if (Cast && OuterExtKind &&
        getPartialReductionExtendKind(Cast) != OuterExtKind)
      return std::nullopt;

  Type *RHSInputType = LHSInputType;
  ExtendKind RHSExtendKind = LHSExtendKind;
  if (RHSCast) {
    RHSInputType = RHSCast->getOperand(0)->getScalarType();
    RHSExtendKind = getPartialReductionExtendKind(RHSCast);
  }

  return ExtendedReductionOperand{
      MulOp, {LHSInputType, LHSExtendKind}, {RHSInputType, RHSExtendKind}};
}

/// Examines each operation in the reduction chain corresponding to \p RedPhiR,
/// and determines if the target can use a cheaper operation with a wider
/// per-iteration input VF and narrower PHI VF. If successful, returns the chain
/// of operations in the reduction.
static std::optional<SmallVector<VPPartialReductionChain>>
getScaledReductions(VPReductionPHIRecipe *RedPhiR) {
  // Get the backedge value from the reduction PHI and find the
  // ComputeReductionResult that uses it (directly or through a select for
  // predicated reductions).
  auto *RdxResult = vputils::findComputeReductionResult(RedPhiR);
  if (!RdxResult)
    return std::nullopt;
  VPValue *ExitValue = RdxResult->getOperand(0);
  match(ExitValue, m_Select(m_VPValue(), m_VPValue(ExitValue), m_VPValue()));

  SmallVector<VPPartialReductionChain> Chain;
  RecurKind RK = RedPhiR->getRecurrenceKind();
  Type *PhiType = RedPhiR->getScalarType();
  TypeSize PHISize = PhiType->getPrimitiveSizeInBits();

  // Work backwards from the ExitValue examining each reduction operation.
  VPValue *CurrentValue = ExitValue;
  while (CurrentValue != RedPhiR) {
    VPBlendRecipe *Blend = dyn_cast<VPBlendRecipe>(CurrentValue);
    std::optional<unsigned> BlendReductionIdx;
    if (Blend) {
      assert(!Blend->isNormalized() && "Expect Blend not to be normalized.");
      if (Blend->getNumIncomingValues() != 2)
        return std::nullopt;

      BlendReductionIdx = getBlendReductionUpdateValueIdx(Blend);
      if (!BlendReductionIdx)
        return std::nullopt;

      CurrentValue = Blend->getIncomingValue(*BlendReductionIdx);
    }

    auto *UpdateR = dyn_cast<VPWidenRecipe>(CurrentValue);
    if (!UpdateR || !Instruction::isBinaryOp(UpdateR->getOpcode()))
      return std::nullopt;

    VPValue *Op = UpdateR->getOperand(1);
    VPValue *PrevValue = UpdateR->getOperand(0);

    // Find the extended operand. The other operand (PrevValue) is the next link
    // in the reduction chain.
    std::optional<ExtendedReductionOperand> ExtendedOp =
        matchExtendedReductionOperand(UpdateR, Op);
    if (!ExtendedOp) {
      ExtendedOp = matchExtendedReductionOperand(UpdateR, PrevValue);
      if (!ExtendedOp)
        return std::nullopt;
      std::swap(Op, PrevValue);
    }

    // Look for VPBlend(reduce(PrevValue, Op), PrevValue), where
    // reduce is equal to CurrentValue. This can be lowered as
    // a conditional reduction by hoisting the select to the inputs.
    if (Blend && Blend->getIncomingValue(1 - *BlendReductionIdx) != PrevValue)
      return std::nullopt;

    Type *ExtSrcType = ExtendedOp->ExtendA.SrcType;
    TypeSize ExtSrcSize = ExtSrcType->getPrimitiveSizeInBits();
    if (!PHISize.hasKnownScalarFactor(ExtSrcSize))
      return std::nullopt;

    VPPartialReductionChain Link(
        {UpdateR, *ExtendedOp, RK,
         PrevValue == UpdateR->getOperand(0) ? 0U : 1U,
         static_cast<unsigned>(PHISize.getKnownScalarFactor(ExtSrcSize)),
         Blend});
    Chain.push_back(Link);
    CurrentValue = PrevValue;
  }

  // The chain links were collected by traversing backwards from the exit value.
  // Reverse the chains so they are in program order.
  std::reverse(Chain.begin(), Chain.end());
  return Chain;
}
} // namespace

void VPlanTransforms::createPartialReductions(VPlan &Plan,
                                              VPCostContext &CostCtx,
                                              VFRange &Range) {
  // Find all possible valid partial reductions, grouping chains by their PHI.
  // This grouping allows invalidating the whole chain, if any link is not a
  // valid partial reduction.
  MapVector<VPReductionPHIRecipe *, SmallVector<VPPartialReductionChain>>
      ChainsByPhi;
  VPBasicBlock *HeaderVPBB = Plan.getVectorLoopRegion()->getEntryBasicBlock();
  for (VPRecipeBase &R : HeaderVPBB->phis()) {
    auto *RedPhiR = dyn_cast<VPReductionPHIRecipe>(&R);
    if (!RedPhiR)
      continue;

    if (auto Chains = getScaledReductions(RedPhiR))
      ChainsByPhi.try_emplace(RedPhiR, std::move(*Chains));
  }

  if (ChainsByPhi.empty())
    return;

  // Build set of partial reduction operations and blends for user validation
  // and a map of reduction bin ops to their scale factors for scale validation.
  SmallPtrSet<VPRecipeBase *, 4> PartialReductionOps;
  SmallPtrSet<VPBlendRecipe *, 4> PartialReductionBlends;
  DenseMap<VPSingleDefRecipe *, unsigned> ScaledReductionMap;
  for (const auto &[_, Chains] : ChainsByPhi)
    for (const VPPartialReductionChain &Chain : Chains) {
      PartialReductionOps.insert(Chain.ExtendedOp.ExtendsUser);
      if (Chain.Blend)
        PartialReductionBlends.insert(Chain.Blend);
      ScaledReductionMap[Chain.ReductionBinOp] = Chain.ScaleFactor;
    }

  // A partial reduction is invalid if any of its extends are used by
  // something that isn't another partial reduction. This is because the
  // extends are intended to be lowered along with the reduction itself.
  auto ExtendUsersValid = [&](VPValue *Ext) {
    return !isa<VPWidenCastRecipe>(Ext) || all_of(Ext->users(), [&](VPUser *U) {
      return PartialReductionOps.contains(cast<VPRecipeBase>(U));
    });
  };

  auto IsProfitablePartialReductionChainForVF =
      [&](ArrayRef<VPPartialReductionChain> Chain, ElementCount VF) -> bool {
    InstructionCost PartialCost = 0, RegularCost = 0;

    // The chain is a profitable partial reduction chain if the cost of handling
    // the entire chain is cheaper when using partial reductions than when
    // handling the entire chain using regular reductions.
    for (const VPPartialReductionChain &Link : Chain) {
      const ExtendedReductionOperand &ExtendedOp = Link.ExtendedOp;
      InstructionCost LinkCost = getPartialReductionLinkCost(CostCtx, Link, VF);
      if (!LinkCost.isValid())
        return false;

      PartialCost += LinkCost;
      RegularCost += Link.ReductionBinOp->computeCost(VF, CostCtx);
      // If ExtendB is not none, then the "ExtendsUser" is the binary operation.
      if (ExtendedOp.ExtendB.Kind != ExtendKind::PR_None)
        RegularCost += ExtendedOp.ExtendsUser->computeCost(VF, CostCtx);
      for (VPValue *Op : ExtendedOp.ExtendsUser->operands())
        if (auto *Extend = dyn_cast<VPWidenCastRecipe>(Op))
          RegularCost += Extend->computeCost(VF, CostCtx);
    }
    return PartialCost.isValid() && PartialCost < RegularCost;
  };

  // Validate chains: check that extends are only used by partial reductions,
  // and that reduction bin ops are only used by other partial reductions with
  // matching scale factors, are outside the loop region or the select
  // introduced by tail-folding. Otherwise we would create users of scaled
  // reductions where the types of the other operands don't match.
  for (auto &[RedPhiR, Chains] : ChainsByPhi) {
    for (const VPPartialReductionChain &Chain : Chains) {
      if (!all_of(Chain.ExtendedOp.ExtendsUser->operands(), ExtendUsersValid)) {
        Chains.clear();
        break;
      }
      auto UseIsValid = [&, RedPhiR = RedPhiR](VPUser *U) {
        if (auto *PhiR = dyn_cast<VPReductionPHIRecipe>(U))
          return PhiR == RedPhiR;
        auto *R = cast<VPSingleDefRecipe>(U);

        if (auto *Blend = dyn_cast<VPBlendRecipe>(R))
          return Blend == Chain.Blend || PartialReductionBlends.contains(Blend);

        return Chain.ScaleFactor == ScaledReductionMap.lookup_or(R, 0) ||
               match(R, m_ComputeReductionResult(
                            m_Specific(Chain.ReductionBinOp))) ||
               match(R, m_Select(m_VPValue(), m_Specific(Chain.ReductionBinOp),
                                 m_Specific(RedPhiR)));
      };
      if (!all_of(Chain.ReductionBinOp->users(), UseIsValid)) {
        Chains.clear();
        break;
      }

      // Check if the compute-reduction-result is used by a sunk store.
      // TODO: Also form partial reductions in those cases.
      if (auto *RdxResult = vputils::findComputeReductionResult(RedPhiR)) {
        if (any_of(RdxResult->users(), [](VPUser *U) {
              auto *RepR = dyn_cast<VPReplicateRecipe>(U);
              return RepR && RepR->getOpcode() == Instruction::Store;
            })) {
          Chains.clear();
          break;
        }
      }
    }

    // Clear the chain if it is not profitable.
    if (!LoopVectorizationPlanner::getDecisionAndClampRange(
            [&, &Chains = Chains](ElementCount VF) {
              return IsProfitablePartialReductionChainForVF(Chains, VF);
            },
            Range))
      Chains.clear();
  }

  for (auto &[Phi, Chains] : ChainsByPhi)
    for (const VPPartialReductionChain &Chain : Chains)
      transformToPartialReduction(Chain, Plan, Phi);
}

void VPlanTransforms::makeMemOpWideningDecisions(VPlan &Plan, VFRange &Range,
                                                 VPRecipeBuilder &RecipeBuilder,
                                                 VPCostContext &CostCtx) {
  // Collect all loads/stores first. We will start with ones having simpler
  // decisions followed by more complex ones that are potentially
  // guided/dependent on the simpler ones.
  SmallVector<VPInstruction *> MemOps;
  for (VPBasicBlock *VPBB :
       VPBlockUtils::blocksOnly<VPBasicBlock>(vp_depth_first_shallow(
           Plan.getVectorLoopRegion()->getEntryBasicBlock()))) {
    for (VPRecipeBase &R : *VPBB) {
      auto *VPI = dyn_cast<VPInstruction>(&R);
      if (VPI && VPI->getUnderlyingValue() &&
          is_contained({Instruction::Load, Instruction::Store},
                       VPI->getOpcode()))
        MemOps.push_back(VPI);
    }
  }

  // Few helpers to process different kinds of memory operations.

  // To be used as argument to `VPlanTransforms::runPass` which explicitly
  // specified pass name, hence `VPlan &` parameter.
  auto ProcessSubset = [&](VPlan &, auto ProcessVPInst) {
    SmallVector<VPInstruction *> RemainingMemOps;
    for (VPInstruction *VPI : MemOps) {
      if (!ProcessVPInst(VPI))
        RemainingMemOps.push_back(VPI);
    }

    MemOps.clear();
    std::swap(MemOps, RemainingMemOps);
  };

  auto ReplaceWith = [&](VPInstruction *VPI, VPRecipeBase *New) {
    assert(New->getParent() && "New recipe must have been inserted");
    if (VPI->getOpcode() == Instruction::Load)
      VPI->replaceAllUsesWith(New->getVPSingleValue());
    VPI->eraseFromParent();

    // VPI has been processed.
    return true;
  };

  auto Scalarize = [&](VPInstruction *VPI) {
    return ReplaceWith(VPI, VPBuilder(VPI).insert(
                                RecipeBuilder.handleReplication(VPI, Range)));
  };

  VPBasicBlock *MiddleVPBB = Plan.getMiddleBlock();
  VPBuilder FinalRedStoresBuilder(MiddleVPBB, MiddleVPBB->getFirstNonPhi());
  VPlanTransforms::runPass(
      "lowerMemoryIdioms", ProcessSubset, Plan, [&](VPInstruction *VPI) {
        if (RecipeBuilder.replaceWithFinalIfReductionStore(
                VPI, FinalRedStoresBuilder))
          return true;

        // Filter out scalar VPlan for the remaining idioms.
        if (LoopVectorizationPlanner::getDecisionAndClampRange(
                [](ElementCount VF) { return VF.isScalar(); }, Range))
          return false;

        if (VPHistogramRecipe *Histogram = RecipeBuilder.widenIfHistogram(VPI))
          return ReplaceWith(VPI, VPBuilder(VPI).insert(Histogram));

        return false;
      });

  // Filter out scalar VPlan for the remaining memory operations.
  if (LoopVectorizationPlanner::getDecisionAndClampRange(
          [](ElementCount VF) { return VF.isScalar(); }, Range))
    return;

  // If the instruction's allocated size doesn't equal it's type size, it
  // requires padding and will be scalarized.
  VPlanTransforms::runPass(
      "scalarizeMemOpsWithIrregularTypes", ProcessSubset, Plan,
      [&](VPInstruction *VPI) {
        Instruction *I = VPI->getUnderlyingInstr();
        if (hasIrregularType(getLoadStoreType(I), I->getDataLayout()))
          return Scalarize(VPI);

        return false;
      });

  if (!RecipeBuilder.prefersVectorizedAddressing()) {
    VPlanTransforms::runPass(
        "makeVPlanMemOpDecision", ProcessSubset, Plan, [&](VPInstruction *VPI) {
          Instruction *I = VPI->getUnderlyingInstr();
          bool IsLoad = VPI->getOpcode() == Instruction::Load;
          if (RecipeBuilder.isPredicatedInst(I) || !IsLoad ||
              !vputils::isUsedByLoadStoreAddress(VPI))
            return false;

          // Scalarize loads used as addresses, matching the legacy CM. The load
          // is single-scalar if the pointer is loop-invariant, otherwise it is
          // replicated per-lane. No mask is needed as the load is not
          // predicated.
          VPValue *Ptr = VPI->getOperand(0);
          const SCEV *PtrSCEV =
              vputils::getSCEVExprForVPValue(Ptr, CostCtx.PSE, CostCtx.L);
          bool IsSingleScalarLoad =
              !isa<SCEVCouldNotCompute>(PtrSCEV) &&
              CostCtx.PSE.getSE()->isLoopInvariant(PtrSCEV, CostCtx.L);

          ReplaceWith(VPI,
                      VPBuilder(VPI).insert(new VPReplicateRecipe(
                          I, Ptr, /*IsSingleScalar=*/IsSingleScalarLoad,
                          /*Mask=*/nullptr, *VPI, *VPI, VPI->getDebugLoc())));
          return true;
        });
  }

  // Widen unit-stride consecutive accesses, matching the legacy CM. Both
  // forward (stride +1) and reverse (stride -1) accesses are handled.
  VPlanTransforms::runPass(
      "widenConsecutiveMemOps", ProcessSubset, Plan, [&](VPInstruction *VPI) {
        Instruction *I = VPI->getUnderlyingInstr();
        bool IsLoad = VPI->getOpcode() == Instruction::Load;
        VPValue *Ptr = VPI->getOperand(!IsLoad);
        Type *ScalarTy =
            IsLoad ? VPI->getScalarType() : VPI->getOperand(0)->getScalarType();
        std::optional<int64_t> Stride =
            getConstantStride(Ptr, ScalarTy, CostCtx.PSE, CostCtx.L);
        if (Stride != 1 && Stride != -1)
          return false;
        bool Reverse = Stride == -1;

        // A predicated access can only be widened (rather than scalarized) if
        // the target supports a masked load/store for it.
        // TODO: Determine if a load/store needs predication directly in VPlan.
        bool IsPredicated = RecipeBuilder.isPredicatedInst(I);
        if (IsPredicated && !CostCtx.Config.isLegalMaskedLoadOrStore(
                                IsLoad, ScalarTy, getLoadStoreAlignment(I),
                                getLoadStoreAddressSpace(I)))
          return false;

        VPBuilder Builder(VPI);
        VPSingleDefRecipe *VectorPtr = Builder.createConsecutiveVectorPointer(
            Ptr, ScalarTy, Reverse, VPI->getDebugLoc());

        VPValue *Mask = IsPredicated ? VPI->getMask() : nullptr;
        // Reverse the mask so it matches the reversed access order.
        if (Reverse && Mask)
          Mask = Builder.createNaryOp(VPInstruction::Reverse, Mask,
                                      VPI->getDebugLoc());

        if (IsLoad) {
          VPSingleDefRecipe *Load = Builder.createWidenLoad(
              *cast<LoadInst>(I), VectorPtr, Mask,
              /*Consecutive=*/true, *VPI, VPI->getDebugLoc());
          // Reverse the loaded values back into program order.
          if (Reverse)
            Load = Builder.createNaryOp(VPInstruction::Reverse, Load,
                                        VPI->getDebugLoc());
          return ReplaceWith(VPI, Load);
        }

        VPValue *StoredVal = VPI->getOperand(0);
        if (Reverse)
          // Reverse the stored values so they are written in descending order.
          StoredVal = Builder.createNaryOp(VPInstruction::Reverse, StoredVal,
                                           VPI->getDebugLoc());

        auto *StoreR = Builder.createWidenStore(
            *cast<StoreInst>(I), VectorPtr, StoredVal, Mask,
            /*Consecutive=*/true, *VPI, VPI->getDebugLoc());
        return ReplaceWith(VPI, StoreR);
      });

  VPlanTransforms::runPass("delegateMemOpWideningToLegacyCM", ProcessSubset,
                           Plan, [&](VPInstruction *VPI) {
                             if (VPRecipeBase *Recipe =
                                     RecipeBuilder.tryToWidenMemory(VPI, Range))
                               return ReplaceWith(VPI, Recipe);

                             return Scalarize(VPI);
                           });
}

void VPlanTransforms::makeScalarizationDecisions(VPlan &Plan, VFRange &Range) {
  if (LoopVectorizationPlanner::getDecisionAndClampRange(
          [&](ElementCount VF) { return VF.isScalar(); }, Range))
    return;

  PostOrderTraversal<VPBlockDeepTraversalWrapper<VPBlockBase *>> POT(
      Plan.getEntry());
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(POT)) {
    for (VPRecipeBase &R : make_early_inc_range(reverse(*VPBB))) {
      auto *VPI = dyn_cast<VPInstruction>(&R);
      if (!VPI)
        continue;

      auto *I = cast_or_null<Instruction>(VPI->getUnderlyingValue());
      // Wouldn't be able to create a `VPReplicateRecipe` anyway.
      if (!I)
        continue;

      // If executing other lanes produces side-effects we can't avoid them.
      if (VPI->mayHaveSideEffects())
        continue;

      // We want to drop the mask operand, verify we can safely do that.
      if (VPI->isMasked() && !VPI->isSafeToSpeculativelyExecute())
        continue;

      // Avoid rewriting IV increment as that interferes with
      // `removeRedundantCanonicalIVs`.
      if (VPI->getOpcode() == Instruction::Add &&
          any_of(VPI->operands(), IsaPred<VPWidenIntOrFpInductionRecipe>))
        continue;

      // Other lanes are needed - can't drop them.
      if (!vputils::onlyFirstLaneUsed(VPI))
        continue;

      auto *Recipe = VPBuilder::createSingleScalarOp(
          VPI->getOpcode(), VPI->operandsWithoutMask(), /*Mask=*/nullptr, *VPI,
          *VPI, VPI->getDebugLoc(), I);
      Recipe->insertBefore(VPI);
      VPI->replaceAllUsesWith(Recipe);
      VPI->eraseFromParent();
    }
  }
}

/// Returns true if \p Info's parameter kinds are compatible with \p Args.
static bool areVFParamsOk(const VFInfo &Info, ArrayRef<VPValue *> Args,
                          PredicatedScalarEvolution &PSE, const Loop *L) {
  ScalarEvolution *SE = PSE.getSE();
  return all_of(Info.Shape.Parameters, [&](VFParameter Param) {
    switch (Param.ParamKind) {
    case VFParamKind::Vector:
    case VFParamKind::GlobalPredicate:
      return true;
    case VFParamKind::OMP_Uniform:
      return SE->isSCEVable(Args[Param.ParamPos]->getScalarType()) &&
             SE->isLoopInvariant(
                 vputils::getSCEVExprForVPValue(Args[Param.ParamPos], PSE, L),
                 L);
    case VFParamKind::OMP_Linear:
      return match(vputils::getSCEVExprForVPValue(Args[Param.ParamPos], PSE, L),
                   m_scev_AffineAddRec(
                       m_SCEV(), m_scev_SpecificSInt(Param.LinearStepOrPos),
                       m_SpecificLoop(L)));
    default:
      return false;
    }
  });
}

/// Find a vector variant of \p CI for \p VF, respecting \p MaskRequired.
/// Returns the variant function, or nullptr. Masked variants are assumed to
/// take the mask as a trailing parameter.
static Function *findVectorVariant(CallInst *CI, ArrayRef<VPValue *> Args,
                                   ElementCount VF, bool MaskRequired,
                                   PredicatedScalarEvolution &PSE,
                                   const Loop *L) {
  if (CI->isNoBuiltin())
    return nullptr;
  auto Mappings = VFDatabase::getMappings(*CI);
  const auto *It = find_if(Mappings, [&](const VFInfo &Info) {
    return Info.Shape.VF == VF && (!MaskRequired || Info.isMasked()) &&
           areVFParamsOk(Info, Args, PSE, L);
  });
  if (It == Mappings.end())
    return nullptr;
  return CI->getModule()->getFunction(It->VectorName);
}

namespace {
/// The outcome of choosing how to widen a call at a given VF.
struct CallWideningDecision {
  enum class KindTy { Scalarize, Intrinsic, VectorVariant };
  CallWideningDecision(KindTy Kind, Function *Variant = nullptr)
      : Kind(Kind), Variant(Variant) {}
  KindTy Kind;

  /// Set when Kind == VectorVariant.
  Function *Variant;

  bool operator==(const CallWideningDecision &Other) const {
    return Kind == Other.Kind && Variant == Other.Variant;
  }
};
} // namespace

/// Pick the cheapest widening for the call \p VPI at \p VF among scalarization,
/// vector intrinsic, and vector library variant.
static CallWideningDecision decideCallWidening(VPInstruction &VPI,
                                               ArrayRef<VPValue *> Ops,
                                               ElementCount VF,
                                               VPCostContext &CostCtx) {
  auto *CI = cast<CallInst>(VPI.getUnderlyingInstr());

  // Scalar VFs and calls forced or known to scalarize always replicate.
  if (VF.isScalar() || CostCtx.willBeScalarized(CI, VF))
    return CallWideningDecision::KindTy::Scalarize;

  auto *CalledFn = cast<Function>(
      VPI.getOperand(VPI.getNumOperandsWithoutMask() - 1)->getLiveInIRValue());
  Type *ResultTy = VPI.getScalarType();
  Intrinsic::ID ID = getVectorIntrinsicIDForCall(CI, &CostCtx.TLI);
  bool MaskRequired = CostCtx.isMaskRequired(CI);

  // Pseudo intrinsics (assume, lifetime, ...) are always scalarized.
  if (ID && VPCostContext::isFreeScalarIntrinsic(ID))
    return CallWideningDecision::KindTy::Scalarize;

  InstructionCost ScalarCost =
      VPReplicateRecipe::computeCallCost(CalledFn, ResultTy, Ops,
                                         /*IsSingleScalar=*/false, VF, CostCtx);

  Function *VecFunc =
      findVectorVariant(CI, Ops, VF, MaskRequired, CostCtx.PSE, CostCtx.L);
  InstructionCost VecCallCost = InstructionCost::getInvalid();
  if (VecFunc)
    VecCallCost = VPWidenCallRecipe::computeCallCost(VecFunc, CostCtx);

  // Prefer the intrinsic if it is at least as cheap as scalarizing and any
  // available vector variant.
  if (ID) {
    InstructionCost IntrinsicCost =
        VPWidenIntrinsicRecipe::computeCallCost(ID, Ops, VPI, VF, CostCtx);
    if (IntrinsicCost.isValid() && ScalarCost >= IntrinsicCost &&
        (!VecFunc || VecCallCost >= IntrinsicCost))
      return CallWideningDecision::KindTy::Intrinsic;
  }

  // Otherwise, use a vector library variant when it beats scalarizing.
  if (VecFunc && ScalarCost >= VecCallCost)
    return {CallWideningDecision::KindTy::VectorVariant, VecFunc};

  return CallWideningDecision::KindTy::Scalarize;
}

void VPlanTransforms::makeCallWideningDecisions(VPlan &Plan, VFRange &Range,
                                                VPRecipeBuilder &RecipeBuilder,
                                                VPCostContext &CostCtx) {
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksAs<VPBasicBlock>(
           vp_depth_first_shallow(Plan.getVectorLoopRegion()->getEntry()))) {
    for (VPRecipeBase &R : make_early_inc_range(*VPBB)) {
      auto *VPI = dyn_cast<VPInstruction>(&R);
      if (!VPI || !VPI->getUnderlyingValue() ||
          VPI->getOpcode() != Instruction::Call)
        continue;

      auto *CI = cast<CallInst>(VPI->getUnderlyingInstr());
      SmallVector<VPValue *, 4> Ops(VPI->op_begin(),
                                    VPI->op_begin() + CI->arg_size());

      CallWideningDecision Decision =
          decideCallWidening(*VPI, Ops, Range.Start, CostCtx);
      LoopVectorizationPlanner::getDecisionAndClampRange(
          [&](ElementCount VF) {
            return Decision == decideCallWidening(*VPI, Ops, VF, CostCtx);
          },
          Range);

      VPSingleDefRecipe *Replacement = nullptr;
      switch (Decision.Kind) {
      case CallWideningDecision::KindTy::Intrinsic: {
        Intrinsic::ID ID = getVectorIntrinsicIDForCall(CI, &CostCtx.TLI);
        Type *ResultTy = VPI->getScalarType();
        Replacement = new VPWidenIntrinsicRecipe(*CI, ID, Ops, ResultTy, *VPI,
                                                 *VPI, VPI->getDebugLoc());
        break;
      }
      case CallWideningDecision::KindTy::VectorVariant: {
        // Masked variants take the mask as a trailing parameter, so they have
        // one more parameter than the original call's arguments.
        if (Decision.Variant->arg_size() > Ops.size()) {
          VPValue *Mask = VPI->isMasked() ? VPI->getMask() : Plan.getTrue();
          Ops.push_back(Mask);
        }
        Ops.push_back(VPI->getOperand(VPI->getNumOperandsWithoutMask() - 1));
        Replacement = new VPWidenCallRecipe(CI, Decision.Variant, Ops, *VPI,
                                            *VPI, VPI->getDebugLoc());
        break;
      }
      case CallWideningDecision::KindTy::Scalarize:
        Replacement = RecipeBuilder.handleReplication(VPI, Range);
        break;
      }

      Replacement->insertBefore(VPI);
      VPI->replaceAllUsesWith(Replacement);
      VPI->eraseFromParent();
    }
  }
}

void VPlanTransforms::convertToStridedAccesses(VPlan &Plan,
                                               PredicatedScalarEvolution &PSE,
                                               Loop &L, VPCostContext &Ctx,
                                               VFRange &Range) {
  if (Plan.hasScalarVFOnly())
    return;

  VPRegionBlock *VectorLoop = Plan.getVectorLoopRegion();
  VPValue *I32VF = nullptr;
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(
           vp_depth_first_shallow(VectorLoop->getEntry()))) {
    for (VPRecipeBase &R : make_early_inc_range(*VPBB)) {
      auto *LoadR = dyn_cast<VPWidenLoadRecipe>(&R);
      // TODO: Support strided store.
      // TODO: Transform reverse access into strided access with -1 stride.
      // TODO: Transform gather/scatter with uniform address into strided access
      // with 0 stride.
      // TODO: Transform interleave access into multiple strided accesses.
      if (!LoadR || LoadR->isConsecutive())
        continue;

      VPValue *Ptr = LoadR->getAddr();
      // Check if this is a strided access by analyzing the address SCEV for an
      // affine addRec.
      const SCEV *PtrSCEV = vputils::getSCEVExprForVPValue(Ptr, PSE, &L);
      const SCEV *Start;
      const SCEVConstant *Step;
      // TODO: Support non-constant loop invariant stride.
      if (!match(PtrSCEV,
                 m_scev_AffineAddRec(m_SCEV(Start), m_SCEVConstant(Step),
                                     m_SpecificLoop(&L))))
        continue;

      Type *LoadTy = LoadR->getScalarType();
      Align Alignment = LoadR->getAlign();
      auto IsProfitable = [&](ElementCount VF) {
        Type *DataTy = toVectorTy(LoadTy, VF);
        if (!Ctx.TTI.isLegalStridedLoadStore(DataTy, Alignment))
          return false;
        const InstructionCost CurrentCost = LoadR->computeCost(VF, Ctx);
        const InstructionCost StridedLoadStoreCost =
            VPWidenMemIntrinsicRecipe::computeMemIntrinsicCost(
                Intrinsic::experimental_vp_strided_load, DataTy,
                LoadR->isMasked(), Alignment, Ctx);
        return StridedLoadStoreCost < CurrentCost;
      };

      if (!LoopVectorizationPlanner::getDecisionAndClampRange(IsProfitable,
                                                              Range))
        continue;

      // Invalidate the legacy widening decision so the cost of replaced load is
      // not counted during precomputeCosts.
      // TODO: Remove once the legacy exit cost computation is retired.
      for (ElementCount VF : Range)
        Ctx.invalidateWideningDecision(&LoadR->getIngredient(), VF);

      // Get VF as i32 for the vector length operand.
      if (!I32VF) {
        VPBuilder Builder(Plan.getVectorPreheader());
        I32VF = Builder.createScalarZExtOrTrunc(
            &Plan.getVF(), Type::getInt32Ty(Plan.getContext()),
            DebugLoc::getUnknown());
      }

      VPBuilder Builder(LoadR);
      // Create the base pointer of strided access.
      // TODO: reuse VPDerivedIVRecipe for base pointer computation when it
      // supports a general VPValue as the start value.
      VPValue *StartVPV =
          VPSCEVExpander(Builder, *PSE.getSE(), LoadR->getDebugLoc())
              .tryToExpand(Start);
      if (!StartVPV)
        StartVPV = VPBuilder(Plan.getEntry()).createExpandSCEV(Start);
      VPValue *StrideInBytes = Plan.getOrAddLiveIn(Step->getValue());
      Type *IndexTy = Plan.getDataLayout().getIndexType(Ptr->getScalarType());
      assert(IndexTy == StrideInBytes->getScalarType() &&
             "Stride type from SCEV must match the index type");
      VPValue *CanIV = Builder.createScalarZExtOrTrunc(
          VectorLoop->getCanonicalIV(), IndexTy, DebugLoc::getUnknown());
      auto *AddRecPtr = cast<SCEVAddRecExpr>(PtrSCEV);
      auto *Offset = Builder.createOverflowingOp(
          Instruction::Mul, {CanIV, StrideInBytes},
          {AddRecPtr->hasNoUnsignedWrap(), /*HasNSW=*/false});
      GEPNoWrapFlags NWFlags = AddRecPtr->hasNoUnsignedWrap()
                                   ? GEPNoWrapFlags::noUnsignedWrap()
                                   : GEPNoWrapFlags::none();
      VPValue *BasePtr = Builder.createNoWrapPtrAdd(StartVPV, Offset, NWFlags);

      // Create a new vector pointer for strided access.
      VPValue *NewPtr = Builder.createVectorPointer(
          BasePtr, Type::getInt8Ty(Plan.getContext()), StrideInBytes, NWFlags,
          LoadR->getDebugLoc());

      VPValue *Mask = LoadR->getMask();
      if (!Mask)
        Mask = Plan.getTrue();
      auto *StridedLoad = Builder.createWidenMemIntrinsic(
          Intrinsic::experimental_vp_strided_load,
          {NewPtr, StrideInBytes, Mask, I32VF}, LoadTy, Alignment, *LoadR,
          LoadR->getDebugLoc());
      LoadR->replaceAllUsesWith(StridedLoad);
    }
  }
}

static std::optional<Instruction::BinaryOps>
getUnmaskedDivRemOpcode(Intrinsic::ID ID) {
  switch (ID) {
  case Intrinsic::masked_udiv:
    return Instruction::UDiv;
  case Intrinsic::masked_sdiv:
    return Instruction::SDiv;
  case Intrinsic::masked_urem:
    return Instruction::URem;
  case Intrinsic::masked_srem:
    return Instruction::SRem;
  default:
    return {};
  }
}

void VPlanTransforms::narrowToSingleScalarRecipes(VPlan &Plan) {
  if (Plan.hasScalarVFOnly())
    return;

  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(
           vp_depth_first_deep(Plan.getEntry()))) {
    for (VPRecipeBase &R : make_early_inc_range(reverse(*VPBB))) {
      if (!isa<VPWidenRecipe, VPWidenGEPRecipe, VPReplicateRecipe,
               VPWidenIntrinsicRecipe>(&R))
        continue;
      auto *RepR = dyn_cast<VPReplicateRecipe>(&R);
      if (RepR && (RepR->isSingleScalar() || RepR->isPredicated()))
        continue;

      auto *RepOrWidenR = cast<VPRecipeWithIRFlags>(&R);
      if (RepR && RepR->getOpcode() == Instruction::Store &&
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

      // Narrow llvm.masked.{u,s}{div,rem} intrinsics with a safe divisor.
      if (auto *IntrR = dyn_cast<VPWidenIntrinsicRecipe>(RepOrWidenR)) {
        if (!vputils::onlyFirstLaneUsed(IntrR))
          continue;
        auto Opc = getUnmaskedDivRemOpcode(IntrR->getVectorIntrinsicID());
        if (!Opc)
          continue;
        VPBuilder Builder(IntrR);
        VPValue *SafeDivisor = Builder.createSelect(
            IntrR->getOperand(2), IntrR->getOperand(1),
            Plan.getConstantInt(IntrR->getScalarType(), 1));
        VPValue *Clone = Builder.createNaryOp(
            *Opc, {IntrR->getOperand(0), SafeDivisor},
            VPIRFlags::getDefaultFlags(*Opc), IntrR->getDebugLoc());
        IntrR->replaceAllUsesWith(Clone);
        IntrR->eraseFromParent();
        continue;
      }

      // Skip recipes that aren't single scalars.
      if (!vputils::isSingleScalar(RepOrWidenR))
        continue;

      // Predicate to check if a user of Op introduces extra broadcasts.
      auto IntroducesBCastOf = [](const VPValue *Op) {
        return [Op](const VPUser *U) {
          if (auto *VPI = dyn_cast<VPInstruction>(U)) {
            if (is_contained({VPInstruction::ExtractLastLane,
                              VPInstruction::ExtractLastPart,
                              VPInstruction::ExtractPenultimateElement},
                             VPI->getOpcode()))
              return false;
          }
          return !U->usesScalars(Op);
        };
      };

      if (any_of(RepOrWidenR->users(), IntroducesBCastOf(RepOrWidenR)) &&
          none_of(RepOrWidenR->operands(), [&](VPValue *Op) {
            if (any_of(
                    make_filter_range(Op->users(), not_equal_to(RepOrWidenR)),
                    IntroducesBCastOf(Op)))
              return false;
            // Non-constant live-ins require broadcasts, while constants do not
            // need explicit broadcasts.
            bool LiveInNeedsBroadcast =
                isa<VPIRValue>(Op) && !isa<VPConstant>(Op);
            auto *OpR = dyn_cast<VPReplicateRecipe>(Op);
            return LiveInNeedsBroadcast || (OpR && OpR->isSingleScalar());
          }))
        continue;

      auto *Clone = VPBuilder::createSingleScalarOp(
          vputils::getOpcode(RepOrWidenR), RepOrWidenR->operands(),
          /*Mask=*/nullptr, *RepOrWidenR, {}, DebugLoc::getUnknown(),
          RepOrWidenR->getUnderlyingInstr());
      Clone->insertBefore(RepOrWidenR);
      RepOrWidenR->replaceAllUsesWith(Clone);
      if (vputils::isDeadRecipe(*RepOrWidenR))
        RepOrWidenR->eraseFromParent();
    }
  }
}
