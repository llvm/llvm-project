//===- VPlanEVLTailFolding.cpp - EVL tail folding transforms --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the VPlan-to-VPlan transforms related to explicit
/// vector length (EVL) tail folding support.
///
//===----------------------------------------------------------------------===//

#include "LoopVectorizationPlanner.h"
#include "VPlan.h"
#include "VPlanCFG.h"
#include "VPlanHelpers.h"
#include "VPlanPatternMatch.h"
#include "VPlanTransforms.h"
#include "VPlanUtils.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/Intrinsics.h"

using namespace llvm;
using namespace VPlanPatternMatch;

/// From the definition of llvm.experimental.get.vector.length,
/// VPInstruction::ExplicitVectorLength(%AVL) = %AVL when %AVL <= VF.
bool VPlanTransforms::simplifyKnownEVL(VPlan &Plan, ElementCount VF,
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
      if (Trunc != AVL) {
        auto *TruncR = cast<VPSingleDefRecipe>(Trunc);
        const DataLayout &DL = Plan.getDataLayout();
        if (VPValue *Folded =
                vputils::tryToFoldLiveIns(*TruncR, TruncR->operands(), DL))
          Trunc = Folded;
      }
      R.getVPSingleValue()->replaceAllUsesWith(Trunc);
      return true;
    }
  }
  return false;
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

static std::optional<Intrinsic::ID> getVPDivRemIntrinsic(Intrinsic::ID IntrID) {
  switch (IntrID) {
  case Intrinsic::masked_udiv:
    return Intrinsic::vp_udiv;
  case Intrinsic::masked_sdiv:
    return Intrinsic::vp_sdiv;
  case Intrinsic::masked_urem:
    return Intrinsic::vp_urem;
  case Intrinsic::masked_srem:
    return Intrinsic::vp_srem;
  default:
    return std::nullopt;
  }
}

/// Try to optimize a \p CurRecipe masked by \p HeaderMask to a corresponding
/// EVL-based recipe without the header mask. Returns nullptr if no EVL-based
/// recipe could be created.
/// \p HeaderMask  Header Mask.
/// \p CurRecipe   Recipe to be transform.
/// \p EVL         The explicit vector length parameter of vector-predication
/// intrinsics.
static VPRecipeBase *optimizeMaskToEVL(VPValue *HeaderMask,
                                       VPRecipeBase &CurRecipe, VPValue &EVL) {
  VPlan *Plan = CurRecipe.getParent()->getPlan();
  DebugLoc DL = CurRecipe.getDebugLoc();
  VPValue *Addr, *Mask, *EndPtr;

  /// Adjust any end pointers so that they point to the end of EVL lanes not VF.
  auto AdjustEndPtr = [&CurRecipe, &EVL](VPValue *EndPtr) {
    auto *EVLEndPtr = cast<VPVectorEndPointerRecipe>(EndPtr)->clone();
    EVLEndPtr->insertBefore(&CurRecipe);
    // Cast EVL (i32) to match the VF operand's type.
    VPValue *EVLAsVF = VPBuilder(EVLEndPtr).createScalarZExtOrTrunc(
        &EVL, EVLEndPtr->getOperand(1)->getScalarType(), EVL.getScalarType(),
        DebugLoc::getUnknown());
    EVLEndPtr->setOperand(1, EVLAsVF);
    return EVLEndPtr;
  };

  auto GetVPReverse = [&CurRecipe, &EVL, Plan,
                       DL](VPValue *V) -> VPWidenIntrinsicRecipe * {
    if (!V)
      return nullptr;
    auto *Reverse = new VPWidenIntrinsicRecipe(
        Intrinsic::experimental_vp_reverse, {V, Plan->getTrue(), &EVL},
        V->getScalarType(), {}, {}, DL);
    Reverse->insertBefore(&CurRecipe);
    return Reverse;
  };

  if (match(&CurRecipe,
            m_MaskedLoad(m_VPValue(Addr), m_RemoveMask(HeaderMask, Mask))))
    return new VPWidenLoadEVLRecipe(cast<VPWidenLoadRecipe>(CurRecipe), Addr,
                                    EVL, Mask);

  if (match(&CurRecipe,
            m_MaskedLoad(m_VPValue(EndPtr),
                         m_Reverse(m_RemoveMask(HeaderMask, Mask)))) &&
      match(EndPtr, m_VecEndPtr(m_VPValue(), m_Specific(&Plan->getVF())))) {
    Mask = GetVPReverse(Mask);
    Addr = AdjustEndPtr(EndPtr);
    auto *LoadR = new VPWidenLoadEVLRecipe(cast<VPWidenLoadRecipe>(CurRecipe),
                                           Addr, EVL, Mask);
    LoadR->insertBefore(&CurRecipe);
    VPValue *Poison = Plan->getPoison(LoadR->getScalarType());
    return new VPWidenIntrinsicRecipe(Intrinsic::vector_splice_left,
                                      {Poison, LoadR, &EVL},
                                      LoadR->getScalarType(), {}, {}, DL);
  }

  VPValue *Stride;
  if (match(&CurRecipe, m_Intrinsic<Intrinsic::experimental_vp_strided_load>(
                            m_VPValue(Addr), m_VPValue(Stride),
                            m_RemoveMask(HeaderMask, Mask),
                            m_TruncOrSelf(m_Specific(&Plan->getVF()))))) {
    if (!Mask)
      Mask = Plan->getTrue();
    auto *NewLoad = cast<VPWidenMemIntrinsicRecipe>(&CurRecipe)->clone();
    NewLoad->setOperand(2, Mask);
    NewLoad->setOperand(3, &EVL);
    return NewLoad;
  }

  VPValue *StoredVal;
  if (match(&CurRecipe, m_MaskedStore(m_VPValue(Addr), m_VPValue(StoredVal),
                                      m_RemoveMask(HeaderMask, Mask))))
    return new VPWidenStoreEVLRecipe(cast<VPWidenStoreRecipe>(CurRecipe), Addr,
                                     StoredVal, EVL, Mask);

  if (match(&CurRecipe,
            m_MaskedStore(m_VPValue(EndPtr), m_VPValue(StoredVal),
                          m_Reverse(m_RemoveMask(HeaderMask, Mask)))) &&
      match(EndPtr, m_VecEndPtr(m_VPValue(), m_Specific(&Plan->getVF())))) {
    Mask = GetVPReverse(Mask);
    Addr = AdjustEndPtr(EndPtr);
    VPValue *Poison = Plan->getPoison(StoredVal->getScalarType());
    auto *SpliceR = new VPWidenIntrinsicRecipe(
        Intrinsic::vector_splice_right, {StoredVal, Poison, &EVL},
        StoredVal->getScalarType(), {}, {}, DL);
    SpliceR->insertBefore(&CurRecipe);
    return new VPWidenStoreEVLRecipe(cast<VPWidenStoreRecipe>(CurRecipe), Addr,
                                     SpliceR, EVL, Mask);
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
  if (match(&CurRecipe, m_SelectLike(m_RemoveMask(HeaderMask, Mask),
                                     m_VPValue(LHS), m_VPValue(RHS))))
    return new VPWidenIntrinsicRecipe(
        Intrinsic::vp_merge, {Mask ? Mask : Plan->getTrue(), LHS, RHS, &EVL},
        LHS->getScalarType(), {}, {}, DL);

  if (match(&CurRecipe, m_LastActiveLane(m_Specific(HeaderMask)))) {
    Type *Ty = CurRecipe.getVPSingleValue()->getScalarType();
    VPValue *ZExt =
        VPBuilder(&CurRecipe)
            .createScalarZExtOrTrunc(&EVL, Ty, EVL.getScalarType(), DL);
    return new VPInstruction(
        Instruction::Sub, {ZExt, Plan->getConstantInt(Ty, 1)},
        VPIRFlags::getDefaultFlags(Instruction::Sub), {}, DL);
  }

  // lhs | (headermask && rhs) -> vp.merge rhs, true, lhs, evl
  if (match(&CurRecipe,
            m_c_BinaryOr(m_VPValue(LHS),
                         m_LogicalAnd(m_Specific(HeaderMask), m_VPValue(RHS)))))
    return new VPWidenIntrinsicRecipe(Intrinsic::vp_merge,
                                      {RHS, Plan->getTrue(), LHS, &EVL},
                                      LHS->getScalarType(), {}, {}, DL);

  if (auto *IntrR = dyn_cast<VPWidenIntrinsicRecipe>(&CurRecipe))
    if (auto VPID = getVPDivRemIntrinsic(IntrR->getVectorIntrinsicID()))
      if (match(IntrR->getOperand(2), m_RemoveMask(HeaderMask, Mask)))
        return new VPWidenIntrinsicRecipe(*VPID,
                                          {IntrR->getOperand(0),
                                           IntrR->getOperand(1),
                                           Mask ? Mask : Plan->getTrue(), &EVL},
                                          IntrR->getScalarType(), {}, {}, DL);

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

  SmallVector<VPRecipeBase *> OldRecipes;
  for (VPUser *U : vputils::collectUsersRecursively(HeaderMask)) {
    VPRecipeBase *R = cast<VPRecipeBase>(U);
    if (auto *NewR = optimizeMaskToEVL(HeaderMask, *R, *EVL)) {
      NewR->insertBefore(R);
      for (auto [Old, New] :
           zip_equal(R->definedValues(), NewR->definedValues()))
        Old->replaceAllUsesWith(New);
      OldRecipes.push_back(R);
    }
  }

  // Replace remaining (HeaderMask && Mask) with vp.merge (True, Mask,
  // False, EVL)
  for (VPUser *U : vputils::collectUsersRecursively(HeaderMask)) {
    VPValue *Mask;
    if (match(U, m_LogicalAnd(m_Specific(HeaderMask), m_VPValue(Mask)))) {
      auto *LogicalAnd = cast<VPInstruction>(U);
      auto *Merge = new VPWidenIntrinsicRecipe(
          Intrinsic::vp_merge, {Plan.getTrue(), Mask, Plan.getFalse(), EVL},
          Mask->getScalarType(), {}, {}, LogicalAnd->getDebugLoc());
      Merge->insertBefore(LogicalAnd);
      LogicalAnd->replaceAllUsesWith(Merge);
      OldRecipes.push_back(LogicalAnd);
    }
  }

  // Pull out left splices from any elementwise op.
  // binop(splice.left(poison, x, evl), live-in)
  // -> splice.left(poison, binop(x,live-in), evl)
  vputils::pullOutPermutations(
      Plan,
      [&EVL](VPValue *&X) {
        return m_Intrinsic<Intrinsic::vector_splice_left>(
            m_Poison(), m_VPValue(X), m_Specific(EVL));
      },
      [&Plan, &EVL](auto *X) {
        return new VPWidenIntrinsicRecipe(
            Intrinsic::vector_splice_left,
            {Plan.getPoison(X->getScalarType()), X, EVL}, X->getScalarType(),
            {}, {}, X->getDebugLoc());
      });

  // Fold the following splice patterns:
  //   splice.right(splice.left(poison, x, evl), poison, evl) -> x
  //   vector.reverse(splice.left(poison, x, evl))  -> vp.reverse(x, true, evl)
  //   splice.right(vector.reverse(x), poison, evl) -> vp.reverse(x, true, evl)
  for (VPUser *U : vputils::collectUsersRecursively(EVL)) {
    auto *R = cast<VPRecipeBase>(U);
    // Remove potentially dead left splices from the transform above.
    if (match(U, m_Intrinsic<Intrinsic::vector_splice_left>()) &&
        R->getVPSingleValue()->getNumUsers() == 0) {
      OldRecipes.push_back(R);
      continue;
    }

    VPValue *X;
    if (match(U, m_Intrinsic<Intrinsic::vector_splice_right>(
                     m_Intrinsic<Intrinsic::vector_splice_left>(
                         m_Poison(), m_VPValue(X), m_Specific(EVL)),
                     m_Poison(), m_Specific(EVL)))) {
      R->getVPSingleValue()->replaceAllUsesWith(X);
      OldRecipes.push_back(R);
      continue;
    }

    if (!match(U,
               m_CombineOr(
                   m_Reverse(m_Intrinsic<Intrinsic::vector_splice_left>(
                       m_Poison(), m_VPValue(X), m_Specific(EVL))),
                   m_Intrinsic<Intrinsic::vector_splice_right>(
                       m_Reverse(m_VPValue(X)), m_Poison(), m_Specific(EVL)))))
      continue;

    auto *VPReverse = new VPWidenIntrinsicRecipe(
        Intrinsic::experimental_vp_reverse, {X, Plan.getTrue(), EVL},
        X->getScalarType(), {}, {}, R->getDebugLoc());
    VPReverse->insertBefore(R);
    R->getVPSingleValue()->replaceAllUsesWith(VPReverse);
    OldRecipes.push_back(R);
  }

  for (VPRecipeBase *R : reverse(OldRecipes)) {
    SmallVector<VPValue *> PossiblyDead(R->operands());
    R->eraseFromParent();
    for (VPValue *Op : PossiblyDead)
      vputils::recursivelyDeleteDeadRecipes(Op);
  }
}

/// After replacing the canonical IV with a EVL-based IV, fixup recipes that use
/// VF to use the EVL instead to avoid incorrect updates on the penultimate
/// iteration.
static void fixupVFUsersForEVL(VPlan &Plan, VPValue &EVL) {
  VPRegionBlock *LoopRegion = Plan.getVectorLoopRegion();
  VPBasicBlock *Header = LoopRegion->getEntryBasicBlock();

  // EVL is i32 but VF/VFxUF are IdxTy. Convert as needed.
  VPValue *EVLAsIdx =
      VPBuilder::getToInsertAfter(EVL.getDefiningRecipe())
          .createScalarZExtOrTrunc(&EVL, Plan.getVF().getScalarType(),
                                   EVL.getScalarType(), DebugLoc::getUnknown());

  assert(all_of(Plan.getVF().users(),
                [&Plan](VPUser *U) {
                  auto IsAllowedUser =
                      IsaPred<VPVectorEndPointerRecipe, VPScalarIVStepsRecipe,
                              VPWidenIntOrFpInductionRecipe,
                              VPWidenMemIntrinsicRecipe>;
                  if (match(U, m_Trunc(m_Specific(&Plan.getVF()))))
                    return all_of(cast<VPSingleDefRecipe>(U)->users(),
                                  IsAllowedUser);
                  return IsAllowedUser(U);
                }) &&
         "User of VF that we can't transform to EVL.");
  Plan.getVF().replaceUsesWithIf(EVLAsIdx, [](VPUser &U, unsigned Idx) {
    return isa<VPWidenIntOrFpInductionRecipe, VPScalarIVStepsRecipe>(U);
  });

  assert(all_of(Plan.getVFxUF().users(),
                match_fn(m_CombineOr(
                    m_c_Add(m_Specific(LoopRegion->getCanonicalIV()),
                            m_Specific(&Plan.getVFxUF())),
                    m_Isa<VPWidenPointerInductionRecipe>()))) &&
         "Only users of VFxUF should be VPWidenPointerInductionRecipe and the "
         "increment of the canonical induction.");
  Plan.getVFxUF().replaceUsesWithIf(EVLAsIdx, [](VPUser &U, unsigned Idx) {
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
        MaxEVL, Type::getInt32Ty(Plan.getContext()), MaxEVL->getScalarType(),
        DebugLoc::getUnknown());

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
            R.getVPSingleValue()->getScalarType(), {}, {}, R.getDebugLoc());
        VPSplice->insertBefore(&R);
        R.getVPSingleValue()->replaceAllUsesWith(VPSplice);
      }
    }
  }

  VPValue *HeaderMask = LoopRegion->getHeaderMask();
  if (!HeaderMask)
    return;

  // Ensure that any reduction that uses a select to mask off tail lanes does so
  // in the vector loop, not the middle block, since EVL tail folding can have
  // tail elements in the penultimate iteration.
  assert(all_of(*Plan.getMiddleBlock(), [&Plan, HeaderMask](VPRecipeBase &R) {
    if (match(&R, m_ComputeReductionResult(m_Select(m_Specific(HeaderMask),
                                                    m_VPValue(), m_VPValue()))))
      return R.getOperand(0)->getDefiningRecipe()->getRegion() ==
             Plan.getVectorLoopRegion();
    return true;
  }));

  // Replace the abstract header mask with a mask equivalent to predicating by
  // EVL: icmp ult step-vector, EVL
  VPRecipeBase *EVLR = EVL.getDefiningRecipe();
  VPBuilder Builder(EVLR->getParent(), std::next(EVLR->getIterator()));
  Type *EVLType = EVL.getScalarType();
  VPValue *EVLMask = Builder.createICmp(
      CmpInst::ICMP_ULT,
      Builder.createNaryOp(VPInstruction::StepVector, {}, EVLType), &EVL);
  HeaderMask->replaceAllUsesWith(EVLMask);
}

/// Converts a tail folded vector loop region to step by
/// VPInstruction::ExplicitVectorLength elements instead of VF elements each
/// iteration.
///
/// - Add a VPCurrentIterationPHIRecipe and related recipes to \p Plan and
///   replaces all uses of the canonical IV except for the canonical IV
///   increment with a VPCurrentIterationPHIRecipe. The canonical IV is used
///   only for loop iterations counting after this transformation.
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
/// %CurrentIter = CURRENT-ITERATION-PHI [ %StartV, %vector.ph ],
///                                      [ %NextIter, %vector.body ]
/// %AVL = phi [ trip-count, %vector.ph ], [ %NextAVL, %vector.body ]
/// %VPEVL = EXPLICIT-VECTOR-LENGTH %AVL
/// ...
/// %OpEVL = cast i32 %VPEVL to IVSize
/// %NextIter = add IVSize %OpEVL, %CurrentIter
/// %NextAVL = sub IVSize nuw %AVL, %OpEVL
/// ...
///
/// If MaxSafeElements is provided, the function adds the following recipes:
/// vector.ph:
/// ...
///
/// vector.body:
/// ...
/// %CurrentIter = CURRENT-ITERATION-PHI [ %StartV, %vector.ph ],
///                                      [ %NextIter, %vector.body ]
/// %AVL = phi [ trip-count, %vector.ph ], [ %NextAVL, %vector.body ]
/// %cmp = cmp ult %AVL, MaxSafeElements
/// %SAFE_AVL = select %cmp, %AVL, MaxSafeElements
/// %VPEVL = EXPLICIT-VECTOR-LENGTH %SAFE_AVL
/// ...
/// %OpEVL = cast i32 %VPEVL to IVSize
/// %NextIter = add IVSize %OpEVL, %CurrentIter
/// %NextAVL = sub IVSize nuw %AVL, %OpEVL
/// ...
///
void VPlanTransforms::addExplicitVectorLength(
    VPlan &Plan, const std::optional<unsigned> &MaxSafeElements) {
  if (Plan.hasScalarVFOnly())
    return;
  VPRegionBlock *LoopRegion = Plan.getVectorLoopRegion();
  VPBasicBlock *Header = LoopRegion->getEntryBasicBlock();

  auto *CanonicalIV = LoopRegion->getCanonicalIV();
  auto *CanIVTy = LoopRegion->getCanonicalIVType();
  VPValue *StartV = Plan.getZero(CanIVTy);
  auto *CanonicalIVIncrement = LoopRegion->getOrCreateCanonicalIVIncrement();

  // Create the CurrentIteration recipe in the vector loop.
  auto *CurrentIteration =
      new VPCurrentIterationPHIRecipe(StartV, DebugLoc::getUnknown());
  CurrentIteration->insertBefore(*Header, Header->begin());
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

  Builder.setInsertPoint(CanonicalIVIncrement);
  VPValue *OpVPEVL = VPEVL;

  auto *I32Ty = Type::getInt32Ty(Plan.getContext());
  OpVPEVL = Builder.createScalarZExtOrTrunc(
      OpVPEVL, CanIVTy, I32Ty, CanonicalIVIncrement->getDebugLoc());

  auto *NextIter = Builder.createAdd(
      OpVPEVL, CurrentIteration, CanonicalIVIncrement->getDebugLoc(),
      "current.iteration.next", CanonicalIVIncrement->getNoWrapFlags());
  CurrentIteration->addBackedgeValue(NextIter);

  VPValue *NextAVL =
      Builder.createSub(AVLPhi, OpVPEVL, DebugLoc::getCompilerGenerated(),
                        "avl.next", {/*NUW=*/true, /*NSW=*/false});
  AVLPhi->addIncoming(NextAVL);

  fixupVFUsersForEVL(Plan, *VPEVL);
  removeDeadRecipes(Plan);

  // Replace all uses of the canonical IV with VPCurrentIterationPHIRecipe
  // except for the canonical IV increment.
  CanonicalIV->replaceUsesWithIf(CurrentIteration,
                                 [CanonicalIVIncrement](VPUser &U, unsigned) {
                                   return &U != CanonicalIVIncrement;
                                 });
  // TODO: support unroll factor > 1.
  Plan.setUF(1);
}

void VPlanTransforms::convertToVariableLengthStep(VPlan &Plan) {
  // Find the vector loop entry by locating VPCurrentIterationPHIRecipe.
  // There should be only one VPCurrentIteration in the entire plan.
  VPCurrentIterationPHIRecipe *CurrentIteration = nullptr;

  for (VPBasicBlock *VPBB : VPBlockUtils::blocksAs<VPBasicBlock>(
           vp_depth_first_shallow(Plan.getEntry())))
    for (VPRecipeBase &R : VPBB->phis())
      if (auto *PhiR = dyn_cast<VPCurrentIterationPHIRecipe>(&R)) {
        assert(!CurrentIteration &&
               "Found multiple CurrentIteration. Only one expected");
        CurrentIteration = PhiR;
      }

  // Early return if it is not variable-length stepping.
  if (!CurrentIteration)
    return;

  VPBasicBlock *HeaderVPBB = CurrentIteration->getParent();
  VPValue *CurrentIterationIncr = CurrentIteration->getBackedgeValue();

  // Convert CurrentIteration to concrete recipe.
  auto *ScalarR =
      VPBuilder(CurrentIteration)
          .createScalarPhi(
              {CurrentIteration->getStartValue(), CurrentIterationIncr},
              CurrentIteration->getDebugLoc(), "current.iteration.iv");
  CurrentIteration->replaceAllUsesWith(ScalarR);
  CurrentIteration->eraseFromParent();

  // Replace CanonicalIVInc with CurrentIteration increment if it exists.
  auto *CanonicalIV = cast<VPPhi>(&*HeaderVPBB->begin());
  if (auto *CanIVInc = findUserOf(
          CanonicalIV, m_c_Add(m_VPValue(), m_Specific(&Plan.getVFxUF())))) {
    cast<VPInstruction>(CanIVInc)->replaceAllUsesWith(CurrentIterationIncr);
    CanIVInc->eraseFromParent();
  }
}

void VPlanTransforms::convertEVLExitCond(VPlan &Plan) {
  VPRegionBlock *LoopRegion = Plan.getVectorLoopRegion();
  if (!LoopRegion)
    return;
  VPBasicBlock *Header = LoopRegion->getEntryBasicBlock();
  if (Header->empty())
    return;
  // The EVL IV is always at the beginning.
  auto *EVLPhi = dyn_cast<VPCurrentIterationPHIRecipe>(&Header->front());
  if (!EVLPhi)
    return;

  // Bail if not an EVL tail folded loop.
  VPValue *AVL;
  if (!match(EVLPhi->getBackedgeValue(),
             m_c_Add(m_ZExtOrSelf(m_EVL(m_VPValue(AVL))), m_Specific(EVLPhi))))
    return;

  // The AVL may be capped to a safe distance.
  VPValue *SafeAVL, *UnsafeAVL;
  if (match(AVL,
            m_Select(m_SpecificICmp(CmpInst::ICMP_ULT, m_VPValue(UnsafeAVL),
                                    m_VPValue(SafeAVL)),
                     m_Deferred(UnsafeAVL), m_Deferred(SafeAVL))))
    AVL = UnsafeAVL;

  VPValue *AVLNext;
  [[maybe_unused]] bool FoundAVLNext =
      match(AVL, m_VPInstruction<Instruction::PHI>(
                     m_Specific(Plan.getTripCount()), m_VPValue(AVLNext)));
  assert(FoundAVLNext && "Didn't find AVL backedge?");

  VPBasicBlock *Latch = LoopRegion->getExitingBasicBlock();
  auto *LatchBr = cast<VPInstruction>(Latch->getTerminator());
  if (match(LatchBr, m_BranchOnCond(m_True())))
    return;

  VPValue *CanIVInc;
  [[maybe_unused]] bool FoundIncrement = match(
      LatchBr,
      m_BranchOnCond(m_SpecificCmp(CmpInst::ICMP_EQ, m_VPValue(CanIVInc),
                                   m_Specific(&Plan.getVectorTripCount()))));
  assert(FoundIncrement &&
         match(CanIVInc, m_Add(m_Specific(LoopRegion->getCanonicalIV()),
                               m_Specific(&Plan.getVFxUF()))) &&
         "Expected BranchOnCond with ICmp comparing CanIV + VFxUF with vector "
         "trip count");

  Type *AVLTy = AVLNext->getScalarType();
  VPBuilder Builder(LatchBr);
  LatchBr->setOperand(
      0, Builder.createICmp(CmpInst::ICMP_EQ, AVLNext, Plan.getZero(AVLTy)));
}
