//===- SLPCostAnalysis.cpp - SLP Vectorizer free cost helpers -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SLPCostAnalysis.h"
#include "SLPTypeUtils.h"
#include "SLPUtils.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/VectorTypeUtils.h"
#include "llvm/Support/Casting.h"

#include <cassert>
#include <utility>

using namespace llvm;

namespace llvm::slpvectorizer {

InstructionCost getShuffleCost(const TargetTransformInfo &TTI,
                               TTI::ShuffleKind Kind, VectorType *Tp,
                               const TTI::TargetCostKind CostKind,
                               ArrayRef<int> Mask, int Index, VectorType *SubTp,
                               ArrayRef<const Value *> Args) {
  VectorType *DstTy = Tp;
  if (!Mask.empty())
    DstTy = FixedVectorType::get(Tp->getScalarType(), Mask.size());

  if (Kind != TTI::SK_PermuteTwoSrc)
    return TTI.getShuffleCost(Kind, DstTy, Tp, CostKind, Mask, Index, SubTp,
                              Args);
  int NumSrcElts = Tp->getElementCount().getKnownMinValue();
  int NumSubElts;
  if (Mask.size() > 2 && ShuffleVectorInst::isInsertSubvectorMask(
                             Mask, NumSrcElts, NumSubElts, Index)) {
    if (Index + NumSubElts > NumSrcElts &&
        Index + NumSrcElts <= static_cast<int>(Mask.size()))
      return TTI.getShuffleCost(TTI::SK_InsertSubvector, DstTy, Tp, CostKind,
                                Mask, Index, Tp);
  }
  return TTI.getShuffleCost(Kind, DstTy, Tp, CostKind, Mask, Index, SubTp,
                            Args);
}

std::pair<InstructionCost, InstructionCost>
getGEPCosts(const TargetTransformInfo &TTI, ArrayRef<Value *> Ptrs,
            Value *BasePtr, unsigned Opcode, const TTI::TargetCostKind CostKind,
            Type *ScalarTy, VectorType *VecTy) {
  InstructionCost ScalarCost = 0;
  InstructionCost VecCost = 0;
  // Here we differentiate two cases: (1) when Ptrs represent a regular
  // vectorization tree node (as they are pointer arguments of scattered
  // loads) or (2) when Ptrs are the arguments of loads or stores being
  // vectorized as plane wide unit-stride load/store since all the
  // loads/stores are known to be from/to adjacent locations.
  if (Opcode == Instruction::Load || Opcode == Instruction::Store) {
    // Case 2: estimate costs for pointer related costs when vectorizing to
    // a wide load/store.
    // Scalar cost is estimated as a set of pointers with known relationship
    // between them.
    // For vector code we will use BasePtr as argument for the wide load/store
    // but we also need to account all the instructions which are going to
    // stay in vectorized code due to uses outside of these scalar
    // loads/stores.
    ScalarCost = TTI.getPointersChainCost(
        Ptrs, BasePtr, TTI::PointersChainInfo::getUnitStride(), ScalarTy,
        CostKind);

    SmallVector<const Value *> PtrsRetainedInVecCode;
    for (Value *V : Ptrs) {
      if (V == BasePtr) {
        PtrsRetainedInVecCode.push_back(V);
        continue;
      }
      auto *Ptr = dyn_cast<GetElementPtrInst>(V);
      // For simplicity assume Ptr to stay in vectorized code if it's not a
      // GEP instruction. We don't care since it's cost considered free.
      // TODO: We should check for any uses outside of vectorizable tree
      // rather than just single use.
      if (!Ptr || !Ptr->hasOneUse())
        PtrsRetainedInVecCode.push_back(V);
    }

    if (PtrsRetainedInVecCode.size() == Ptrs.size()) {
      // If all pointers stay in vectorized code then we don't have
      // any savings on that.
      return std::make_pair(TTI::TCC_Free, TTI::TCC_Free);
    }
    VecCost = TTI.getPointersChainCost(PtrsRetainedInVecCode, BasePtr,
                                       TTI::PointersChainInfo::getKnownStride(),
                                       VecTy, CostKind);
  } else {
    // Case 1: Ptrs are the arguments of loads that we are going to transform
    // into masked gather load intrinsic.
    // All the scalar GEPs will be removed as a result of vectorization.
    // For any external uses of some lanes extract element instructions will
    // be generated (which cost is estimated separately).
    TTI::PointersChainInfo PtrsInfo =
        all_of(Ptrs,
               [](const Value *V) {
                 auto *Ptr = dyn_cast<GetElementPtrInst>(V);
                 return Ptr && !Ptr->hasAllConstantIndices();
               })
            ? TTI::PointersChainInfo::getUnknownStride()
            : TTI::PointersChainInfo::getKnownStride();

    ScalarCost =
        TTI.getPointersChainCost(Ptrs, BasePtr, PtrsInfo, ScalarTy, CostKind);
    auto *BaseGEP = dyn_cast<GEPOperator>(BasePtr);
    if (!BaseGEP) {
      auto *It = find_if(Ptrs, IsaPred<GEPOperator>);
      if (It != Ptrs.end())
        BaseGEP = cast<GEPOperator>(*It);
    }
    if (BaseGEP) {
      SmallVector<const Value *> Indices(BaseGEP->indices());
      VecCost = TTI.getGEPCost(BaseGEP->getSourceElementType(),
                               BaseGEP->getPointerOperand(), Indices, CostKind,
                               VecTy);
    }
  }

  return std::make_pair(ScalarCost, VecCost);
}

InstructionCost getBlendedLoadCost(const TargetTransformInfo &TTI, Type *VecTy,
                                   Align Alignment, unsigned AddressSpace,
                                   const TTI::TargetCostKind CostKind) {
  Type *CmpTy = CmpInst::makeCmpResultType(VecTy);
  return 2 * TTI.getMemIntrinsicInstrCost(
                 MemIntrinsicCostAttributes(Intrinsic::masked_load, VecTy,
                                            Alignment, AddressSpace),
                 CostKind) +
         TTI.getArithmeticInstrCost(Instruction::Xor, CmpTy, CostKind) +
         TTI.getCmpSelInstrCost(Instruction::Select, VecTy, CmpTy,
                                CmpInst::BAD_ICMP_PREDICATE, CostKind);
}

InstructionCost getMaskedDivRemCost(const TargetTransformInfo &TTI, bool ReVec,
                                    unsigned Opcode, Type *ScalarTy,
                                    unsigned NumElts,
                                    const TTI::TargetCostKind CostKind,
                                    FixedVectorType **PaddedTy) {
  FixedVectorType *PaddedVecTy =
      getMaskedDivRemType(TTI, Opcode, ScalarTy, NumElts, ReVec);
  if (!PaddedVecTy)
    return InstructionCost::getInvalid();
  // One mask bit per element of the padded vector, not per padded lane.
  auto *MaskTy =
      FixedVectorType::get(IntegerType::getInt1Ty(ScalarTy->getContext()),
                           PaddedVecTy->getNumElements());
  InstructionCost DirectCost = TTI.getArithmeticInstrCost(
      Opcode, getWidenedType(ScalarTy, NumElts), CostKind);
  IntrinsicCostAttributes ICA(getMaskedDivRemIntrinsic(Opcode), PaddedVecTy,
                              {PaddedVecTy, PaddedVecTy, MaskTy});
  InstructionCost MaskedCost = TTI.getIntrinsicInstrCost(ICA, CostKind);
  if (!MaskedCost.isValid() || MaskedCost >= DirectCost)
    return InstructionCost::getInvalid();
  if (PaddedTy)
    *PaddedTy = PaddedVecTy;
  return MaskedCost;
}

InstructionCost
getScalarizationOverhead(const TargetTransformInfo &TTI, bool ReVec,
                         Type *ScalarTy, VectorType *Ty,
                         const APInt &DemandedElts, bool Insert, bool Extract,
                         const TTI::TargetCostKind CostKind, bool ForPoisonSrc,
                         ArrayRef<Value *> VL, TTI::VectorInstrContext VIC) {
  assert(!isa<ScalableVectorType>(Ty) &&
         "ScalableVectorType is not supported.");
  assert(getNumElements(ScalarTy) * DemandedElts.getBitWidth() ==
             getNumElements(Ty) &&
         "Incorrect usage.");
  if (auto *VecTy = dyn_cast<FixedVectorType>(ScalarTy)) {
    assert(ReVec && "Only supported by REVEC.");
    // If ScalarTy is FixedVectorType, we should use CreateInsertVector instead
    // of CreateInsertElement.
    unsigned ScalarTyNumElements = VecTy->getNumElements();
    InstructionCost Cost = 0;
    for (unsigned I : seq(DemandedElts.getBitWidth())) {
      if (!DemandedElts[I])
        continue;
      if (Insert)
        Cost += getShuffleCost(TTI, TTI::SK_InsertSubvector, Ty, CostKind, {},
                               I * ScalarTyNumElements, VecTy);
      if (Extract)
        Cost += getShuffleCost(TTI, TTI::SK_ExtractSubvector, Ty, CostKind, {},
                               I * ScalarTyNumElements, VecTy);
    }
    return Cost;
  }
  return TTI.getScalarizationOverhead(Ty, DemandedElts, Insert, Extract,
                                      CostKind, ForPoisonSrc, VL, VIC);
}

InstructionCost getVectorInstrCost(
    const TargetTransformInfo &TTI, bool ReVec, Type *ScalarTy, unsigned Opcode,
    Type *Val, const TTI::TargetCostKind CostKind, unsigned Index,
    Value *Scalar,
    ArrayRef<std::tuple<Value *, User *, int>> ScalarUserAndIdx) {
  if (Opcode == Instruction::ExtractElement) {
    if (auto *VecTy = dyn_cast<FixedVectorType>(ScalarTy)) {
      assert(ReVec && "Only supported by REVEC.");
      assert(isa<VectorType>(Val) && "Val must be a vector type.");
      return getShuffleCost(TTI, TTI::SK_ExtractSubvector,
                            cast<VectorType>(Val), CostKind, {},
                            Index * VecTy->getNumElements(), VecTy);
    }
  }
  return TTI.getVectorInstrCost(Opcode, Val, CostKind, Index, Scalar,
                                ScalarUserAndIdx);
}

InstructionCost getExtractWithExtendCost(const TargetTransformInfo &TTI,
                                         bool ReVec, unsigned Opcode, Type *Dst,
                                         VectorType *VecTy, unsigned Index,
                                         const TTI::TargetCostKind CostKind) {
  if (isVectorizedTy(Dst)) {
    assert(ReVec && "Only supported by REVEC.");
    auto *SubTp = cast<FixedVectorType>(
        getWidenedType(toScalarizedTy(VecTy), getNumElements(Dst)));
    return getShuffleCost(TTI, TTI::SK_ExtractSubvector, VecTy, CostKind, {},
                          Index * getNumElements(Dst), SubTp) +
           TTI.getCastInstrCost(Opcode, Dst, SubTp, TTI::CastContextHint::None,
                                CostKind);
  }
  return TTI.getExtractWithExtendCost(Opcode, Dst, VecTy, Index, CostKind);
}

} // namespace llvm::slpvectorizer
