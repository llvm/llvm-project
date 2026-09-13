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
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/VectorTypeUtils.h"
#include "llvm/Support/Casting.h"

#include <cassert>
#include <utility>

using namespace llvm;
using namespace llvm::PatternMatch;

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

/// Returns the cast context hint for the trunc of the booleanized reduction
/// result, which inherits the uses of the reduction root \p Root.
static TTI::CastContextHint getBoolReduxResultCCH(const Value *Root) {
  if (!Root->hasOneUse())
    return TTI::CastContextHint::None;
  const Value *U = *Root->user_begin();
  if (isa<StoreInst>(U))
    return TTI::CastContextHint::Normal;
  if (match(U, m_Intrinsic<Intrinsic::masked_store>()))
    return TTI::CastContextHint::Masked;
  if (match(U, m_Intrinsic<Intrinsic::masked_scatter>()))
    return TTI::CastContextHint::GatherScatter;
  return TTI::CastContextHint::None;
}

InstructionCost getBoolReduxWideRdxCost(const TargetTransformInfo &TTI,
                                        RecurKind RdxKind,
                                        FixedVectorType *VecTy,
                                        const Value *Root, FastMathFlags FMF,
                                        const TTI::TargetCostKind CostKind) {
  Type *I1Ty = Type::getInt1Ty(VecTy->getContext());
  return TTI.getArithmeticReductionCost(
             RecurrenceDescriptor::getOpcode(RdxKind), VecTy, FMF, CostKind) +
         TTI.getCastInstrCost(Instruction::Trunc, I1Ty, VecTy->getScalarType(),
                              getBoolReduxResultCCH(Root), CostKind);
}

InstructionCost getBoolReduxBitcastCmpCost(const TargetTransformInfo &TTI,
                                           RecurKind RdxKind,
                                           FixedVectorType *VecTy,
                                           const Value *Root,
                                           ArrayRef<Instruction *> ChainInsts,
                                           const TTI::TargetCostKind CostKind) {
  // The new instructions are costed in the context of the replaced cast chain
  // instructions.
  auto TruncIt =
      find_if(ChainInsts, [](Instruction *I) { return isa<TruncInst>(I); });
  const Instruction *TruncI = TruncIt == ChainInsts.end() ? nullptr : *TruncIt;
  auto CmpIt =
      find_if(ChainInsts, [](Instruction *I) { return isa<ICmpInst>(I); });
  const Instruction *CmpI = CmpIt == ChainInsts.end() ? nullptr : *CmpIt;
  unsigned VF = VecTy->getNumElements();
  auto *I1VecTy =
      FixedVectorType::get(Type::getInt1Ty(VecTy->getContext()), VF);
  Type *IntTy = IntegerType::get(VecTy->getContext(), VF);
  Constant *CmpRHS = RdxKind == RecurKind::And
                         ? Constant::getAllOnesValue(IntTy)
                         : Constant::getNullValue(IntTy);
  return TTI.getCastInstrCost(Instruction::Trunc, I1VecTy, VecTy,
                              TTI.getCastContextHint(TruncI), CostKind,
                              TruncI) +
         TTI.getCastInstrCost(Instruction::BitCast, IntTy, I1VecTy,
                              TTI.getCastContextHint(TruncI), CostKind) +
         TTI.getCmpSelInstrCost(Instruction::ICmp, IntTy, /*CondTy=*/nullptr,
                                RdxKind == RecurKind::And ? CmpInst::ICMP_EQ
                                                          : CmpInst::ICMP_NE,
                                CostKind, TTI.getOperandInfo(Root),
                                TTI.getOperandInfo(CmpRHS), CmpI);
}

InstructionCost getBitPackCost(const TargetTransformInfo &TTI,
                               FixedVectorType *SrcTy, Type *ResultTy,
                               const BitPackInfo &Info, bool FreeByteTrunc,
                               TTI::TargetCostKind CostKind,
                               const TargetLibraryInfo *TLI,
                               const Instruction *CxtI, unsigned &ShiftWidth) {
  unsigned BitWidth = SrcTy->getScalarSizeInBits();
  unsigned NumElts = SrcTy->getNumElements();
  uint64_t MaxAmt = *max_element(Info.LShrAmts);
  // The shift amounts form a constant vector.
  TTI::OperandValueInfo ShiftAmtInfo = {
      all_of(Info.LShrAmts,
             [&](uint64_t A) { return A == Info.LShrAmts.front(); })
          ? TTI::OK_UniformConstantValue
          : TTI::OK_NonUniformConstantValue,
      all_of(Info.LShrAmts, isPowerOf2_64) ? TTI::OP_PowerOf2 : TTI::OP_None};
  // After the shift the field content of each lane sits in the low bits of
  // the lane, so the packing is a single byte shuffle of the shifted lanes.
  // Pick the cheapest shift width: the narrowest type still holding the field
  // content is not always the cheapest (e.g. missing narrow variable shifts).
  Type *Int8Ty = IntegerType::get(SrcTy->getContext(), 8);
  unsigned OutBytes = BitWidth / 8;
  auto *PackTy = FixedVectorType::get(Int8Ty, OutBytes);
  unsigned MinShiftWidth = 8;
  while (MinShiftWidth < MaxAmt + Info.FieldWidth)
    MinShiftWidth *= 2;
  InstructionCost NewCost = InstructionCost::getInvalid();
  ShiftWidth = 0;
  for (unsigned W2 = MinShiftWidth; W2 <= BitWidth; W2 *= 2) {
    auto *ShiftTy = FixedVectorType::get(
        IntegerType::get(SrcTy->getContext(), W2), NumElts);
    unsigned BytesPerLane = W2 / 8;
    unsigned InBytes = NumElts * BytesPerLane;
    SmallVector<int> Mask =
        getBitPackMask(Info, OutBytes, NumElts, BytesPerLane);
    InstructionCost C =
        TTI.getCastInstrCost(Instruction::BitCast, ResultTy, PackTy,
                             TTI::CastContextHint::None, CostKind);
    // A plain byte reversal of the shifted lanes is a bswap, no shuffle.
    if (ShuffleVectorInst::isReverseMask(Mask, InBytes)) {
      IntrinsicCostAttributes CostAttrs(Intrinsic::bswap, ResultTy, {ResultTy});
      C += TTI.getIntrinsicInstrCost(CostAttrs, CostKind);
    } else if (!ShuffleVectorInst::isIdentityMask(Mask, InBytes)) {
      C += TTI.getShuffleCost(
          is_contained(Info.LaneOfField, BitPackInfo::NoLane)
              ? TargetTransformInfo::SK_PermuteTwoSrc
              : TargetTransformInfo::SK_PermuteSingleSrc,
          PackTy, FixedVectorType::get(Int8Ty, InBytes), CostKind, Mask,
          /*Index=*/0, /*SubTp=*/nullptr, /*Args=*/{}, CxtI);
    }
    if (W2 != BitWidth && !(W2 == 8 && FreeByteTrunc))
      C += TTI.getCastInstrCost(Instruction::Trunc, ShiftTy, SrcTy,
                                TTI::CastContextHint::None, CostKind);
    if (Info.needsShift())
      C += TTI.getArithmeticInstrCost(Instruction::LShr, ShiftTy, CostKind,
                                      /*Opd1Info=*/{}, ShiftAmtInfo,
                                      /*Args=*/{}, CxtI, TLI);
    if (C.isValid() && (!NewCost.isValid() || C < NewCost)) {
      NewCost = C;
      ShiftWidth = W2;
    }
  }
  return NewCost;
}

} // namespace llvm::slpvectorizer
