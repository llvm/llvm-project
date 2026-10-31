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
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/VectorTypeUtils.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"

#include <cassert>
#include <utility>

using namespace llvm;
using namespace llvm::PatternMatch;

namespace llvm::slpvectorizer {

InstructionCost getShuffleCost(const TargetTransformInfo &TTI,
                               TTI::ShuffleKind Kind, VectorType *Tp,
                               const TTI::TargetCostKind CostKind,
                               ArrayRef<int> Mask, int Index, VectorType *SubTp,
                               ArrayRef<const Value *> Args,
                               TTI::VectorInstrContext VIC) {
  VectorType *DstTy = Tp;
  if (!Mask.empty())
    DstTy = FixedVectorType::get(Tp->getScalarType(), Mask.size());

  if (Kind != TTI::SK_PermuteTwoSrc)
    return TTI.getShuffleCost(Kind, DstTy, Tp, CostKind, Mask, Index, SubTp,
                              Args, /*CtxI=*/nullptr, VIC);
  int NumSrcElts = Tp->getElementCount().getKnownMinValue();
  int NumSubElts;
  if (Mask.size() > 2 && ShuffleVectorInst::isInsertSubvectorMask(
                             Mask, NumSrcElts, NumSubElts, Index)) {
    if (Index + NumSubElts > NumSrcElts &&
        Index + NumSrcElts <= static_cast<int>(Mask.size()))
      return TTI.getShuffleCost(TTI::SK_InsertSubvector, DstTy, Tp, CostKind,
                                Mask, Index, Tp);
  }
  return TTI.getShuffleCost(Kind, DstTy, Tp, CostKind, Mask, Index, SubTp, Args,
                            /*CtxI=*/nullptr, VIC);
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

    const bool IsSameBase =
        !all_of(Ptrs, IsaPred<GetElementPtrInst>) ||
        all_equal(map_range(Ptrs, [](Value *V) {
          return cast<GetElementPtrInst>(V)->getPointerOperand();
        }));
    // The GEPs of the masked gather loads are accessed with the loaded type
    // and form a chain only if the lanes share the base.
    Type *AccessTy = ScalarTy;
    if (all_of(Ptrs, [](const Value *V) {
          auto *Ptr = dyn_cast<GetElementPtrInst>(V);
          return Ptr && Ptr->hasOneUse() && isa<LoadInst>(Ptr->user_back());
        })) {
      PtrsInfo.IsSameBaseAddress = IsSameBase;
      AccessTy = Ptrs.front()->user_back()->getType();
    }
    ScalarCost =
        TTI.getPointersChainCost(Ptrs, BasePtr, PtrsInfo, AccessTy, CostKind);
    auto *BaseGEP = dyn_cast<GEPOperator>(BasePtr);
    if (!BaseGEP) {
      auto *It = find_if(Ptrs, IsaPred<GEPOperator>);
      if (It != Ptrs.end())
        BaseGEP = cast<GEPOperator>(*It);
    }
    if (!IsSameBase) {
      // The different bases are not folded into the addressing of the vector
      // access, the scaled indices are added to them.
      const DataLayout &DL = cast<Instruction>(Ptrs.front())->getDataLayout();
      Type *IdxTy = DL.getIndexType(VecTy);
      VecCost = TTI.getArithmeticInstrCost(Instruction::Add, IdxTy, CostKind);
      uint64_t Stride = DL.getTypeAllocSize(BaseGEP->getSourceElementType())
                            .getKnownMinValue();
      if (Stride != 1 && !all_of(Ptrs, [](Value *V) {
            return isa<Constant>(cast<GetElementPtrInst>(V)->getOperand(1));
          }))
        VecCost += TTI.getArithmeticInstrCost(
            isPowerOf2_64(Stride) ? Instruction::Shl : Instruction::Mul, IdxTy,
            CostKind, {}, {TTI::OK_UniformConstantValue, TTI::OP_None});
    } else if (BaseGEP) {
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

SmallVector<Type *> buildIntrinsicArgTypes(const CallInst *CI,
                                           const Intrinsic::ID ID,
                                           const unsigned VF, unsigned MinBW,
                                           const TargetTransformInfo *TTI) {
  SmallVector<Type *> ArgTys;
  for (auto [Idx, Arg] : enumerate(CI->args())) {
    if (ID != Intrinsic::not_intrinsic) {
      if (isVectorIntrinsicWithScalarOpAtArg(ID, Idx, TTI)) {
        ArgTys.push_back(Arg->getType());
        continue;
      }
      if (MinBW > 0) {
        ArgTys.push_back(
            getWidenedType(IntegerType::get(CI->getContext(), MinBW), VF));
        continue;
      }
    }
    ArgTys.push_back(getWidenedType(Arg->getType(), VF));
  }
  return ArgTys;
}

std::pair<InstructionCost, InstructionCost>
getVectorCallCosts(CallInst *CI, Type *VecTy, const TargetTransformInfo *TTI,
                   const TargetLibraryInfo *TLI, ArrayRef<Type *> ArgTys,
                   const TTI::TargetCostKind CostKind) {
  auto Shape = VFShape::get(CI->getFunctionType(),
                            ElementCount::getFixed(getNumElements(VecTy)),
                            false /*HasGlobalPred*/);
  Function *VecFunc = VFDatabase(*CI).getVectorizedFunction(Shape);
  auto LibCost = InstructionCost::getInvalid();
  if (!CI->isNoBuiltin() && VecFunc) {
    // Calculate the cost of the vector library call.
    // If the corresponding vector call is cheaper, return its cost.
    LibCost = TTI->getCallInstrCost(nullptr, VecTy, ArgTys, CostKind);
  }
  Intrinsic::ID ID = getVectorIntrinsicIDForCall(CI, TLI);

  // Calculate the cost of the vector intrinsic call.
  FastMathFlags FMF;
  if (auto *FPCI = dyn_cast<FPMathOperator>(CI))
    FMF = FPCI->getFastMathFlags();
  const InstructionCost ScalarLimit = 10000;
  IntrinsicCostAttributes CostAttrs(ID, VecTy, ArgTys, FMF,
                                    dyn_cast<IntrinsicInst>(CI),
                                    LibCost.isValid() ? LibCost : ScalarLimit);
  auto IntrinsicCost = TTI->getIntrinsicInstrCost(CostAttrs, CostKind);
  if (LibCost.isValid()) {
    if (IntrinsicCost > LibCost)
      IntrinsicCost = InstructionCost::getInvalid();
  } else if (IntrinsicCost > ScalarLimit) {
    // A type-based query always scalarizes struct-returning intrinsics (e.g.
    // llvm.sincos), which do not have a VFDatabase name mapping. Retry with an
    // argument-aware query (as the loop vectorizer does) so such lowerings are
    // taken into account.
    SmallVector<const Value *> Args(CI->args());
    IntrinsicCostAttributes ArgAwareAttrs(
        ID, VecTy, Args, ArgTys, FMF, dyn_cast<IntrinsicInst>(CI), ScalarLimit);
    IntrinsicCost = TTI->getIntrinsicInstrCost(ArgAwareAttrs, CostKind);
    if (IntrinsicCost > ScalarLimit)
      IntrinsicCost = InstructionCost::getInvalid();
  }

  return {IntrinsicCost, LibCost};
}

InstructionCost getMergeInputOpsNodeCost(const TargetTransformInfo &TTI,
                                         const TargetLibraryInfo &TLI,
                                         const InstructionsState &S,
                                         ArrayRef<Value *> VL, Type *ScalarTy,
                                         TTI::TargetCostKind CostKind) {
  assert(S && !S.isAltShuffle() && !S.areInstructionsWithCopyableElements() &&
         "Expected a plain same-opcode node.");
  Instruction *MainOp = S.getMainOp();
  const unsigned Opcode = S.getOpcode();
  const unsigned NumLanes = VL.size();
  auto *VecTy = getWidenedType(ScalarTy, NumLanes);
  // Interchangeable lanes compute the node's opcode from other operands, so
  // their operand columns fall back to the default info.
  auto GetOperandInfo = [&](unsigned OpIdx) -> TTI::OperandValueInfo {
    SmallVector<Value *> Column;
    Column.reserve(NumLanes);
    for (Value *V : VL) {
      auto *I = dyn_cast<Instruction>(V);
      if (!I || I->getOpcode() != Opcode)
        return {};
      Column.push_back(I->getOperand(OpIdx));
    }
    return TTI::getOperandInfo(Column);
  };
  switch (Opcode) {
  case Instruction::PHI:
  case Instruction::Freeze:
    // No code of its own; the node is just the vectorized incoming value.
    return TTI::TCC_Free;
  case Instruction::ExtractElement:
  case Instruction::ExtractValue: {
    // The node reuses its source vector and only permutes the lanes into
    // place. Extracts from per-lane struct-returning calls fold into the call
    // node instead.
    Value *Src = MainOp->getOperand(0);
    if (any_of(VL, [&](Value *V) {
          auto *I = dyn_cast<Instruction>(V);
          return I && I->getOperand(0) != Src;
        }))
      return TTI::TCC_Free;
    unsigned NumSrcElts;
    if (auto *SrcVecTy = dyn_cast<FixedVectorType>(Src->getType()))
      NumSrcElts = SrcVecTy->getNumElements();
    else if (auto *ArrTy = dyn_cast<ArrayType>(Src->getType()))
      NumSrcElts = ArrTy->getNumElements();
    else
      NumSrcElts = Src->getType()->getStructNumElements();
    SmallVector<int> Mask(NumLanes, PoisonMaskElem);
    for (auto [Lane, V] : enumerate(VL))
      if (auto *I = dyn_cast<Instruction>(V))
        if (std::optional<unsigned> Idx = getExtractIndex(I))
          Mask[Lane] = *Idx;
    if (NumSrcElts == NumLanes &&
        ShuffleVectorInst::isIdentityMask(Mask, NumLanes))
      return TTI::TCC_Free;
    return getShuffleCost(
        TTI, TTI::SK_PermuteSingleSrc,
        cast<FixedVectorType>(getWidenedType(ScalarTy, NumSrcElts)), CostKind,
        Mask);
  }
  case Instruction::Load: {
    auto *LI = cast<LoadInst>(MainOp);
    return TTI.getMemoryOpCost(Instruction::Load, VecTy, LI->getAlign(),
                               LI->getPointerAddressSpace(), CostKind,
                               TTI::getOperandInfo(LI->getPointerOperand()));
  }
  case Instruction::ICmp:
  case Instruction::FCmp:
  case Instruction::Select: {
    Type *ValTy = Opcode == Instruction::Select
                      ? ScalarTy
                      : MainOp->getOperand(0)->getType();
    const CmpInst::Predicate BadPred = ValTy->isFPOrFPVectorTy()
                                           ? CmpInst::BAD_FCMP_PREDICATE
                                           : CmpInst::BAD_ICMP_PREDICATE;
    // The predicate of a compare lane, or of a select lane's compare.
    auto GetPred = [&](Value *V) -> CmpInst::Predicate {
      CmpPredicate Pred;
      auto MatchCmp = m_Cmp(Pred, m_Value(), m_Value());
      if (match(V, MatchCmp) ||
          match(V, m_Select(MatchCmp, m_Value(), m_Value())))
        return Pred;
      return BadPred;
    };
    // Same choice as getEntryCost(): lanes with the swapped predicate swap
    // their operands in the node; any other mix has no single predicate.
    CmpInst::Predicate VecPred = GetPred(MainOp);
    bool AnySwapped = false;
    for (Value *V : VL) {
      if (!isa<Instruction>(V))
        continue;
      CmpInst::Predicate Pred = GetPred(V);
      if (Pred == VecPred)
        continue;
      if (VecPred != BadPred && Pred == CmpInst::getSwappedPredicate(VecPred)) {
        AnySwapped = true;
        continue;
      }
      VecPred = BadPred;
      break;
    }
    if (Opcode == Instruction::Select)
      return TTI.getCmpSelInstrCost(
          Opcode, VecTy,
          getWidenedType(MainOp->getOperand(0)->getType(), NumLanes), VecPred,
          CostKind, GetOperandInfo(1), GetOperandInfo(2), MainOp);
    // Swapped lanes take their operands in the other order, so the operand
    // columns are only known without them.
    const bool KnownOperands = VecPred != BadPred && !AnySwapped;
    return TTI.getCmpSelInstrCost(
        Opcode, getWidenedType(ValTy, NumLanes), VecTy, VecPred, CostKind,
        KnownOperands ? GetOperandInfo(0) : TTI::OperandValueInfo(),
        KnownOperands ? GetOperandInfo(1) : TTI::OperandValueInfo(), MainOp);
  }
  case Instruction::Call: {
    auto *CI = cast<CallInst>(MainOp);
    Intrinsic::ID ID = getVectorIntrinsicIDForCall(CI, &TLI);
    SmallVector<Type *> ArgTys =
        buildIntrinsicArgTypes(CI, ID, NumLanes, /*MinBW=*/0, &TTI);
    auto [IntrinsicCost, LibCost] =
        getVectorCallCosts(CI, VecTy, &TTI, &TLI, ArgTys, CostKind);
    return std::min(IntrinsicCost, LibCost);
  }
  default:
    break;
  }
  if (Instruction::isBinaryOp(Opcode))
    return TTI.getArithmeticInstrCost(Opcode, VecTy, CostKind,
                                      GetOperandInfo(0), GetOperandInfo(1), {},
                                      MainOp);
  if (Instruction::isUnaryOp(Opcode))
    return TTI.getArithmeticInstrCost(Opcode, VecTy, CostKind,
                                      GetOperandInfo(0), {}, {}, MainOp);
  if (Instruction::isCast(Opcode)) {
    // The cast folds into its source memory op only if every lane does;
    // otherwise the vector node just has a plain vector source.
    TTI::CastContextHint CCH = TTI::getCastContextHint(MainOp);
    if (any_of(VL, [&](Value *V) {
          return TTI::getCastContextHint(dyn_cast<Instruction>(V)) != CCH;
        }))
      CCH = TTI::CastContextHint::None;
    return TTI.getCastInstrCost(
        Opcode, VecTy,
        getWidenedType(MainOp->getOperand(0)->getType(), NumLanes), CCH,
        CostKind);
  }
  // Geps, inserts, shuffles and stores produce pointers, vectors or nothing,
  // so they never feed a binary operator; any other opcode gathers.
  llvm_unreachable("Unexpected opcode for a merged-inputs operand node.");
}

InstructionCost getWidenedStridedCastCost(const TargetTransformInfo &TTI,
                                          Type *SrcTy, Type *DstTy,
                                          const DataLayout &DL,
                                          TTI::CastContextHint CCH,
                                          TTI::TargetCostKind CostKind) {
  bool ToPtr = cast<VectorType>(DstTy)->getElementType()->isPointerTy();
  if (ToPtr == cast<VectorType>(SrcTy)->getElementType()->isPointerTy())
    return TTI.getCastInstrCost(Instruction::BitCast, DstTy, SrcTy, CCH,
                                CostKind);
  // The ptr/int conversion keeps the vector shape, the bitcast transforms the
  // resulting integer vector.
  if (ToPtr) {
    Type *IntVecTy = DL.getIntPtrType(DstTy);
    return TTI.getCastInstrCost(Instruction::IntToPtr, DstTy, IntVecTy, CCH,
                                CostKind) +
           TTI.getCastInstrCost(Instruction::BitCast, IntVecTy, SrcTy, CCH,
                                CostKind);
  }
  Type *IntVecTy = DL.getIntPtrType(SrcTy);
  return TTI.getCastInstrCost(Instruction::PtrToInt, IntVecTy, SrcTy, CCH,
                              CostKind) +
         TTI.getCastInstrCost(Instruction::BitCast, DstTy, IntVecTy, CCH,
                              CostKind);
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
    Value *Scalar, ArrayRef<std::tuple<Value *, User *, int>> ScalarUserAndIdx,
    TTI::VectorInstrContext VIC) {
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
                                ScalarUserAndIdx, VIC);
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
         TTI.getCmpSelInstrCost(
             Instruction::ICmp, IntTy, CmpInst::makeCmpResultType(IntTy),
             RdxKind == RecurKind::And ? CmpInst::ICMP_EQ : CmpInst::ICMP_NE,
             CostKind, TTI.getOperandInfo(Root), TTI.getOperandInfo(CmpRHS),
             CmpI);
}

static InstructionCost
getBoolLogicRdxBitcastCost(RecurKind Kind, const TargetTransformInfo &TTI,
                           FixedVectorType *VectorTy, TTI::CastContextHint Ctx,
                           TTI::TargetCostKind CostKind) {
  assert((Kind == RecurKind::And || Kind == RecurKind::Or) &&
         VectorTy->getElementType()->isIntegerTy(1) &&
         "Expected and/or reduction of i1");
  auto *IntTy =
      IntegerType::get(VectorTy->getContext(), getNumElements(VectorTy));
  CmpInst::Predicate Pred =
      Kind == RecurKind::And ? CmpInst::ICMP_EQ : CmpInst::ICMP_NE;
  // The compare is against the all-ones (and) or zero (or) constant.
  return TTI.getCastInstrCost(Instruction::BitCast, IntTy, VectorTy, Ctx,
                              CostKind) +
         TTI.getCmpSelInstrCost(Instruction::ICmp, IntTy,
                                CmpInst::makeCmpResultType(IntTy), Pred,
                                CostKind, /*Op1Info=*/{},
                                {TTI::OK_UniformConstantValue, TTI::OP_None});
}

std::pair<InstructionCost, bool>
getI1ReductionCost(RecurKind Kind, const TargetTransformInfo &TTI,
                   FixedVectorType *VectorTy, Type *ScalarTy,
                   TTI::CastContextHint Ctx, TTI::TargetCostKind CostKind) {
  unsigned RdxOpcode = RecurrenceDescriptor::getOpcode(Kind);
  if (Kind == RecurKind::And || Kind == RecurKind::Or) {
    InstructionCost RdxCost = TTI.getArithmeticReductionCost(
        RdxOpcode, VectorTy, std::nullopt, CostKind);
    InstructionCost BitcastCost =
        getBoolLogicRdxBitcastCost(Kind, TTI, VectorTy, Ctx, CostKind);
    return {std::min(RdxCost, BitcastCost), BitcastCost < RdxCost};
  }
  assert(Kind == RecurKind::Add && !ScalarTy->isIntegerTy(1) &&
         "Expected add reduction of zexted i1 values");
  // The bitcast+ctpop form is estimated as the cheaper of the extended
  // reduction cost, which models it for the zexted i1 add reduction, and the
  // explicitly priced components, including the cast of the ctpop result to
  // the destination type.
  auto *IntTy =
      IntegerType::get(VectorTy->getContext(), getNumElements(VectorTy));
  InstructionCost ExplicitCost =
      TTI.getCastInstrCost(Instruction::BitCast, IntTy, VectorTy, Ctx,
                           CostKind) +
      TTI.getIntrinsicInstrCost(
          IntrinsicCostAttributes(Intrinsic::ctpop, IntTy, {IntTy}), CostKind);
  if (IntTy != ScalarTy)
    ExplicitCost += TTI.getCastInstrCost(IntTy->getBitWidth() <
                                                 ScalarTy->getIntegerBitWidth()
                                             ? Instruction::ZExt
                                             : Instruction::Trunc,
                                         ScalarTy, IntTy, Ctx, CostKind);
  InstructionCost CtpopCost = std::min(
      TTI.getExtendedReductionCost(RdxOpcode, /*IsUnsigned=*/true, ScalarTy,
                                   VectorTy, std::nullopt, CostKind),
      ExplicitCost);
  // The plain form is the zext to the wide vector type plus the reduction.
  auto *ExtTy = VectorType::get(ScalarTy, VectorTy);
  InstructionCost ExtRdxCost =
      TTI.getCastInstrCost(Instruction::ZExt, ExtTy, VectorTy, Ctx, CostKind) +
      TTI.getArithmeticReductionCost(RdxOpcode, ExtTy, std::nullopt, CostKind);
  return {std::min(ExtRdxCost, CtpopCost), CtpopCost <= ExtRdxCost};
}

InstructionCost getBitPackCost(const TargetTransformInfo &TTI,
                               FixedVectorType *SrcTy, Type *ResultTy,
                               const BitPackInfo &Info, unsigned ZExtSrcWidth,
                               TTI::CastContextHint CCH,
                               TTI::TargetCostKind CostKind,
                               const TargetLibraryInfo *TLI,
                               const Instruction *CtxI, unsigned &ShiftWidth) {
  unsigned BitWidth = SrcTy->getScalarSizeInBits();
  unsigned NumElts = SrcTy->getNumElements();
  uint64_t MaxAmt = *max_element(Info.LShrAmts);
  // The shift amounts form a constant vector.
  TTI::OperandValueInfo ShiftAmtInfo = {
      all_equal(Info.LShrAmts) ? TTI::OK_UniformConstantValue
                               : TTI::OK_NonUniformConstantValue,
      all_of(Info.LShrAmts,
             [](uint64_t A) { return A == 0 || isPowerOf2_64(A); })
          ? TTI::OP_PowerOf2
          : TTI::OP_None};
  // After the shift the field content of each lane sits in the low bits of
  // the lane, so the packing is a single byte shuffle of the shifted lanes.
  // Pick the cheapest shift width: the narrowest type still holding the field
  // content is not always the cheapest (e.g. missing narrow variable shifts).
  Type *Int8Ty = Type::getInt8Ty(SrcTy->getContext());
  assert(BitWidth % 8 == 0 &&
         "The byte-multiple field width divides the result bit width.");
  unsigned OutBytes = ResultTy->getScalarSizeInBits() / 8;
  auto *PackTy = FixedVectorType::get(Int8Ty, OutBytes);
  unsigned MinShiftWidth = 8;
  while (MinShiftWidth < MaxAmt + Info.FieldWidth)
    MinShiftWidth *= 2;
  bool IsReversed = equal(Info.LaneOfField, reverse(seq<unsigned>(NumElts)));
  bool IsTrunc = IsReversed || equal(Info.LaneOfField, seq<unsigned>(NumElts));
  auto *FieldTy = FixedVectorType::get(
      IntegerType::get(SrcTy->getContext(), Info.FieldWidth), NumElts);
  InstructionCost NewCost = InstructionCost::getInvalid();
  ShiftWidth = 0;
  for (unsigned W2 = MinShiftWidth; W2 <= BitWidth; W2 *= 2) {
    auto *ShiftTy = FixedVectorType::get(
        IntegerType::get(SrcTy->getContext(), W2), NumElts);
    unsigned BytesPerLane = W2 / 8;
    unsigned InBytes = NumElts * BytesPerLane;
    SmallVector<int> Mask =
        getBitPackMask(Info, OutBytes, NumElts, BytesPerLane);
    InstructionCost C = TTI.getCastInstrCost(Instruction::BitCast, ResultTy,
                                             PackTy, CCH, CostKind);
    if (IsTrunc) {
      // Fields in lane order, possibly reversed, are a truncation of the
      // shifted lanes. A reversal of byte fields is a bswap.
      if (Info.FieldWidth < W2)
        C += TTI.getCastInstrCost(Instruction::Trunc, FieldTy, ShiftTy,
                                  TTI::CastContextHint::None, CostKind);
      if (IsReversed && Info.FieldWidth == 8) {
        IntrinsicCostAttributes CostAttrs(Intrinsic::bswap, ResultTy,
                                          {ResultTy});
        C += TTI.getIntrinsicInstrCost(CostAttrs, CostKind);
      } else if (IsReversed) {
        C += TTI.getShuffleCost(TargetTransformInfo::SK_Reverse, FieldTy,
                                FieldTy, CostKind);
      }
    } else {
      // A narrowing shuffle is a shuffle at the source width followed by an
      // extract of the low elements.
      auto *SrcByteTy = FixedVectorType::get(Int8Ty, InBytes);
      SmallVector<int> WideMask(Mask);
      WideMask.resize(std::max(InBytes, OutBytes), PoisonMaskElem);
      C += TTI.getShuffleCost(
          is_contained(Info.LaneOfField, BitPackInfo::NoLane)
              ? TargetTransformInfo::SK_PermuteTwoSrc
              : TargetTransformInfo::SK_PermuteSingleSrc,
          FixedVectorType::get(Int8Ty, WideMask.size()), SrcByteTy, CostKind,
          WideMask, /*Index=*/0, /*SubTp=*/nullptr, /*Args=*/{}, CtxI);
      if (InBytes > OutBytes)
        C += TTI.getShuffleCost(TargetTransformInfo::SK_ExtractSubvector,
                                PackTy, SrcByteTy, CostKind, /*Mask=*/{},
                                /*Index=*/0, PackTy, /*Args=*/{}, CtxI);
    }
    if (W2 != BitWidth && W2 != ZExtSrcWidth)
      C += TTI.getCastInstrCost(Instruction::Trunc, ShiftTy, SrcTy, CCH,
                                CostKind);
    if (Info.needsShift())
      C += TTI.getArithmeticInstrCost(Instruction::LShr, ShiftTy, CostKind,
                                      /*Opd1Info=*/{}, ShiftAmtInfo,
                                      /*Args=*/{}, CtxI, TLI);
    if (C.isValid() && (!NewCost.isValid() || C < NewCost)) {
      NewCost = C;
      ShiftWidth = W2;
    }
  }
  return NewCost;
}

InstructionCost getBoolBitmaskCost(const TargetTransformInfo &TTI,
                                   bool NeedMask, Type *NarrowScalarTy,
                                   Type *WideTy, unsigned VF,
                                   ArrayRef<int> PermMask, const Value *Root,
                                   const TTI::TargetCostKind CostKind) {
  auto *NarrowVecTy = cast<VectorType>(getWidenedType(NarrowScalarTy, VF));
  Type *CmpTy = CmpInst::makeCmpResultType(NarrowVecTy);
  auto *MaskTy = IntegerType::get(WideTy->getContext(), VF);
  // The result cast inherits the uses of the reduction root.
  TTI::CastContextHint CCH = getBoolReduxResultCCH(Root);
  const auto *CtxI = cast<Instruction>(Root);
  InstructionCost Cost = 0;
  if (NeedMask)
    Cost += TTI.getArithmeticInstrCost(
        Instruction::And, NarrowVecTy, CostKind,
        {TTI::OK_AnyValue, TTI::OP_None},
        {TTI::OK_NonUniformConstantValue, TTI::OP_None}, {}, CtxI);
  if (!ShuffleVectorInst::isIdentityMask(PermMask, VF))
    Cost += getShuffleCost(TTI, TTI::SK_PermuteSingleSrc, NarrowVecTy, CostKind,
                           PermMask);
  if (!NarrowScalarTy->isIntegerTy(1))
    Cost += TTI.getCmpSelInstrCost(
        Instruction::ICmp, NarrowVecTy, CmpTy, CmpInst::ICMP_NE, CostKind,
        {TTI::OK_AnyValue, TTI::OP_None},
        {TTI::OK_UniformConstantValue, TTI::OP_None});
  // Only the final cast inherits the uses of the reduction root.
  Cost += TTI.getCastInstrCost(
      Instruction::BitCast, MaskTy, CmpTy,
      MaskTy == WideTy ? CCH : TTI::CastContextHint::None, CostKind);
  if (MaskTy != WideTy)
    Cost +=
        TTI.getCastInstrCost(Instruction::ZExt, WideTy, MaskTy, CCH, CostKind);
  return Cost;
}

InstructionCost getNarrowedLeafOpsCost(
    const TargetTransformInfo &TTI,
    const SmallDenseMap<Value *, NarrowedLeafInfo> &NarrowedLeafShifts,
    VectorType *NarrowVecTy, VectorType *WideVecTy, const Instruction *CtxI,
    const TTI::TargetCostKind CostKind) {
  InstructionCost Cost = 0;
  if (any_of(NarrowedLeafShifts,
             [](const auto &P) { return P.second.Shift != 0; }))
    Cost += TTI.getArithmeticInstrCost(
        Instruction::Shl, WideVecTy, CostKind, {TTI::OK_AnyValue, TTI::OP_None},
        {TTI::OK_NonUniformConstantValue, TTI::OP_None}, {}, CtxI);
  if (any_of(NarrowedLeafShifts,
             [](const auto &P) { return !P.second.Mask.isAllOnes(); }))
    Cost += TTI.getArithmeticInstrCost(
        Instruction::And, NarrowVecTy, CostKind,
        {TTI::OK_AnyValue, TTI::OP_None},
        {TTI::OK_NonUniformConstantValue, TTI::OP_None}, {}, CtxI);
  return Cost;
}
} // namespace llvm::slpvectorizer
