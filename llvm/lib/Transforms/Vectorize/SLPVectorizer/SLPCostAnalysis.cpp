//===- SLPCostAnalysis.cpp - SLP Vectorizer free cost helpers -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SLPCostAnalysis.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"

#include <utility>

using namespace llvm;

namespace llvm::slpvectorizer {

InstructionCost getShuffleCost(const TargetTransformInfo &TTI,
                               TTI::ShuffleKind Kind, VectorType *Tp,
                               ArrayRef<int> Mask, TTI::TargetCostKind CostKind,
                               int Index, VectorType *SubTp,
                               ArrayRef<const Value *> Args) {
  VectorType *DstTy = Tp;
  if (!Mask.empty())
    DstTy = FixedVectorType::get(Tp->getScalarType(), Mask.size());

  if (Kind != TTI::SK_PermuteTwoSrc)
    return TTI.getShuffleCost(Kind, DstTy, Tp, Mask, CostKind, Index, SubTp,
                              Args);
  int NumSrcElts = Tp->getElementCount().getKnownMinValue();
  int NumSubElts;
  if (Mask.size() > 2 && ShuffleVectorInst::isInsertSubvectorMask(
                             Mask, NumSrcElts, NumSubElts, Index)) {
    if (Index + NumSubElts > NumSrcElts &&
        Index + NumSrcElts <= static_cast<int>(Mask.size()))
      return TTI.getShuffleCost(TTI::SK_InsertSubvector, DstTy, Tp, Mask,
                                TTI::TCK_RecipThroughput, Index, Tp);
  }
  return TTI.getShuffleCost(Kind, DstTy, Tp, Mask, CostKind, Index, SubTp,
                            Args);
}

std::pair<InstructionCost, InstructionCost>
getGEPCosts(const TargetTransformInfo &TTI, ArrayRef<Value *> Ptrs,
            Value *BasePtr, unsigned Opcode, TTI::TargetCostKind CostKind,
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
                               BaseGEP->getPointerOperand(), Indices, VecTy,
                               CostKind);
    }
  }

  return std::make_pair(ScalarCost, VecCost);
}

InstructionCost getBlendedLoadCost(const TargetTransformInfo &TTI, Type *VecTy,
                                   Align Alignment, unsigned AddressSpace,
                                   TTI::TargetCostKind CostKind) {
  Type *CmpTy = CmpInst::makeCmpResultType(VecTy);
  return 2 * TTI.getMemIntrinsicInstrCost(
                 MemIntrinsicCostAttributes(Intrinsic::masked_load, VecTy,
                                            Alignment, AddressSpace),
                 CostKind) +
         TTI.getArithmeticInstrCost(Instruction::Xor, CmpTy, CostKind) +
         TTI.getCmpSelInstrCost(Instruction::Select, VecTy, CmpTy,
                                CmpInst::BAD_ICMP_PREDICATE, CostKind);
}

} // namespace llvm::slpvectorizer
