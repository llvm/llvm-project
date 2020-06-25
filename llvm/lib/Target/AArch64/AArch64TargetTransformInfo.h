//===- AArch64TargetTransformInfo.h - AArch64 specific TTI ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file a TargetTransformInfo::Concept conforming object specific to the
/// AArch64 target machine. It uses the target's detailed information to
/// provide more precise answers to certain TTI queries, while letting the
/// target independent and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AARCH64_AARCH64TARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_AARCH64_AARCH64TARGETTRANSFORMINFO_H

#include "AArch64.h"
#include "AArch64Subtarget.h"
#include "AArch64TargetMachine.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include <cstdint>

namespace llvm {

class APInt;
class Instruction;
class IntrinsicInst;
class Loop;
class SCEV;
class ScalarEvolution;
class Type;
class Value;
class VectorType;

class AArch64TTIImpl : public BasicTTIImplBase<AArch64TTIImpl> {
  using BaseT = BasicTTIImplBase<AArch64TTIImpl>;
  using TTI = TargetTransformInfo;

  friend BaseT;

  const AArch64Subtarget *ST;
  const AArch64TargetLowering *TLI;

  const AArch64Subtarget *getST() const { return ST; }
  const AArch64TargetLowering *getTLI() const { return TLI; }

  enum MemIntrinsicType {
    VECTOR_LDST_TWO_ELEMENTS,
    VECTOR_LDST_THREE_ELEMENTS,
    VECTOR_LDST_FOUR_ELEMENTS
  };

  bool isWideningInstruction(Type *Ty, unsigned Opcode,
                             ArrayRef<const Value *> Args);

public:
  explicit AArch64TTIImpl(const AArch64TargetMachine *TM, const Function &F)
      : BaseT(TM, F.getParent()->getDataLayout()), ST(TM->getSubtargetImpl(F)),
        TLI(ST->getTargetLowering()) {}

  bool areInlineCompatible(const Function *Caller,
                           const Function *Callee) const;

  /// \name Scalar TTI Implementations
  /// @{

  using BaseT::getIntImmCost;
  int getIntImmCost(int64_t Val);
  int getIntImmCost(const APInt &Imm, Type *Ty, TTI::TargetCostKind CostKind);
  int getIntImmCostInst(unsigned Opcode, unsigned Idx, const APInt &Imm,
                        Type *Ty, TTI::TargetCostKind CostKind);
  int getIntImmCostIntrin(Intrinsic::ID IID, unsigned Idx, const APInt &Imm,
                          Type *Ty, TTI::TargetCostKind CostKind);
  TTI::PopcntSupportKind getPopcntSupport(unsigned TyWidth);

  /// @}

  /// \name Vector TTI Implementations
  /// @{

  bool enableInterleavedAccessVectorization() { return true; }

  unsigned getNumberOfRegisters(unsigned ClassID) const {
    bool Vector = (ClassID == 1);
    if (Vector) {
      if (ST->hasNEON())
        return 32;
      return 0;
    }
    return 31;
  }

  unsigned getRegisterBitWidth(bool Vector) const {
    if (Vector) {
      if (ST->hasSVE())
        return std::max(ST->getMinSVEVectorSizeInBits(), 128u);
      if (ST->hasNEON())
        return 128;
      return 0;
    }
    return 64;
  }

  unsigned getMinVectorRegisterBitWidth() {
    return ST->getMinVectorRegisterBitWidth();
  }

  unsigned getMaxInterleaveFactor(unsigned VF);

  int getCastInstrCost(unsigned Opcode, Type *Dst, Type *Src,
                       TTI::TargetCostKind CostKind,
                       const Instruction *I = nullptr);

  int getExtractWithExtendCost(unsigned Opcode, Type *Dst, VectorType *VecTy,
                               unsigned Index);

  int getVectorInstrCost(unsigned Opcode, Type *Val, unsigned Index);

  int getArithmeticInstrCost(
      unsigned Opcode, Type *Ty,
      TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput,
      TTI::OperandValueKind Opd1Info = TTI::OK_AnyValue,
      TTI::OperandValueKind Opd2Info = TTI::OK_AnyValue,
      TTI::OperandValueProperties Opd1PropInfo = TTI::OP_None,
      TTI::OperandValueProperties Opd2PropInfo = TTI::OP_None,
      ArrayRef<const Value *> Args = ArrayRef<const Value *>(),
      const Instruction *CxtI = nullptr);

  int getAddressComputationCost(Type *Ty, ScalarEvolution *SE, const SCEV *Ptr);

  int getCmpSelInstrCost(unsigned Opcode, Type *ValTy, Type *CondTy,
                         TTI::TargetCostKind CostKind,
                         const Instruction *I = nullptr);

  TTI::MemCmpExpansionOptions enableMemCmpExpansion(bool OptSize,
                                                    bool IsZeroCmp) const;

  int getMemoryOpCost(unsigned Opcode, Type *Src, MaybeAlign Alignment,
                      unsigned AddressSpace,
                      TTI::TargetCostKind CostKind,
                      const Instruction *I = nullptr);

  int getCostOfKeepingLiveOverCall(ArrayRef<Type *> Tys);

  void getUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                               TTI::UnrollingPreferences &UP);

  Value *getOrCreateResultFromMemIntrinsic(IntrinsicInst *Inst,
                                           Type *ExpectedType);

  bool getTgtMemIntrinsic(IntrinsicInst *Inst, MemIntrinsicInfo &Info);

  bool isLegalMaskedLoadStore(Type *DataType, Align Alignment) {
    if (!isa<VectorType>(DataType) || !ST->hasSVE())
      return false;

    Type *Ty = cast<VectorType>(DataType)->getElementType();
    if (Ty->isBFloatTy() || Ty->isHalfTy() ||
        Ty->isFloatTy() || Ty->isDoubleTy())
      return true;

    if (Ty->isIntegerTy(8) || Ty->isIntegerTy(16) ||
        Ty->isIntegerTy(32) || Ty->isIntegerTy(64))
      return true;

    return false;
  }

  bool isLegalMaskedLoad(Type *DataType, Align Alignment) {
    return isLegalMaskedLoadStore(DataType, Alignment);
  }

  bool isLegalMaskedStore(Type *DataType, Align Alignment) {
    return isLegalMaskedLoadStore(DataType, Alignment);
  }

  bool isLegalNTStore(Type *DataType, Align Alignment) {
    // NOTE: The logic below is mostly geared towards LV, which calls it with
    //       vectors with 2 elements. We might want to improve that, if other
    //       users show up.
    // Nontemporal vector stores can be directly lowered to STNP, if the vector
    // can be halved so that each half fits into a register. That's the case if
    // the element type fits into a register and the number of elements is a
    // power of 2 > 1.
    if (auto *DataTypeVTy = dyn_cast<VectorType>(DataType)) {
      unsigned NumElements = DataTypeVTy->getNumElements();
      unsigned EltSize = DataTypeVTy->getElementType()->getScalarSizeInBits();
      return NumElements > 1 && isPowerOf2_64(NumElements) && EltSize >= 8 &&
             EltSize <= 128 && isPowerOf2_64(EltSize);
    }
    return BaseT::isLegalNTStore(DataType, Alignment);
  }

  int getInterleavedMemoryOpCost(unsigned Opcode, Type *VecTy, unsigned Factor,
                                 ArrayRef<unsigned> Indices, unsigned Alignment,
                                 unsigned AddressSpace,
                                 TTI::TargetCostKind CostKind = TTI::TCK_SizeAndLatency,
                                 bool UseMaskForCond = false,
                                 bool UseMaskForGaps = false);

  bool
  shouldConsiderAddressTypePromotion(const Instruction &I,
                                     bool &AllowPromotionWithoutCommonHeader);

  bool shouldExpandReduction(const IntrinsicInst *II) const {
    switch (II->getIntrinsicID()) {
    case Intrinsic::experimental_vector_reduce_v2_fadd:
    case Intrinsic::experimental_vector_reduce_v2_fmul:
      // We don't have legalization support for ordered FP reductions.
      return !II->getFastMathFlags().allowReassoc();

    case Intrinsic::experimental_vector_reduce_fmax:
    case Intrinsic::experimental_vector_reduce_fmin:
      // Lowering asserts that there are no NaNs.
      return !II->getFastMathFlags().noNaNs();

    default:
      // Don't expand anything else, let legalization deal with it.
      return false;
    }
  }

  unsigned getGISelRematGlobalCost() const {
    return 2;
  }

  bool useReductionIntrinsic(unsigned Opcode, Type *Ty,
                             TTI::ReductionFlags Flags) const;

  int getArithmeticReductionCost(unsigned Opcode, VectorType *Ty,
                                 bool IsPairwiseForm,
                                 TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput);

  int getShuffleCost(TTI::ShuffleKind Kind, VectorType *Tp, int Index,
                     VectorType *SubTp);
  /// @}
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AARCH64_AARCH64TARGETTRANSFORMINFO_H
