//===- ARMTargetTransformInfo.h - ARM specific TTI --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file a TargetTransformInfoImplBase conforming object specific to the
/// ARM target machine. It uses the target's detailed information to
/// provide more precise answers to certain TTI queries, while letting the
/// target independent and default TTI implementations handle the rest.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ARM_ARMTARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_ARM_ARMTARGETTRANSFORMINFO_H

#include "ARM.h"
#include "ARMSubtarget.h"
#include "ARMTargetMachine.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Function.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include <optional>

namespace llvm {

class APInt;
class ARMTargetLowering;
class Instruction;
class Loop;
class SCEV;
class ScalarEvolution;
class Type;
class Value;

namespace TailPredication {
enum Mode {
  Disabled = 0,
  EnabledNoReductions,
  Enabled,
  ForceEnabledNoReductions,
  ForceEnabled
};
}

// For controlling conversion of memcpy into Tail Predicated loop.
namespace TPLoop {
enum MemTransfer { ForceDisabled = 0, ForceEnabled, Allow };
}

class ARMTTIImpl final : public BasicTTIImplBase<ARMTTIImpl> {
  using BaseT = BasicTTIImplBase<ARMTTIImpl>;
  using TTI = TargetTransformInfo;

  friend BaseT;

  const ARMSubtarget *ST;
  const ARMTargetLowering *TLI;

  const ARMSubtarget *getST() const { return ST; }
  const ARMTargetLowering *getTLI() const { return TLI; }

public:
  explicit ARMTTIImpl(const ARMBaseTargetMachine *TM, const Function &F)
      : BaseT(TM, F.getDataLayout()), ST(TM->getSubtargetImpl(F)),
        TLI(ST->getTargetLowering()) {}

  bool enableInterleavedAccessVectorization() const override { return true; }

  TTI::AddressingModeKind
  getPreferredAddressingMode(const Loop *L, ScalarEvolution *SE) const override;

  /// Floating-point computation using ARMv8 AArch32 Advanced
  /// SIMD instructions remains unchanged from ARMv7. Only AArch64 SIMD
  /// and Arm MVE are IEEE-754 compliant.
  bool isFPVectorizationPotentiallyUnsafe() const override {
    return !ST->isTargetDarwin() && !ST->hasMVEFloatOps();
  }

  std::optional<Instruction *>
  instCombineIntrinsic(InstCombiner &IC, IntrinsicInst &II) const override;
  std::optional<Value *> simplifyDemandedVectorEltsIntrinsic(
      InstCombiner &IC, IntrinsicInst &II, APInt DemandedElts, APInt &UndefElts,
      APInt &UndefElts2, APInt &UndefElts3,
      std::function<void(Instruction *, unsigned, APInt, APInt &)>
          SimplifyAndSetOp) const override;

  /// \name Scalar TTI Implementations
  /// @{

  InstructionCost getIntImmCodeSizeCost(unsigned Opcode, unsigned Idx,
                                        const APInt &Imm,
                                        Type *Ty) const override;

  using BaseT::getIntImmCost;
  InstructionCost getIntImmCost(const APInt &Imm, Type *Ty,
                                TTI::TargetCostKind CostKind) const override;

  InstructionCost getIntImmCostInst(unsigned Opcode, unsigned Idx,
                                    const APInt &Imm, Type *Ty,
                                    TTI::TargetCostKind CostKind,
                                    Instruction *Inst = nullptr) const override;

  /// @}

  /// \name Vector TTI Implementations
  /// @{

  unsigned getNumberOfRegisters(unsigned ClassID) const override {
    bool Vector = (ClassID == 1);
    if (Vector) {
      if (ST->hasNEON())
        return 16;
      if (ST->hasMVEIntegerOps())
        return 8;
      return 0;
    }

    if (ST->isThumb1Only())
      return 8;
    return 13;
  }

  TypeSize
  getRegisterBitWidth(TargetTransformInfo::RegisterKind K) const override {
    switch (K) {
    case TargetTransformInfo::RGK_Scalar:
      return TypeSize::getFixed(32);
    case TargetTransformInfo::RGK_FixedWidthVector:
      if (ST->hasNEON())
        return TypeSize::getFixed(128);
      if (ST->hasMVEIntegerOps())
        return TypeSize::getFixed(128);
      return TypeSize::getFixed(0);
    case TargetTransformInfo::RGK_ScalableVector:
      return TypeSize::getScalable(0);
    }
    llvm_unreachable("Unsupported register kind");
  }

  unsigned getMaxInterleaveFactor(ElementCount VF,
                                  bool HasUnorderedReductions) const override {
    return ST->getMaxInterleaveFactor();
  }

  bool isProfitableLSRChainElement(Instruction *I) const override;

  bool
  isLegalMaskedLoad(Type *DataTy, Align Alignment, unsigned AddressSpace,
                    TTI::MaskKind MaskKind =
                        TTI::MaskKind::VariableOrConstantMask) const override;

  bool
  isLegalMaskedStore(Type *DataTy, Align Alignment, unsigned AddressSpace,
                     TTI::MaskKind MaskKind =
                         TTI::MaskKind::VariableOrConstantMask) const override {
    return isLegalMaskedLoad(DataTy, Alignment, AddressSpace, MaskKind);
  }

  bool forceScalarizeMaskedGather(VectorType *VTy,
                                  Align Alignment) const override {
    // For MVE, we have a custom lowering pass that will already have custom
    // legalised any gathers that we can lower to MVE intrinsics, and want to
    // expand all the rest. The pass runs before the masked intrinsic lowering
    // pass.
    return true;
  }

  bool forceScalarizeMaskedScatter(VectorType *VTy,
                                   Align Alignment) const override {
    return forceScalarizeMaskedGather(VTy, Alignment);
  }

  bool isLegalMaskedGather(Type *Ty, Align Alignment) const override;

  bool isLegalMaskedScatter(Type *Ty, Align Alignment) const override {
    return isLegalMaskedGather(Ty, Alignment);
  }

  InstructionCost getMemcpyCost(const Instruction *I) const override;

  uint64_t getMaxMemIntrinsicInlineSizeThreshold() const override {
    return ST->getMaxInlineSizeThreshold();
  }

  int getNumMemOps(const IntrinsicInst *I) const;

  InstructionCost
  getShuffleCost(TTI::ShuffleKind Kind, VectorType *DstTy, VectorType *SrcTy,
                 TTI::TargetCostKind CostKind, ArrayRef<int> Mask, int Index,
                 VectorType *SubTp, ArrayRef<const Value *> Args = {},
                 const Instruction *CxtI = nullptr) const override;

  bool preferInLoopReduction(RecurKind Kind, Type *Ty) const override;

  bool preferPredicatedReductionSelect() const override;

  bool shouldExpandReduction(const IntrinsicInst *II) const override {
    return false;
  }

  InstructionCost getCFInstrCost(unsigned Opcode, TTI::TargetCostKind CostKind,
                                 const Instruction *I = nullptr) const override;

  InstructionCost
  getCastInstrCost(unsigned Opcode, Type *Dst, Type *Src,
                   TTI::CastContextHint CCH, TTI::TargetCostKind CostKind,
                   const Instruction *I = nullptr) const override;

  InstructionCost getCmpSelInstrCost(
      unsigned Opcode, Type *ValTy, Type *CondTy, CmpInst::Predicate VecPred,
      TTI::TargetCostKind CostKind,
      TTI::OperandValueInfo Op1Info = {TTI::OK_AnyValue, TTI::OP_None},
      TTI::OperandValueInfo Op2Info = {TTI::OK_AnyValue, TTI::OP_None},
      const Instruction *I = nullptr) const override;

  using BaseT::getVectorInstrCost;
  InstructionCost
  getVectorInstrCost(unsigned Opcode, Type *Ty, TTI::TargetCostKind CostKind,
                     unsigned Index, const Value *Op0, const Value *Op1,
                     TTI::VectorInstrContext VIC =
                         TTI::VectorInstrContext::None) const override;

  InstructionCost
  getAddressComputationCost(Type *Ty, ScalarEvolution *SE, const SCEV *Ptr,
                            TTI::TargetCostKind CostKind) const override;

  InstructionCost getArithmeticInstrCost(
      unsigned Opcode, Type *Ty, TTI::TargetCostKind CostKind,
      TTI::OperandValueInfo Op1Info = {TTI::OK_AnyValue, TTI::OP_None},
      TTI::OperandValueInfo Op2Info = {TTI::OK_AnyValue, TTI::OP_None},
      ArrayRef<const Value *> Args = {},
      const Instruction *CxtI = nullptr) const override;

  InstructionCost getMemoryOpCost(
      unsigned Opcode, Type *Src, Align Alignment, unsigned AddressSpace,
      TTI::TargetCostKind CostKind,
      TTI::OperandValueInfo OpInfo = {TTI::OK_AnyValue, TTI::OP_None},
      const Instruction *I = nullptr) const override;

  InstructionCost
  getMemIntrinsicInstrCost(const MemIntrinsicCostAttributes &MICA,
                           TTI::TargetCostKind CostKind) const override;

  InstructionCost getMaskedMemoryOpCost(const MemIntrinsicCostAttributes &MICA,
                                        TTI::TargetCostKind CostKind) const;

  InstructionCost getInterleavedMemoryOpCost(
      unsigned Opcode, Type *VecTy, unsigned Factor, ArrayRef<unsigned> Indices,
      Align Alignment, unsigned AddressSpace, TTI::TargetCostKind CostKind,
      bool UseMaskForCond = false, bool UseMaskForGaps = false) const override;

  InstructionCost getGatherScatterOpCost(const MemIntrinsicCostAttributes &MICA,
                                         TTI::TargetCostKind CostKind) const;

  InstructionCost
  getArithmeticReductionCost(unsigned Opcode, VectorType *ValTy,
                             std::optional<FastMathFlags> FMF,
                             TTI::TargetCostKind CostKind) const override;
  InstructionCost
  getExtendedReductionCost(unsigned Opcode, bool IsUnsigned, Type *ResTy,
                           VectorType *ValTy, std::optional<FastMathFlags> FMF,
                           TTI::TargetCostKind CostKind) const override;
  InstructionCost
  getMulAccReductionCost(bool IsUnsigned, unsigned RedOpcode, Type *ResTy,
                         VectorType *ValTy,
                         TTI::TargetCostKind CostKind) const override;

  InstructionCost
  getMinMaxReductionCost(Intrinsic::ID IID, VectorType *Ty, FastMathFlags FMF,
                         TTI::TargetCostKind CostKind) const override;

  InstructionCost
  getIntrinsicInstrCost(const IntrinsicCostAttributes &ICA,
                        TTI::TargetCostKind CostKind) const override;

  InstructionCost getPartialReductionCost(
      unsigned Opcode, Type *InputTypeA, Type *InputTypeB, Type *AccumType,
      ElementCount VF, TTI::PartialReductionExtendKind OpAExtend,
      TTI::PartialReductionExtendKind OpBExtend, std::optional<unsigned> BinOp,
      TTI::TargetCostKind CostKind,
      std::optional<FastMathFlags> FMF) const override {
    return InstructionCost::getInvalid();
  }

  /// getScalingFactorCost - Return the cost of the scaling used in
  /// addressing mode represented by AM.
  /// If the AM is supported, the return value must be >= 0.
  /// If the AM is not supported, the return value is an invalid cost.
  InstructionCost getScalingFactorCost(Type *Ty, GlobalValue *BaseGV,
                                       StackOffset BaseOffset, bool HasBaseReg,
                                       int64_t Scale,
                                       unsigned AddrSpace) const override;

  bool maybeLoweredToCall(Instruction &I) const;
  bool isLoweredToCall(const Function *F) const override;
  bool isHardwareLoopProfitable(Loop *L, ScalarEvolution &SE,
                                AssumptionCache &AC, TargetLibraryInfo *LibInfo,
                                HardwareLoopInfo &HWLoopInfo) const override;
  bool preferTailFoldingOverEpilogue(TailFoldingInfo *TFI) const override;
  void getUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                               TTI::UnrollingPreferences &UP,
                               OptimizationRemarkEmitter *ORE) const override;

  TailFoldingStyle getPreferredTailFoldingStyle() const override;

  void getPeelingPreferences(Loop *L, ScalarEvolution &SE,
                             TTI::PeelingPreferences &PP) const override;
  bool shouldBuildLookupTablesForConstant(Constant *C) const override {
    // In the ROPI and RWPI relocation models we can't have pointers to global
    // variables or functions in constant data, so don't convert switches to
    // lookup tables if any of the values would need relocation.
    if (ST->isROPI() || ST->isRWPI())
      return !C->needsDynamicRelocation();

    return true;
  }

  bool shouldConsiderVectorizationRegPressure() const override;

  bool hasArmWideBranch(bool Thumb) const override;

  bool isProfitableToSinkOperands(Instruction *I,
                                  SmallVectorImpl<Use *> &Ops) const override;

  unsigned getNumBytesToPadGlobalArray(unsigned Size,
                                       Type *ArrayType) const override;

  /// @}
};

/// isVREVMask - Check if a vector shuffle corresponds to a VREV
/// instruction with the specified blocksize.  (The order of the elements
/// within each block of the vector is reversed.)
inline bool isVREVMask(ArrayRef<int> M, EVT VT, unsigned BlockSize) {
  assert((BlockSize == 16 || BlockSize == 32 || BlockSize == 64) &&
         "Only possible block sizes for VREV are: 16, 32, 64");

  unsigned EltSz = VT.getScalarSizeInBits();
  if (EltSz != 8 && EltSz != 16 && EltSz != 32)
    return false;

  unsigned BlockElts = M[0] + 1;
  // If the first shuffle index is UNDEF, be optimistic.
  if (M[0] < 0)
    BlockElts = BlockSize / EltSz;

  if (BlockSize <= EltSz || BlockSize != BlockElts * EltSz)
    return false;

  for (unsigned i = 0, e = M.size(); i < e; ++i) {
    if (M[i] < 0)
      continue; // ignore UNDEF indices
    if ((unsigned)M[i] != (i - i % BlockElts) + (BlockElts - 1 - i % BlockElts))
      return false;
  }

  return true;
}

inline unsigned SelectPairHalf(unsigned Elements, ArrayRef<int> Mask,
                               unsigned Index) {
  if (Mask.size() == Elements * 2)
    return Index / Elements;
  return Mask[Index] == 0 ? 0 : 1;
}

// Checks whether the shuffle mask represents a vector transpose (VTRN) by
// checking that pairs of elements in the shuffle mask represent the same index
// in each vector, incrementing the expected index by 2 at each step.
// e.g. For v1,v2 of type v4i32 a valid shuffle mask is: [0, 4, 2, 6]
//  v1={a,b,c,d} => x=shufflevector v1, v2 shufflemask => x={a,e,c,g}
//  v2={e,f,g,h}
// WhichResult gives the offset for each element in the mask based on which
// of the two results it belongs to.
//
// The transpose can be represented either as:
// result1 = shufflevector v1, v2, result1_shuffle_mask
// result2 = shufflevector v1, v2, result2_shuffle_mask
// where v1/v2 and the shuffle masks have the same number of elements
// (here WhichResult (see below) indicates which result is being checked)
//
// or as:
// results = shufflevector v1, v2, shuffle_mask
// where both results are returned in one vector and the shuffle mask has twice
// as many elements as v1/v2 (here WhichResult will always be 0 if true) here we
// want to check the low half and high half of the shuffle mask as if it were
// the other case
inline bool isVTRNMask(ArrayRef<int> M, EVT VT, unsigned &WhichResult) {
  unsigned EltSz = VT.getScalarSizeInBits();
  if (EltSz == 64)
    return false;

  unsigned NumElts = VT.getVectorNumElements();
  if ((M.size() != NumElts && M.size() != NumElts * 2) || NumElts % 2 != 0)
    return false;

  // If the mask is twice as long as the input vector then we need to check the
  // upper and lower parts of the mask with a matching value for WhichResult
  // FIXME: A mask with only even values will be rejected in case the first
  // element is undefined, e.g. [-1, 4, 2, 6] will be rejected, because only
  // M[0] is used to determine WhichResult
  for (unsigned i = 0; i < M.size(); i += NumElts) {
    WhichResult = SelectPairHalf(NumElts, M, i);
    for (unsigned j = 0; j < NumElts; j += 2) {
      if ((M[i + j] >= 0 && (unsigned)M[i + j] != j + WhichResult) ||
          (M[i + j + 1] >= 0 &&
           (unsigned)M[i + j + 1] != j + NumElts + WhichResult))
        return false;
    }
  }

  if (M.size() == NumElts * 2)
    WhichResult = 0;

  return true;
}

/// isVTRN_v_undef_Mask - Special case of isVTRNMask for canonical form of
/// "vector_shuffle v, v", i.e., "vector_shuffle v, undef".
/// Mask is e.g., <0, 0, 2, 2> instead of <0, 4, 2, 6>.
inline bool isVTRN_v_undef_Mask(ArrayRef<int> M, EVT VT,
                                unsigned &WhichResult) {
  unsigned EltSz = VT.getScalarSizeInBits();
  if (EltSz == 64)
    return false;

  unsigned NumElts = VT.getVectorNumElements();
  if ((M.size() != NumElts && M.size() != NumElts * 2) || NumElts % 2 != 0)
    return false;

  for (unsigned i = 0; i < M.size(); i += NumElts) {
    WhichResult = SelectPairHalf(NumElts, M, i);
    for (unsigned j = 0; j < NumElts; j += 2) {
      if ((M[i + j] >= 0 && (unsigned)M[i + j] != j + WhichResult) ||
          (M[i + j + 1] >= 0 && (unsigned)M[i + j + 1] != j + WhichResult))
        return false;
    }
  }

  if (M.size() == NumElts * 2)
    WhichResult = 0;

  return true;
}

// Checks whether the shuffle mask represents a vector unzip (VUZP) by checking
// that the mask elements are either all even and in steps of size 2 or all odd
// and in steps of size 2.
// e.g. For v1,v2 of type v4i32 a valid shuffle mask is: [0, 2, 4, 6]
//  v1={a,b,c,d} => x=shufflevector v1, v2 shufflemask => x={a,c,e,g}
//  v2={e,f,g,h}
// Requires similar checks to that of isVTRNMask with
// respect the how results are returned.
inline bool isVUZPMask(ArrayRef<int> M, EVT VT, unsigned &WhichResult) {
  unsigned EltSz = VT.getScalarSizeInBits();
  if (EltSz == 64)
    return false;

  unsigned NumElts = VT.getVectorNumElements();
  if (M.size() != NumElts && M.size() != NumElts * 2)
    return false;

  for (unsigned i = 0; i < M.size(); i += NumElts) {
    WhichResult = SelectPairHalf(NumElts, M, i);
    for (unsigned j = 0; j < NumElts; ++j) {
      if (M[i + j] >= 0 && (unsigned)M[i + j] != 2 * j + WhichResult)
        return false;
    }
  }

  if (M.size() == NumElts * 2)
    WhichResult = 0;

  // VUZP.32 for 64-bit vectors is a pseudo-instruction alias for VTRN.32.
  if (VT.is64BitVector() && EltSz == 32)
    return false;

  return true;
}

/// isVUZP_v_undef_Mask - Special case of isVUZPMask for canonical form of
/// "vector_shuffle v, v", i.e., "vector_shuffle v, undef".
/// Mask is e.g., <0, 2, 0, 2> instead of <0, 2, 4, 6>,
inline bool isVUZP_v_undef_Mask(ArrayRef<int> M, EVT VT,
                                unsigned &WhichResult) {
  unsigned EltSz = VT.getScalarSizeInBits();
  if (EltSz == 64)
    return false;

  unsigned NumElts = VT.getVectorNumElements();
  if (M.size() != NumElts && M.size() != NumElts * 2)
    return false;

  unsigned Half = NumElts / 2;
  for (unsigned i = 0; i < M.size(); i += NumElts) {
    WhichResult = SelectPairHalf(NumElts, M, i);
    for (unsigned j = 0; j < NumElts; j += Half) {
      unsigned Idx = WhichResult;
      for (unsigned k = 0; k < Half; ++k) {
        int MIdx = M[i + j + k];
        if (MIdx >= 0 && (unsigned)MIdx != Idx)
          return false;
        Idx += 2;
      }
    }
  }

  if (M.size() == NumElts * 2)
    WhichResult = 0;

  // VUZP.32 for 64-bit vectors is a pseudo-instruction alias for VTRN.32.
  if (VT.is64BitVector() && EltSz == 32)
    return false;

  return true;
}

// Checks whether the shuffle mask represents a vector zip (VZIP) by checking
// that pairs of elements of the shufflemask represent the same index in each
// vector incrementing sequentially through the vectors.
// e.g. For v1,v2 of type v4i32 a valid shuffle mask is: [0, 4, 1, 5]
//  v1={a,b,c,d} => x=shufflevector v1, v2 shufflemask => x={a,e,b,f}
//  v2={e,f,g,h}
// Requires similar checks to that of isVTRNMask with respect the how results
// are returned.
inline bool isVZIPMask(ArrayRef<int> M, EVT VT, unsigned &WhichResult) {
  unsigned EltSz = VT.getScalarSizeInBits();
  if (EltSz == 64)
    return false;

  unsigned NumElts = VT.getVectorNumElements();
  if ((M.size() != NumElts && M.size() != NumElts * 2) || NumElts % 2 != 0)
    return false;

  for (unsigned i = 0; i < M.size(); i += NumElts) {
    WhichResult = SelectPairHalf(NumElts, M, i);
    unsigned Idx = WhichResult * NumElts / 2;
    for (unsigned j = 0; j < NumElts; j += 2) {
      if ((M[i + j] >= 0 && (unsigned)M[i + j] != Idx) ||
          (M[i + j + 1] >= 0 && (unsigned)M[i + j + 1] != Idx + NumElts))
        return false;
      Idx += 1;
    }
  }

  if (M.size() == NumElts * 2)
    WhichResult = 0;

  // VZIP.32 for 64-bit vectors is a pseudo-instruction alias for VTRN.32.
  if (VT.is64BitVector() && EltSz == 32)
    return false;

  return true;
}

/// isVZIP_v_undef_Mask - Special case of isVZIPMask for canonical form of
/// "vector_shuffle v, v", i.e., "vector_shuffle v, undef".
/// Mask is e.g., <0, 0, 1, 1> instead of <0, 4, 1, 5>.
inline bool isVZIP_v_undef_Mask(ArrayRef<int> M, EVT VT,
                                unsigned &WhichResult) {
  unsigned EltSz = VT.getScalarSizeInBits();
  if (EltSz == 64)
    return false;

  unsigned NumElts = VT.getVectorNumElements();
  if ((M.size() != NumElts && M.size() != NumElts * 2) || NumElts % 2 != 0)
    return false;

  for (unsigned i = 0; i < M.size(); i += NumElts) {
    WhichResult = SelectPairHalf(NumElts, M, i);
    unsigned Idx = WhichResult * NumElts / 2;
    for (unsigned j = 0; j < NumElts; j += 2) {
      if ((M[i + j] >= 0 && (unsigned)M[i + j] != Idx) ||
          (M[i + j + 1] >= 0 && (unsigned)M[i + j + 1] != Idx))
        return false;
      Idx += 1;
    }
  }

  if (M.size() == NumElts * 2)
    WhichResult = 0;

  // VZIP.32 for 64-bit vectors is a pseudo-instruction alias for VTRN.32.
  if (VT.is64BitVector() && EltSz == 32)
    return false;

  return true;
}

} // end namespace llvm

#endif // LLVM_LIB_TARGET_ARM_ARMTARGETTRANSFORMINFO_H
