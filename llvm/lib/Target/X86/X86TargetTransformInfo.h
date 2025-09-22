//===-- X86TargetTransformInfo.h - X86 specific TTI -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file a TargetTransformInfoImplBase conforming object specific to the
/// X86 target machine. It uses the target's detailed information to
/// provide more precise answers to certain TTI queries, while letting the
/// target independent and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_X86_X86TARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_X86_X86TARGETTRANSFORMINFO_H

#include "X86TargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include <optional>

namespace llvm {

class InstCombiner;

class X86TTIImpl final : public BasicTTIImplBase<X86TTIImpl> {
  typedef BasicTTIImplBase<X86TTIImpl> BaseT;
  typedef TargetTransformInfo TTI;
  friend BaseT;

  const X86Subtarget *ST;
  const X86TargetLowering *TLI;

  const X86Subtarget *getST() const { return ST; }
  const X86TargetLowering *getTLI() const { return TLI; }

  const FeatureBitset InlineFeatureIgnoreList = {
      // This indicates the CPU is 64 bit capable not that we are in 64-bit
      // mode.
      X86::FeatureX86_64,

      // These features don't have any intrinsics or ABI effect.
      X86::FeatureNOPL,
      X86::FeatureCX16,
      X86::FeatureLAHFSAHF64,

      // Some older targets can be setup to fold unaligned loads.
      X86::FeatureSSEUnalignedMem,

      // Codegen control options.
      X86::TuningFast11ByteNOP,
      X86::TuningFast15ByteNOP,
      X86::TuningFastBEXTR,
      X86::TuningFastHorizontalOps,
      X86::TuningFastLZCNT,
      X86::TuningFastScalarFSQRT,
      X86::TuningFastSHLDRotate,
      X86::TuningFastScalarShiftMasks,
      X86::TuningFastVectorShiftMasks,
      X86::TuningFastVariableCrossLaneShuffle,
      X86::TuningFastVariablePerLaneShuffle,
      X86::TuningFastVectorFSQRT,
      X86::TuningLEAForSP,
      X86::TuningLEAUsesAG,
      X86::TuningLZCNTFalseDeps,
      X86::TuningBranchFusion,
      X86::TuningMacroFusion,
      X86::TuningPadShortFunctions,
      X86::TuningPOPCNTFalseDeps,
      X86::TuningMULCFalseDeps,
      X86::TuningPERMFalseDeps,
      X86::TuningRANGEFalseDeps,
      X86::TuningGETMANTFalseDeps,
      X86::TuningMULLQFalseDeps,
      X86::TuningSlow3OpsLEA,
      X86::TuningSlowDivide32,
      X86::TuningSlowDivide64,
      X86::TuningSlowIncDec,
      X86::TuningSlowLEA,
      X86::TuningSlowPMADDWD,
      X86::TuningSlowPMULLD,
      X86::TuningSlowSHLD,
      X86::TuningSlowTwoMemOps,
      X86::TuningSlowUAMem16,
      X86::TuningPreferMaskRegisters,
      X86::TuningInsertVZEROUPPER,
      X86::TuningUseSLMArithCosts,
      X86::TuningUseGLMDivSqrtCosts,
      X86::TuningNoDomainDelay,
      X86::TuningNoDomainDelayMov,
      X86::TuningNoDomainDelayShuffle,
      X86::TuningNoDomainDelayBlend,
      X86::TuningPreferShiftShuffle,
      X86::TuningFastImmVectorShift,
      X86::TuningFastDPWSSD,

      // Perf-tuning flags.
      X86::TuningFastGather,
      X86::TuningSlowUAMem32,
      X86::TuningAllowLight256Bit,

      // Based on whether user set the -mprefer-vector-width command line.
      X86::TuningPrefer128Bit,
      X86::TuningPrefer256Bit,

      // CPU name enums. These just follow CPU string.
      X86::ProcIntelAtom
  };

public:
  explicit X86TTIImpl(const X86TargetMachine *TM, const Function &F)
      : BaseT(TM, F.getDataLayout()), ST(TM->getSubtargetImpl(F)),
        TLI(ST->getTargetLowering()) {}

  /// \name Scalar TTI Implementations
  /// @{
  TTI::PopcntSupportKind getPopcntSupport(unsigned TyWidth) const override;

  /// @}

  /// \name Cache TTI Implementation
  /// @{
  std::optional<unsigned> getCacheSize(
    TargetTransformInfo::CacheLevel Level) const override;
  std::optional<unsigned> getCacheAssociativity(
    TargetTransformInfo::CacheLevel Level) const override;
  /// @}

  /// \name Vector TTI Implementations
  /// @{

  unsigned getNumberOfRegisters(unsigned ClassID) const override;
  unsigned getRegisterClassForType(bool Vector, Type *Ty) const override;
  bool hasConditionalLoadStoreForType(Type *Ty, bool IsStore) const override;
  TypeSize
  getRegisterBitWidth(TargetTransformInfo::RegisterKind K) const override;
  unsigned getLoadStoreVecRegBitWidth(unsigned AS) const override;
  unsigned getMaxInterleaveFactor(ElementCount VF) const override;
  InstructionCost getArithmeticInstrCost(
      unsigned Opcode, Type *Ty, TTI::TargetCostKind CostKind,
      TTI::OperandValueInfo Op1Info = {TTI::OK_AnyValue, TTI::OP_None},
      TTI::OperandValueInfo Op2Info = {TTI::OK_AnyValue, TTI::OP_None},
      ArrayRef<const Value *> Args = {},
      const Instruction *CxtI = nullptr) const override;
  InstructionCost getAltInstrCost(VectorType *VecTy, unsigned Opcode0,
                                  unsigned Opcode1,
                                  const SmallBitVector &OpcodeMask,
                                  TTI::TargetCostKind CostKind) const override;

  InstructionCost
  getShuffleCost(TTI::ShuffleKind Kind, VectorType *DstTy, VectorType *SrcTy,
                 ArrayRef<int> Mask, TTI::TargetCostKind CostKind, int Index,
                 VectorType *SubTp, ArrayRef<const Value *> Args = {},
                 const Instruction *CxtI = nullptr) const override;
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
  InstructionCost getVectorInstrCost(unsigned Opcode, Type *Val,
                                     TTI::TargetCostKind CostKind,
                                     unsigned Index, const Value *Op0,
                                     const Value *Op1) const override;
  InstructionCost getScalarizationOverhead(
      VectorType *Ty, const APInt &DemandedElts, bool Insert, bool Extract,
      TTI::TargetCostKind CostKind, bool ForPoisonSrc = true,
      ArrayRef<Value *> VL = {}) const override;
  InstructionCost
  getReplicationShuffleCost(Type *EltTy, int ReplicationFactor, int VF,
                            const APInt &DemandedDstElts,
                            TTI::TargetCostKind CostKind) const override;
  InstructionCost getMemoryOpCost(
      unsigned Opcode, Type *Src, Align Alignment, unsigned AddressSpace,
      TTI::TargetCostKind CostKind,
      TTI::OperandValueInfo OpInfo = {TTI::OK_AnyValue, TTI::OP_None},
      const Instruction *I = nullptr) const override;
  InstructionCost
  getMaskedMemoryOpCost(unsigned Opcode, Type *Src, Align Alignment,
                        unsigned AddressSpace,
                        TTI::TargetCostKind CostKind) const override;
  InstructionCost getGatherScatterOpCost(unsigned Opcode, Type *DataTy,
                                         const Value *Ptr, bool VariableMask,
                                         Align Alignment,
                                         TTI::TargetCostKind CostKind,
                                         const Instruction *I) const override;
  InstructionCost
  getPointersChainCost(ArrayRef<const Value *> Ptrs, const Value *Base,
                       const TTI::PointersChainInfo &Info, Type *AccessTy,
                       TTI::TargetCostKind CostKind) const override;
  InstructionCost
  getAddressComputationCost(Type *PtrTy, ScalarEvolution *SE, const SCEV *Ptr,
                            TTI::TargetCostKind CostKind) const override;

  std::optional<Instruction *>
  instCombineIntrinsic(InstCombiner &IC, IntrinsicInst &II) const override;
  std::optional<Value *>
  simplifyDemandedUseBitsIntrinsic(InstCombiner &IC, IntrinsicInst &II,
                                   APInt DemandedMask, KnownBits &Known,
                                   bool &KnownBitsComputed) const override;
  std::optional<Value *> simplifyDemandedVectorEltsIntrinsic(
      InstCombiner &IC, IntrinsicInst &II, APInt DemandedElts, APInt &UndefElts,
      APInt &UndefElts2, APInt &UndefElts3,
      std::function<void(Instruction *, unsigned, APInt, APInt &)>
          SimplifyAndSetOp) const override;

  unsigned getAtomicMemIntrinsicMaxElementSize() const override;

  InstructionCost
  getIntrinsicInstrCost(const IntrinsicCostAttributes &ICA,
                        TTI::TargetCostKind CostKind) const override;

  InstructionCost
  getArithmeticReductionCost(unsigned Opcode, VectorType *Ty,
                             std::optional<FastMathFlags> FMF,
                             TTI::TargetCostKind CostKind) const override;

  InstructionCost getMinMaxCost(Intrinsic::ID IID, Type *Ty,
                                TTI::TargetCostKind CostKind,
                                FastMathFlags FMF) const;

  InstructionCost
  getMinMaxReductionCost(Intrinsic::ID IID, VectorType *Ty, FastMathFlags FMF,
                         TTI::TargetCostKind CostKind) const override;

  InstructionCost getInterleavedMemoryOpCost(
      unsigned Opcode, Type *VecTy, unsigned Factor, ArrayRef<unsigned> Indices,
      Align Alignment, unsigned AddressSpace, TTI::TargetCostKind CostKind,
      bool UseMaskForCond = false, bool UseMaskForGaps = false) const override;
  InstructionCost getInterleavedMemoryOpCostAVX512(
      unsigned Opcode, FixedVectorType *VecTy, unsigned Factor,
      ArrayRef<unsigned> Indices, Align Alignment, unsigned AddressSpace,
      TTI::TargetCostKind CostKind, bool UseMaskForCond = false,
      bool UseMaskForGaps = false) const;

  InstructionCost getIntImmCost(int64_t) const;

  InstructionCost getIntImmCost(const APInt &Imm, Type *Ty,
                                TTI::TargetCostKind CostKind) const override;

  InstructionCost getCFInstrCost(unsigned Opcode, TTI::TargetCostKind CostKind,
                                 const Instruction *I = nullptr) const override;

  InstructionCost getIntImmCostInst(unsigned Opcode, unsigned Idx,
                                    const APInt &Imm, Type *Ty,
                                    TTI::TargetCostKind CostKind,
                                    Instruction *Inst = nullptr) const override;
  InstructionCost
  getIntImmCostIntrin(Intrinsic::ID IID, unsigned Idx, const APInt &Imm,
                      Type *Ty, TTI::TargetCostKind CostKind) const override;
  /// Return the cost of the scaling factor used in the addressing
  /// mode represented by AM for this target, for a load/store
  /// of the specified type.
  /// If the AM is supported, the return value must be >= 0.
  /// If the AM is not supported, it returns an invalid cost.
  InstructionCost getScalingFactorCost(Type *Ty, GlobalValue *BaseGV,
                                       StackOffset BaseOffset, bool HasBaseReg,
                                       int64_t Scale,
                                       unsigned AddrSpace) const override;

  bool isLSRCostLess(const TargetTransformInfo::LSRCost &C1,
                     const TargetTransformInfo::LSRCost &C2) const override;
  bool canMacroFuseCmp() const override;
  bool isLegalMaskedLoad(Type *DataType, Align Alignment,
                         unsigned AddressSpace) const override;
  bool isLegalMaskedStore(Type *DataType, Align Alignment,
                          unsigned AddressSpace) const override;
  bool isLegalNTLoad(Type *DataType, Align Alignment) const override;
  bool isLegalNTStore(Type *DataType, Align Alignment) const override;
  bool isLegalBroadcastLoad(Type *ElementTy,
                            ElementCount NumElements) const override;
  bool forceScalarizeMaskedGather(VectorType *VTy,
                                  Align Alignment) const override;
  bool forceScalarizeMaskedScatter(VectorType *VTy,
                                   Align Alignment) const override {
    return forceScalarizeMaskedGather(VTy, Alignment);
  }
  bool isLegalMaskedGatherScatter(Type *DataType, Align Alignment) const;
  bool isLegalMaskedGather(Type *DataType, Align Alignment) const override;
  bool isLegalMaskedScatter(Type *DataType, Align Alignment) const override;
  bool isLegalMaskedExpandLoad(Type *DataType, Align Alignment) const override;
  bool isLegalMaskedCompressStore(Type *DataType,
                                  Align Alignment) const override;
  bool isLegalAltInstr(VectorType *VecTy, unsigned Opcode0, unsigned Opcode1,
                       const SmallBitVector &OpcodeMask) const override;
  bool hasDivRemOp(Type *DataType, bool IsSigned) const override;
  bool isExpensiveToSpeculativelyExecute(const Instruction *I) const override;
  bool isFCmpOrdCheaperThanFCmpZero(Type *Ty) const override;
  bool areInlineCompatible(const Function *Caller,
                           const Function *Callee) const override;
  bool areTypesABICompatible(const Function *Caller, const Function *Callee,
                             const ArrayRef<Type *> &Type) const override;

  uint64_t getMaxMemIntrinsicInlineSizeThreshold() const override {
    return ST->getMaxInlineSizeThreshold();
  }

  TTI::MemCmpExpansionOptions
  enableMemCmpExpansion(bool OptSize, bool IsZeroCmp) const override;
  bool preferAlternateOpcodeVectorization() const override { return false; }
  bool prefersVectorizedAddressing() const override;
  bool supportsEfficientVectorElementLoadStore() const override;
  bool enableInterleavedAccessVectorization() const override;

  InstructionCost getBranchMispredictPenalty() const override;

  bool isProfitableToSinkOperands(Instruction *I,
                                  SmallVectorImpl<Use *> &Ops) const override;

  bool isVectorShiftByScalarCheap(Type *Ty) const override;

  unsigned getStoreMinimumVF(unsigned VF, Type *ScalarMemTy,
                             Type *ScalarValTy) const override;

private:
  bool supportsGather() const;
  InstructionCost getGSVectorCost(unsigned Opcode, TTI::TargetCostKind CostKind,
                                  Type *DataTy, const Value *Ptr,
                                  Align Alignment, unsigned AddressSpace) const;

  int getGatherOverhead() const;
  int getScatterOverhead() const;

  /// @}
};

} // end namespace llvm

#endif
