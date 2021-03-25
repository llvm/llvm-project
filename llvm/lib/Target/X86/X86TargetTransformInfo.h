//===-- X86TargetTransformInfo.h - X86 specific TTI -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file a TargetTransformInfo::Concept conforming object specific to the
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

namespace llvm {

class InstCombiner;

class X86TTIImpl : public BasicTTIImplBase<X86TTIImpl> {
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
      X86::Feature64Bit,

      // These features don't have any intrinsics or ABI effect.
      X86::FeatureNOPL,
      X86::FeatureCMPXCHG16B,
      X86::FeatureLAHFSAHF,

      // Codegen control options.
      X86::FeatureFast11ByteNOP,
      X86::FeatureFast15ByteNOP,
      X86::FeatureFastBEXTR,
      X86::FeatureFastHorizontalOps,
      X86::FeatureFastLZCNT,
      X86::FeatureFastScalarFSQRT,
      X86::FeatureFastSHLDRotate,
      X86::FeatureFastScalarShiftMasks,
      X86::FeatureFastVectorShiftMasks,
      X86::FeatureFastVariableShuffle,
      X86::FeatureFastVectorFSQRT,
      X86::FeatureLEAForSP,
      X86::FeatureLEAUsesAG,
      X86::FeatureLZCNTFalseDeps,
      X86::FeatureBranchFusion,
      X86::FeatureMacroFusion,
      X86::FeaturePadShortFunctions,
      X86::FeaturePOPCNTFalseDeps,
      X86::FeatureSSEUnalignedMem,
      X86::FeatureSlow3OpsLEA,
      X86::FeatureSlowDivide32,
      X86::FeatureSlowDivide64,
      X86::FeatureSlowIncDec,
      X86::FeatureSlowLEA,
      X86::FeatureSlowPMADDWD,
      X86::FeatureSlowPMULLD,
      X86::FeatureSlowSHLD,
      X86::FeatureSlowTwoMemOps,
      X86::FeatureSlowUAMem16,
      X86::FeaturePreferMaskRegisters,
      X86::FeatureInsertVZEROUPPER,
      X86::FeatureUseGLMDivSqrtCosts,

      // Perf-tuning flags.
      X86::FeatureHasFastGather,
      X86::FeatureSlowUAMem32,

      // Based on whether user set the -mprefer-vector-width command line.
      X86::FeaturePrefer128Bit,
      X86::FeaturePrefer256Bit,

      // CPU name enums. These just follow CPU string.
      X86::ProcIntelAtom,
      X86::ProcIntelSLM,
  };

public:
  explicit X86TTIImpl(const X86TargetMachine *TM, const Function &F)
      : BaseT(TM, F.getParent()->getDataLayout()), ST(TM->getSubtargetImpl(F)),
        TLI(ST->getTargetLowering()) {}

  /// \name Scalar TTI Implementations
  /// @{
  TTI::PopcntSupportKind getPopcntSupport(unsigned TyWidth);

  /// @}

  /// \name Cache TTI Implementation
  /// @{
  llvm::Optional<unsigned> getCacheSize(
    TargetTransformInfo::CacheLevel Level) const override;
  llvm::Optional<unsigned> getCacheAssociativity(
    TargetTransformInfo::CacheLevel Level) const override;
  /// @}

  /// \name Vector TTI Implementations
  /// @{

  unsigned getNumberOfRegisters(unsigned ClassID) const;
  TypeSize getRegisterBitWidth(TargetTransformInfo::RegisterKind K) const;
  unsigned getLoadStoreVecRegBitWidth(unsigned AS) const;
  unsigned getMaxInterleaveFactor(unsigned VF);
  int getArithmeticInstrCost(
      unsigned Opcode, Type *Ty,
      TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput,
      TTI::OperandValueKind Opd1Info = TTI::OK_AnyValue,
      TTI::OperandValueKind Opd2Info = TTI::OK_AnyValue,
      TTI::OperandValueProperties Opd1PropInfo = TTI::OP_None,
      TTI::OperandValueProperties Opd2PropInfo = TTI::OP_None,
      ArrayRef<const Value *> Args = ArrayRef<const Value *>(),
      const Instruction *CxtI = nullptr);
  int getShuffleCost(TTI::ShuffleKind Kind, VectorType *Tp, ArrayRef<int> Mask,
                     int Index, VectorType *SubTp);
  int getCastInstrCost(unsigned Opcode, Type *Dst, Type *Src,
                       TTI::CastContextHint CCH, TTI::TargetCostKind CostKind,
                       const Instruction *I = nullptr);
  int getCmpSelInstrCost(unsigned Opcode, Type *ValTy, Type *CondTy,
                         CmpInst::Predicate VecPred,
                         TTI::TargetCostKind CostKind,
                         const Instruction *I = nullptr);
  int getVectorInstrCost(unsigned Opcode, Type *Val, unsigned Index);
  unsigned getScalarizationOverhead(VectorType *Ty, const APInt &DemandedElts,
                                    bool Insert, bool Extract);
  int getMemoryOpCost(unsigned Opcode, Type *Src, MaybeAlign Alignment,
                      unsigned AddressSpace,
                      TTI::TargetCostKind CostKind,
                      const Instruction *I = nullptr);
  int getMaskedMemoryOpCost(
      unsigned Opcode, Type *Src, Align Alignment, unsigned AddressSpace,
      TTI::TargetCostKind CostKind = TTI::TCK_SizeAndLatency);
  int getGatherScatterOpCost(unsigned Opcode, Type *DataTy, const Value *Ptr,
                             bool VariableMask, Align Alignment,
                             TTI::TargetCostKind CostKind,
                             const Instruction *I);
  int getAddressComputationCost(Type *PtrTy, ScalarEvolution *SE,
                                const SCEV *Ptr);

  Optional<Instruction *> instCombineIntrinsic(InstCombiner &IC,
                                               IntrinsicInst &II) const;
  Optional<Value *>
  simplifyDemandedUseBitsIntrinsic(InstCombiner &IC, IntrinsicInst &II,
                                   APInt DemandedMask, KnownBits &Known,
                                   bool &KnownBitsComputed) const;
  Optional<Value *> simplifyDemandedVectorEltsIntrinsic(
      InstCombiner &IC, IntrinsicInst &II, APInt DemandedElts, APInt &UndefElts,
      APInt &UndefElts2, APInt &UndefElts3,
      std::function<void(Instruction *, unsigned, APInt, APInt &)>
          SimplifyAndSetOp) const;

  unsigned getAtomicMemIntrinsicMaxElementSize() const;

  int getTypeBasedIntrinsicInstrCost(const IntrinsicCostAttributes &ICA,
                                     TTI::TargetCostKind CostKind);
  int getIntrinsicInstrCost(const IntrinsicCostAttributes &ICA,
                            TTI::TargetCostKind CostKind);

  int getArithmeticReductionCost(unsigned Opcode, VectorType *Ty,
                                 bool IsPairwiseForm,
                                 TTI::TargetCostKind CostKind = TTI::TCK_SizeAndLatency);

  int getMinMaxCost(Type *Ty, Type *CondTy, bool IsUnsigned);

  int getMinMaxReductionCost(VectorType *Ty, VectorType *CondTy,
                             bool IsPairwiseForm, bool IsUnsigned,
                             TTI::TargetCostKind CostKind);

  int getInterleavedMemoryOpCost(
      unsigned Opcode, Type *VecTy, unsigned Factor, ArrayRef<unsigned> Indices,
      Align Alignment, unsigned AddressSpace,
      TTI::TargetCostKind CostKind = TTI::TCK_SizeAndLatency,
      bool UseMaskForCond = false, bool UseMaskForGaps = false);
  int getInterleavedMemoryOpCostAVX512(
      unsigned Opcode, FixedVectorType *VecTy, unsigned Factor,
      ArrayRef<unsigned> Indices, Align Alignment, unsigned AddressSpace,
      TTI::TargetCostKind CostKind = TTI::TCK_SizeAndLatency,
      bool UseMaskForCond = false, bool UseMaskForGaps = false);
  int getInterleavedMemoryOpCostAVX2(
      unsigned Opcode, FixedVectorType *VecTy, unsigned Factor,
      ArrayRef<unsigned> Indices, Align Alignment, unsigned AddressSpace,
      TTI::TargetCostKind CostKind = TTI::TCK_SizeAndLatency,
      bool UseMaskForCond = false, bool UseMaskForGaps = false);

  int getIntImmCost(int64_t);

  int getIntImmCost(const APInt &Imm, Type *Ty, TTI::TargetCostKind CostKind);

  unsigned getCFInstrCost(unsigned Opcode, TTI::TargetCostKind CostKind);

  int getIntImmCostInst(unsigned Opcode, unsigned Idx, const APInt &Imm,
                        Type *Ty, TTI::TargetCostKind CostKind,
                        Instruction *Inst = nullptr);
  int getIntImmCostIntrin(Intrinsic::ID IID, unsigned Idx, const APInt &Imm,
                          Type *Ty, TTI::TargetCostKind CostKind);
  bool isLSRCostLess(TargetTransformInfo::LSRCost &C1,
                     TargetTransformInfo::LSRCost &C2);
  bool canMacroFuseCmp();
  bool isLegalMaskedLoad(Type *DataType, Align Alignment);
  bool isLegalMaskedStore(Type *DataType, Align Alignment);
  bool isLegalNTLoad(Type *DataType, Align Alignment);
  bool isLegalNTStore(Type *DataType, Align Alignment);
  bool isLegalMaskedGather(Type *DataType, Align Alignment);
  bool isLegalMaskedScatter(Type *DataType, Align Alignment);
  bool isLegalMaskedExpandLoad(Type *DataType);
  bool isLegalMaskedCompressStore(Type *DataType);
  bool hasDivRemOp(Type *DataType, bool IsSigned);
  bool isFCmpOrdCheaperThanFCmpZero(Type *Ty);
  bool areInlineCompatible(const Function *Caller,
                           const Function *Callee) const;
  bool areFunctionArgsABICompatible(const Function *Caller,
                                    const Function *Callee,
                                    SmallPtrSetImpl<Argument *> &Args) const;
  TTI::MemCmpExpansionOptions enableMemCmpExpansion(bool OptSize,
                                                    bool IsZeroCmp) const;
  bool enableInterleavedAccessVectorization();

private:
  int getGSScalarCost(unsigned Opcode, Type *DataTy, bool VariableMask,
                      Align Alignment, unsigned AddressSpace);
  int getGSVectorCost(unsigned Opcode, Type *DataTy, const Value *Ptr,
                      Align Alignment, unsigned AddressSpace);

  int getGatherOverhead() const;
  int getScatterOverhead() const;

  /// @}
};

} // end namespace llvm

#endif
