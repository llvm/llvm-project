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

#include "X86.h"
#include "X86TargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/CodeGen/TargetLowering.h"

namespace llvm {

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
      X86::FeatureMergeToThreeWayBranch,
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
    TargetTransformInfo::CacheLevel Level) const;
  llvm::Optional<unsigned> getCacheAssociativity(
    TargetTransformInfo::CacheLevel Level) const;
  /// @}

  /// \name Vector TTI Implementations
  /// @{

  unsigned getNumberOfRegisters(unsigned ClassID) const;
  unsigned getRegisterBitWidth(bool Vector) const;
  unsigned getLoadStoreVecRegBitWidth(unsigned AS) const;
  unsigned getMaxInterleaveFactor(unsigned VF);
  int getArithmeticInstrCost(
      unsigned Opcode, Type *Ty,
      TTI::OperandValueKind Opd1Info = TTI::OK_AnyValue,
      TTI::OperandValueKind Opd2Info = TTI::OK_AnyValue,
      TTI::OperandValueProperties Opd1PropInfo = TTI::OP_None,
      TTI::OperandValueProperties Opd2PropInfo = TTI::OP_None,
      ArrayRef<const Value *> Args = ArrayRef<const Value *>(),
      const Instruction *CxtI = nullptr);
  int getShuffleCost(TTI::ShuffleKind Kind, Type *Tp, int Index, Type *SubTp);
  int getCastInstrCost(unsigned Opcode, Type *Dst, Type *Src,
                       const Instruction *I = nullptr);
  int getCmpSelInstrCost(unsigned Opcode, Type *ValTy, Type *CondTy,
                         const Instruction *I = nullptr);
  int getVectorInstrCost(unsigned Opcode, Type *Val, unsigned Index);
  int getMemoryOpCost(unsigned Opcode, Type *Src, MaybeAlign Alignment,
                      unsigned AddressSpace, const Instruction *I = nullptr);
  int getMaskedMemoryOpCost(unsigned Opcode, Type *Src, unsigned Alignment,
                            unsigned AddressSpace);
  int getGatherScatterOpCost(unsigned Opcode, Type *DataTy, Value *Ptr,
                             bool VariableMask, unsigned Alignment,
                             const Instruction *I);
  int getAddressComputationCost(Type *PtrTy, ScalarEvolution *SE,
                                const SCEV *Ptr);

  unsigned getAtomicMemIntrinsicMaxElementSize() const;

  int getIntrinsicInstrCost(Intrinsic::ID IID, Type *RetTy,
                            ArrayRef<Type *> Tys, FastMathFlags FMF,
                            unsigned ScalarizationCostPassed = UINT_MAX,
                            const Instruction *I = nullptr);
  int getIntrinsicInstrCost(Intrinsic::ID IID, Type *RetTy,
                            ArrayRef<Value *> Args, FastMathFlags FMF,
                            unsigned VF = 1, const Instruction *I = nullptr);

  int getArithmeticReductionCost(unsigned Opcode, Type *Ty,
                                 bool IsPairwiseForm);

  int getMinMaxReductionCost(Type *Ty, Type *CondTy, bool IsPairwiseForm,
                             bool IsUnsigned);

  int getInterleavedMemoryOpCost(unsigned Opcode, Type *VecTy,
                                 unsigned Factor, ArrayRef<unsigned> Indices,
                                 unsigned Alignment, unsigned AddressSpace,
                                 bool UseMaskForCond = false,
                                 bool UseMaskForGaps = false);
  int getInterleavedMemoryOpCostAVX512(unsigned Opcode, Type *VecTy,
                                 unsigned Factor, ArrayRef<unsigned> Indices,
                                 unsigned Alignment, unsigned AddressSpace,
                                 bool UseMaskForCond = false,
                                 bool UseMaskForGaps = false);
  int getInterleavedMemoryOpCostAVX2(unsigned Opcode, Type *VecTy,
                                 unsigned Factor, ArrayRef<unsigned> Indices,
                                 unsigned Alignment, unsigned AddressSpace,
                                 bool UseMaskForCond = false,
                                 bool UseMaskForGaps = false);

  int getIntImmCost(int64_t);

  int getIntImmCost(const APInt &Imm, Type *Ty);

  unsigned getUserCost(const User *U, ArrayRef<const Value *> Operands);

  int getIntImmCostInst(unsigned Opcode, unsigned Idx, const APInt &Imm, Type *Ty);
  int getIntImmCostIntrin(Intrinsic::ID IID, unsigned Idx, const APInt &Imm,
                          Type *Ty);
  bool isLSRCostLess(TargetTransformInfo::LSRCost &C1,
                     TargetTransformInfo::LSRCost &C2);
  bool canMacroFuseCmp();
  bool isLegalMaskedLoad(Type *DataType, MaybeAlign Alignment);
  bool isLegalMaskedStore(Type *DataType, MaybeAlign Alignment);
  bool isLegalNTLoad(Type *DataType, Align Alignment);
  bool isLegalNTStore(Type *DataType, Align Alignment);
  bool isLegalMaskedGather(Type *DataType, MaybeAlign Alignment);
  bool isLegalMaskedScatter(Type *DataType, MaybeAlign Alignment);
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
                      unsigned Alignment, unsigned AddressSpace);
  int getGSVectorCost(unsigned Opcode, Type *DataTy, Value *Ptr,
                      unsigned Alignment, unsigned AddressSpace);

  /// @}
};

} // end namespace llvm

#endif
