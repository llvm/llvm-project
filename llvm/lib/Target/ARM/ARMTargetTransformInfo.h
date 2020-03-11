//===- ARMTargetTransformInfo.h - ARM specific TTI --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file a TargetTransformInfo::Concept conforming object specific to the
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
#include "llvm/MC/SubtargetFeature.h"

namespace llvm {

class APInt;
class ARMTargetLowering;
class Instruction;
class Loop;
class SCEV;
class ScalarEvolution;
class Type;
class Value;

class ARMTTIImpl : public BasicTTIImplBase<ARMTTIImpl> {
  using BaseT = BasicTTIImplBase<ARMTTIImpl>;
  using TTI = TargetTransformInfo;

  friend BaseT;

  const ARMSubtarget *ST;
  const ARMTargetLowering *TLI;

  // Currently the following features are excluded from InlineFeatureWhitelist.
  // ModeThumb, FeatureNoARM, ModeSoftFloat, FeatureFP64, FeatureD32
  // Depending on whether they are set or unset, different
  // instructions/registers are available. For example, inlining a callee with
  // -thumb-mode in a caller with +thumb-mode, may cause the assembler to
  // fail if the callee uses ARM only instructions, e.g. in inline asm.
  const FeatureBitset InlineFeatureWhitelist = {
      ARM::FeatureVFP2, ARM::FeatureVFP3, ARM::FeatureNEON, ARM::FeatureThumb2,
      ARM::FeatureFP16, ARM::FeatureVFP4, ARM::FeatureFPARMv8,
      ARM::FeatureFullFP16, ARM::FeatureFP16FML, ARM::FeatureHWDivThumb,
      ARM::FeatureHWDivARM, ARM::FeatureDB, ARM::FeatureV7Clrex,
      ARM::FeatureAcquireRelease, ARM::FeatureSlowFPBrcc,
      ARM::FeaturePerfMon, ARM::FeatureTrustZone, ARM::Feature8MSecExt,
      ARM::FeatureCrypto, ARM::FeatureCRC, ARM::FeatureRAS,
      ARM::FeatureFPAO, ARM::FeatureFuseAES, ARM::FeatureZCZeroing,
      ARM::FeatureProfUnpredicate, ARM::FeatureSlowVGETLNi32,
      ARM::FeatureSlowVDUP32, ARM::FeaturePreferVMOVSR,
      ARM::FeaturePrefISHSTBarrier, ARM::FeatureMuxedUnits,
      ARM::FeatureSlowOddRegister, ARM::FeatureSlowLoadDSubreg,
      ARM::FeatureDontWidenVMOVS, ARM::FeatureExpandMLx,
      ARM::FeatureHasVMLxHazards, ARM::FeatureNEONForFPMovs,
      ARM::FeatureNEONForFP, ARM::FeatureCheckVLDnAlign,
      ARM::FeatureHasSlowFPVMLx, ARM::FeatureHasSlowFPVFMx,
      ARM::FeatureVMLxForwarding, ARM::FeaturePref32BitThumb,
      ARM::FeatureAvoidPartialCPSR, ARM::FeatureCheapPredicableCPSR,
      ARM::FeatureAvoidMOVsShOp, ARM::FeatureHasRetAddrStack,
      ARM::FeatureHasNoBranchPredictor, ARM::FeatureDSP, ARM::FeatureMP,
      ARM::FeatureVirtualization, ARM::FeatureMClass, ARM::FeatureRClass,
      ARM::FeatureAClass, ARM::FeatureNaClTrap, ARM::FeatureStrictAlign,
      ARM::FeatureLongCalls, ARM::FeatureExecuteOnly, ARM::FeatureReserveR9,
      ARM::FeatureNoMovt, ARM::FeatureNoNegativeImmediates
  };

  const ARMSubtarget *getST() const { return ST; }
  const ARMTargetLowering *getTLI() const { return TLI; }

public:
  explicit ARMTTIImpl(const ARMBaseTargetMachine *TM, const Function &F)
      : BaseT(TM, F.getParent()->getDataLayout()), ST(TM->getSubtargetImpl(F)),
        TLI(ST->getTargetLowering()) {}

  bool areInlineCompatible(const Function *Caller,
                           const Function *Callee) const;

  bool enableInterleavedAccessVectorization() { return true; }

  bool shouldFavorBackedgeIndex(const Loop *L) const;
  bool shouldFavorPostInc() const;

  /// Floating-point computation using ARMv8 AArch32 Advanced
  /// SIMD instructions remains unchanged from ARMv7. Only AArch64 SIMD
  /// and Arm MVE are IEEE-754 compliant.
  bool isFPVectorizationPotentiallyUnsafe() {
    return !ST->isTargetDarwin() && !ST->hasMVEFloatOps();
  }

  /// \name Scalar TTI Implementations
  /// @{

  int getIntImmCodeSizeCost(unsigned Opcode, unsigned Idx, const APInt &Imm,
                            Type *Ty);

  using BaseT::getIntImmCost;
  int getIntImmCost(const APInt &Imm, Type *Ty);

  int getIntImmCostInst(unsigned Opcode, unsigned Idx, const APInt &Imm, Type *Ty);

  /// @}

  /// \name Vector TTI Implementations
  /// @{

  unsigned getNumberOfRegisters(unsigned ClassID) const {
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

  unsigned getRegisterBitWidth(bool Vector) const {
    if (Vector) {
      if (ST->hasNEON())
        return 128;
      if (ST->hasMVEIntegerOps())
        return 128;
      return 0;
    }

    return 32;
  }

  unsigned getMaxInterleaveFactor(unsigned VF) {
    return ST->getMaxInterleaveFactor();
  }

  bool isLegalMaskedLoad(Type *DataTy, MaybeAlign Alignment);

  bool isLegalMaskedStore(Type *DataTy, MaybeAlign Alignment) {
    return isLegalMaskedLoad(DataTy, Alignment);
  }

  bool isLegalMaskedGather(Type *Ty, MaybeAlign Alignment);

  bool isLegalMaskedScatter(Type *Ty, MaybeAlign Alignment) {
    return isLegalMaskedGather(Ty, Alignment);
  }

  int getMemcpyCost(const Instruction *I);

  int getShuffleCost(TTI::ShuffleKind Kind, Type *Tp, int Index, Type *SubTp);

  bool useReductionIntrinsic(unsigned Opcode, Type *Ty,
                             TTI::ReductionFlags Flags) const;

  bool shouldExpandReduction(const IntrinsicInst *II) const {
    switch (II->getIntrinsicID()) {
    case Intrinsic::experimental_vector_reduce_v2_fadd:
    case Intrinsic::experimental_vector_reduce_v2_fmul:
      // We don't have legalization support for ordered FP reductions.
      if (!II->getFastMathFlags().allowReassoc())
        return true;
      // Can't legalize reductions with soft floats.
      return TLI->useSoftFloat() || !TLI->getSubtarget()->hasFPRegs();

    case Intrinsic::experimental_vector_reduce_fmin:
    case Intrinsic::experimental_vector_reduce_fmax:
      // Can't legalize reductions with soft floats, and NoNan will create
      // fminimum which we do not know how to lower.
      return TLI->useSoftFloat() || !TLI->getSubtarget()->hasFPRegs() ||
             !II->getFastMathFlags().noNaNs();

    default:
      // Don't expand anything else, let legalization deal with it.
      return false;
    }
  }

  int getCastInstrCost(unsigned Opcode, Type *Dst, Type *Src,
                       const Instruction *I = nullptr);

  int getCmpSelInstrCost(unsigned Opcode, Type *ValTy, Type *CondTy,
                         const Instruction *I = nullptr);

  int getVectorInstrCost(unsigned Opcode, Type *Val, unsigned Index);

  int getAddressComputationCost(Type *Val, ScalarEvolution *SE,
                                const SCEV *Ptr);

  int getArithmeticInstrCost(
      unsigned Opcode, Type *Ty,
      TTI::OperandValueKind Op1Info = TTI::OK_AnyValue,
      TTI::OperandValueKind Op2Info = TTI::OK_AnyValue,
      TTI::OperandValueProperties Opd1PropInfo = TTI::OP_None,
      TTI::OperandValueProperties Opd2PropInfo = TTI::OP_None,
      ArrayRef<const Value *> Args = ArrayRef<const Value *>(),
      const Instruction *CxtI = nullptr);

  int getMemoryOpCost(unsigned Opcode, Type *Src, MaybeAlign Alignment,
                      unsigned AddressSpace, const Instruction *I = nullptr);

  int getInterleavedMemoryOpCost(unsigned Opcode, Type *VecTy, unsigned Factor,
                                 ArrayRef<unsigned> Indices, unsigned Alignment,
                                 unsigned AddressSpace,
                                 bool UseMaskForCond = false,
                                 bool UseMaskForGaps = false);

  unsigned getGatherScatterOpCost(unsigned Opcode, Type *DataTy, Value *Ptr,
                                  bool VariableMask, unsigned Alignment,
                                  const Instruction *I = nullptr);

  bool isLoweredToCall(const Function *F);
  bool isHardwareLoopProfitable(Loop *L, ScalarEvolution &SE,
                                AssumptionCache &AC,
                                TargetLibraryInfo *LibInfo,
                                HardwareLoopInfo &HWLoopInfo);
  bool preferPredicateOverEpilogue(Loop *L, LoopInfo *LI,
                                   ScalarEvolution &SE,
                                   AssumptionCache &AC,
                                   TargetLibraryInfo *TLI,
                                   DominatorTree *DT,
                                   const LoopAccessInfo *LAI);
  void getUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                               TTI::UnrollingPreferences &UP);

  bool shouldBuildLookupTablesForConstant(Constant *C) const {
    // In the ROPI and RWPI relocation models we can't have pointers to global
    // variables or functions in constant data, so don't convert switches to
    // lookup tables if any of the values would need relocation.
    if (ST->isROPI() || ST->isRWPI())
      return !C->needsRelocation();

    return true;
  }
  /// @}
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_ARM_ARMTARGETTRANSFORMINFO_H
