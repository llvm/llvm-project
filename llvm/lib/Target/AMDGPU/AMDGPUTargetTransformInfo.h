//===- AMDGPUTargetTransformInfo.h - AMDGPU specific TTI --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file a TargetTransformInfo::Concept conforming object specific to the
/// AMDGPU target machine. It uses the target's detailed information to
/// provide more precise answers to certain TTI queries, while letting the
/// target independent and default TTI implementations handle the rest.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUTARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUTARGETTRANSFORMINFO_H

#include "AMDGPU.h"
#include "AMDGPUSubtarget.h"
#include "AMDGPUTargetMachine.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/IR/Function.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>

namespace llvm {

class AMDGPUTargetLowering;
class InstCombiner;
class Loop;
class ScalarEvolution;
class Type;
class Value;

class AMDGPUTTIImpl final : public BasicTTIImplBase<AMDGPUTTIImpl> {
  using BaseT = BasicTTIImplBase<AMDGPUTTIImpl>;
  using TTI = TargetTransformInfo;

  friend BaseT;

  Triple TargetTriple;

  const GCNSubtarget *ST;
  const TargetLoweringBase *TLI;

  const TargetSubtargetInfo *getST() const { return ST; }
  const TargetLoweringBase *getTLI() const { return TLI; }

public:
  explicit AMDGPUTTIImpl(const AMDGPUTargetMachine *TM, const Function &F)
      : BaseT(TM, F.getParent()->getDataLayout()),
        TargetTriple(TM->getTargetTriple()),
        ST(static_cast<const GCNSubtarget *>(TM->getSubtargetImpl(F))),
        TLI(ST->getTargetLowering()) {}

  void getUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                               TTI::UnrollingPreferences &UP);

  void getPeelingPreferences(Loop *L, ScalarEvolution &SE,
                             TTI::PeelingPreferences &PP);
};

class GCNTTIImpl final : public BasicTTIImplBase<GCNTTIImpl> {
  using BaseT = BasicTTIImplBase<GCNTTIImpl>;
  using TTI = TargetTransformInfo;

  friend BaseT;

  const GCNSubtarget *ST;
  const SITargetLowering *TLI;
  AMDGPUTTIImpl CommonTTI;
  bool IsGraphics;
  bool HasFP32Denormals;
  bool HasFP64FP16Denormals;
  unsigned MaxVGPRs;

  const FeatureBitset InlineFeatureIgnoreList = {
    // Codegen control options which don't matter.
    AMDGPU::FeatureEnableLoadStoreOpt,
    AMDGPU::FeatureEnableSIScheduler,
    AMDGPU::FeatureEnableUnsafeDSOffsetFolding,
    AMDGPU::FeatureFlatForGlobal,
    AMDGPU::FeaturePromoteAlloca,
    AMDGPU::FeatureUnalignedScratchAccess,
    AMDGPU::FeatureUnalignedAccessMode,

    AMDGPU::FeatureAutoWaitcntBeforeBarrier,

    // Property of the kernel/environment which can't actually differ.
    AMDGPU::FeatureSGPRInitBug,
    AMDGPU::FeatureXNACK,
    AMDGPU::FeatureTrapHandler,

    // The default assumption needs to be ecc is enabled, but no directly
    // exposed operations depend on it, so it can be safely inlined.
    AMDGPU::FeatureSRAMECC,

    // Perf-tuning features
    AMDGPU::FeatureFastFMAF32,
    AMDGPU::HalfRate64Ops
  };

  const GCNSubtarget *getST() const { return ST; }
  const AMDGPUTargetLowering *getTLI() const { return TLI; }

  static inline int getFullRateInstrCost() {
    return TargetTransformInfo::TCC_Basic;
  }

  static inline int getHalfRateInstrCost(
      TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput) {
    return CostKind == TTI::TCK_CodeSize ? 2
                                         : 2 * TargetTransformInfo::TCC_Basic;
  }

  // TODO: The size is usually 8 bytes, but takes 4x as many cycles. Maybe
  // should be 2 or 4.
  static inline int getQuarterRateInstrCost(
      TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput) {
    return CostKind == TTI::TCK_CodeSize ? 2
                                         : 4 * TargetTransformInfo::TCC_Basic;
  }

  // On some parts, normal fp64 operations are half rate, and others
  // quarter. This also applies to some integer operations.
  inline int get64BitInstrCost(
      TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput) const {
    return ST->hasHalfRate64Ops() ? getHalfRateInstrCost(CostKind)
                                  : getQuarterRateInstrCost(CostKind);
  }

public:
  explicit GCNTTIImpl(const AMDGPUTargetMachine *TM, const Function &F)
      : BaseT(TM, F.getParent()->getDataLayout()),
        ST(static_cast<const GCNSubtarget *>(TM->getSubtargetImpl(F))),
        TLI(ST->getTargetLowering()), CommonTTI(TM, F),
        IsGraphics(AMDGPU::isGraphics(F.getCallingConv())),
        MaxVGPRs(ST->getMaxNumVGPRs(
            std::max(ST->getWavesPerEU(F).first,
                     ST->getWavesPerEUForWorkGroup(
                         ST->getFlatWorkGroupSizes(F).second)))) {
    AMDGPU::SIModeRegisterDefaults Mode(F);
    HasFP32Denormals = Mode.allFP32Denormals();
    HasFP64FP16Denormals = Mode.allFP64FP16Denormals();
  }

  bool hasBranchDivergence() { return true; }
  bool useGPUDivergenceAnalysis() const;

  void getUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                               TTI::UnrollingPreferences &UP);

  void getPeelingPreferences(Loop *L, ScalarEvolution &SE,
                             TTI::PeelingPreferences &PP);

  TTI::PopcntSupportKind getPopcntSupport(unsigned TyWidth) {
    assert(isPowerOf2_32(TyWidth) && "Ty width must be power of 2");
    return TTI::PSK_FastHardware;
  }

  unsigned getHardwareNumberOfRegisters(bool Vector) const;
  unsigned getNumberOfRegisters(bool Vector) const;
  unsigned getNumberOfRegisters(unsigned RCID) const;
  unsigned getRegisterBitWidth(bool Vector) const;
  unsigned getMinVectorRegisterBitWidth() const;
  unsigned getLoadVectorFactor(unsigned VF, unsigned LoadSize,
                               unsigned ChainSizeInBytes,
                               VectorType *VecTy) const;
  unsigned getStoreVectorFactor(unsigned VF, unsigned StoreSize,
                                unsigned ChainSizeInBytes,
                                VectorType *VecTy) const;
  unsigned getLoadStoreVecRegBitWidth(unsigned AddrSpace) const;

  bool isLegalToVectorizeMemChain(unsigned ChainSizeInBytes, Align Alignment,
                                  unsigned AddrSpace) const;
  bool isLegalToVectorizeLoadChain(unsigned ChainSizeInBytes, Align Alignment,
                                   unsigned AddrSpace) const;
  bool isLegalToVectorizeStoreChain(unsigned ChainSizeInBytes, Align Alignment,
                                    unsigned AddrSpace) const;
  Type *getMemcpyLoopLoweringType(LLVMContext &Context, Value *Length,
                                  unsigned SrcAddrSpace, unsigned DestAddrSpace,
                                  unsigned SrcAlign, unsigned DestAlign) const;

  void getMemcpyLoopResidualLoweringType(SmallVectorImpl<Type *> &OpsOut,
                                         LLVMContext &Context,
                                         unsigned RemainingBytes,
                                         unsigned SrcAddrSpace,
                                         unsigned DestAddrSpace,
                                         unsigned SrcAlign,
                                         unsigned DestAlign) const;
  unsigned getMaxInterleaveFactor(unsigned VF);

  bool getTgtMemIntrinsic(IntrinsicInst *Inst, MemIntrinsicInfo &Info) const;

  int getArithmeticInstrCost(
      unsigned Opcode, Type *Ty,
      TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput,
      TTI::OperandValueKind Opd1Info = TTI::OK_AnyValue,
      TTI::OperandValueKind Opd2Info = TTI::OK_AnyValue,
      TTI::OperandValueProperties Opd1PropInfo = TTI::OP_None,
      TTI::OperandValueProperties Opd2PropInfo = TTI::OP_None,
      ArrayRef<const Value *> Args = ArrayRef<const Value *>(),
      const Instruction *CxtI = nullptr);

  unsigned getCFInstrCost(unsigned Opcode, TTI::TargetCostKind CostKind);

  bool isInlineAsmSourceOfDivergence(const CallInst *CI,
                                     ArrayRef<unsigned> Indices = {}) const;

  int getVectorInstrCost(unsigned Opcode, Type *ValTy, unsigned Index);
  bool isSourceOfDivergence(const Value *V) const;
  bool isAlwaysUniform(const Value *V) const;

  unsigned getFlatAddressSpace() const {
    // Don't bother running InferAddressSpaces pass on graphics shaders which
    // don't use flat addressing.
    if (IsGraphics)
      return -1;
    return AMDGPUAS::FLAT_ADDRESS;
  }

  bool collectFlatAddressOperands(SmallVectorImpl<int> &OpIndexes,
                                  Intrinsic::ID IID) const;
  Value *rewriteIntrinsicWithAddressSpace(IntrinsicInst *II, Value *OldV,
                                          Value *NewV) const;

  bool canSimplifyLegacyMulToMul(const Value *Op0, const Value *Op1,
                                 InstCombiner &IC) const;
  Optional<Instruction *> instCombineIntrinsic(InstCombiner &IC,
                                               IntrinsicInst &II) const;
  Optional<Value *> simplifyDemandedVectorEltsIntrinsic(
      InstCombiner &IC, IntrinsicInst &II, APInt DemandedElts, APInt &UndefElts,
      APInt &UndefElts2, APInt &UndefElts3,
      std::function<void(Instruction *, unsigned, APInt, APInt &)>
          SimplifyAndSetOp) const;

  unsigned getVectorSplitCost() { return 0; }

  unsigned getShuffleCost(TTI::ShuffleKind Kind, VectorType *Tp, int Index,
                          VectorType *SubTp);

  bool areInlineCompatible(const Function *Caller,
                           const Function *Callee) const;

  unsigned getInliningThresholdMultiplier() { return 11; }

  int getInlinerVectorBonusPercent() { return 0; }

  int getArithmeticReductionCost(
      unsigned Opcode,
      VectorType *Ty,
      bool IsPairwise,
      TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput);

  int getIntrinsicInstrCost(const IntrinsicCostAttributes &ICA,
                            TTI::TargetCostKind CostKind);
  int getMinMaxReductionCost(
    VectorType *Ty, VectorType *CondTy, bool IsPairwiseForm, bool IsUnsigned,
    TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput);
};

class R600TTIImpl final : public BasicTTIImplBase<R600TTIImpl> {
  using BaseT = BasicTTIImplBase<R600TTIImpl>;
  using TTI = TargetTransformInfo;

  friend BaseT;

  const R600Subtarget *ST;
  const AMDGPUTargetLowering *TLI;
  AMDGPUTTIImpl CommonTTI;

public:
  explicit R600TTIImpl(const AMDGPUTargetMachine *TM, const Function &F)
    : BaseT(TM, F.getParent()->getDataLayout()),
      ST(static_cast<const R600Subtarget*>(TM->getSubtargetImpl(F))),
      TLI(ST->getTargetLowering()),
      CommonTTI(TM, F) {}

  const R600Subtarget *getST() const { return ST; }
  const AMDGPUTargetLowering *getTLI() const { return TLI; }

  void getUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                               TTI::UnrollingPreferences &UP);
  void getPeelingPreferences(Loop *L, ScalarEvolution &SE,
                             TTI::PeelingPreferences &PP);
  unsigned getHardwareNumberOfRegisters(bool Vec) const;
  unsigned getNumberOfRegisters(bool Vec) const;
  unsigned getRegisterBitWidth(bool Vector) const;
  unsigned getMinVectorRegisterBitWidth() const;
  unsigned getLoadStoreVecRegBitWidth(unsigned AddrSpace) const;
  bool isLegalToVectorizeMemChain(unsigned ChainSizeInBytes, Align Alignment,
                                  unsigned AddrSpace) const;
  bool isLegalToVectorizeLoadChain(unsigned ChainSizeInBytes, Align Alignment,
                                   unsigned AddrSpace) const;
  bool isLegalToVectorizeStoreChain(unsigned ChainSizeInBytes, Align Alignment,
                                    unsigned AddrSpace) const;
  unsigned getMaxInterleaveFactor(unsigned VF);
  unsigned getCFInstrCost(unsigned Opcode, TTI::TargetCostKind CostKind);
  int getVectorInstrCost(unsigned Opcode, Type *ValTy, unsigned Index);
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUTARGETTRANSFORMINFO_H
