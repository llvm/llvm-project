//===- AMDGPUTargetTransformInfo.h - AMDGPU specific TTI --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file a TargetTransformInfoImplBase conforming object specific to the
/// AMDGPU target machine. It uses the target's detailed information to
/// provide more precise answers to certain TTI queries, while letting the
/// target independent and default TTI implementations handle the rest.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUTARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUTARGETTRANSFORMINFO_H

#include "AMDGPU.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/Support/AMDGPUAddrSpace.h"
#include <optional>

namespace llvm {

class AMDGPUTargetMachine;
class GCNSubtarget;
class InstCombiner;
class Loop;
class ScalarEvolution;
class SITargetLowering;
class Type;
class Value;

class AMDGPUTTIImpl final : public BasicTTIImplBase<AMDGPUTTIImpl> {
  using BaseT = BasicTTIImplBase<AMDGPUTTIImpl>;
  using TTI = TargetTransformInfo;

  friend BaseT;

  Triple TargetTriple;

  const TargetSubtargetInfo *ST;
  const TargetLoweringBase *TLI;

  const TargetSubtargetInfo *getST() const { return ST; }
  const TargetLoweringBase *getTLI() const { return TLI; }

public:
  explicit AMDGPUTTIImpl(const AMDGPUTargetMachine *TM, const Function &F);

  void getUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                               TTI::UnrollingPreferences &UP,
                               OptimizationRemarkEmitter *ORE) const override;

  void getPeelingPreferences(Loop *L, ScalarEvolution &SE,
                             TTI::PeelingPreferences &PP) const override;

  uint64_t getMaxMemIntrinsicInlineSizeThreshold() const override;
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
  static constexpr bool InlinerVectorBonusPercent = 0;

  static const FeatureBitset InlineFeatureIgnoreList;

  const GCNSubtarget *getST() const { return ST; }
  const SITargetLowering *getTLI() const { return TLI; }

  static inline int getFullRateInstrCost() {
    return TargetTransformInfo::TCC_Basic;
  }

  static inline int getHalfRateInstrCost(TTI::TargetCostKind CostKind) {
    return CostKind == TTI::TCK_CodeSize ? 2
                                         : 2 * TargetTransformInfo::TCC_Basic;
  }

  // TODO: The size is usually 8 bytes, but takes 4x as many cycles. Maybe
  // should be 2 or 4.
  static inline int getQuarterRateInstrCost(TTI::TargetCostKind CostKind) {
    return CostKind == TTI::TCK_CodeSize ? 2
                                         : 4 * TargetTransformInfo::TCC_Basic;
  }

  // On some parts, normal fp64 operations are half rate, and others
  // quarter. This also applies to some integer operations.
  int get64BitInstrCost(TTI::TargetCostKind CostKind) const;

  std::pair<InstructionCost, MVT> getTypeLegalizationCost(Type *Ty) const;

public:
  explicit GCNTTIImpl(const AMDGPUTargetMachine *TM, const Function &F);

  bool hasBranchDivergence(const Function *F = nullptr) const override;

  void getUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                               TTI::UnrollingPreferences &UP,
                               OptimizationRemarkEmitter *ORE) const override;

  void getPeelingPreferences(Loop *L, ScalarEvolution &SE,
                             TTI::PeelingPreferences &PP) const override;

  TTI::PopcntSupportKind getPopcntSupport(unsigned TyWidth) const override {
    assert(isPowerOf2_32(TyWidth) && "Ty width must be power of 2");
    return TTI::PSK_FastHardware;
  }

  unsigned getNumberOfRegisters(unsigned RCID) const override;
  TypeSize
  getRegisterBitWidth(TargetTransformInfo::RegisterKind Vector) const override;
  unsigned getMinVectorRegisterBitWidth() const override;
  unsigned getMaximumVF(unsigned ElemWidth, unsigned Opcode) const override;
  unsigned getLoadVectorFactor(unsigned VF, unsigned LoadSize,
                               unsigned ChainSizeInBytes,
                               VectorType *VecTy) const override;
  unsigned getStoreVectorFactor(unsigned VF, unsigned StoreSize,
                                unsigned ChainSizeInBytes,
                                VectorType *VecTy) const override;
  unsigned getLoadStoreVecRegBitWidth(unsigned AddrSpace) const override;

  bool isLegalToVectorizeMemChain(unsigned ChainSizeInBytes, Align Alignment,
                                  unsigned AddrSpace) const;
  bool isLegalToVectorizeLoadChain(unsigned ChainSizeInBytes, Align Alignment,
                                   unsigned AddrSpace) const override;
  bool isLegalToVectorizeStoreChain(unsigned ChainSizeInBytes, Align Alignment,
                                    unsigned AddrSpace) const override;

  uint64_t getMaxMemIntrinsicInlineSizeThreshold() const override;
  Type *getMemcpyLoopLoweringType(
      LLVMContext &Context, Value *Length, unsigned SrcAddrSpace,
      unsigned DestAddrSpace, Align SrcAlign, Align DestAlign,
      std::optional<uint32_t> AtomicElementSize) const override;

  void getMemcpyLoopResidualLoweringType(
      SmallVectorImpl<Type *> &OpsOut, LLVMContext &Context,
      unsigned RemainingBytes, unsigned SrcAddrSpace, unsigned DestAddrSpace,
      Align SrcAlign, Align DestAlign,
      std::optional<uint32_t> AtomicCpySize) const override;
  unsigned getMaxInterleaveFactor(ElementCount VF) const override;

  bool getTgtMemIntrinsic(IntrinsicInst *Inst,
                          MemIntrinsicInfo &Info) const override;

  InstructionCost getArithmeticInstrCost(
      unsigned Opcode, Type *Ty, TTI::TargetCostKind CostKind,
      TTI::OperandValueInfo Op1Info = {TTI::OK_AnyValue, TTI::OP_None},
      TTI::OperandValueInfo Op2Info = {TTI::OK_AnyValue, TTI::OP_None},
      ArrayRef<const Value *> Args = {},
      const Instruction *CxtI = nullptr) const override;

  InstructionCost getCFInstrCost(unsigned Opcode, TTI::TargetCostKind CostKind,
                                 const Instruction *I = nullptr) const override;

  bool isInlineAsmSourceOfDivergence(const CallInst *CI,
                                     ArrayRef<unsigned> Indices = {}) const;

  using BaseT::getVectorInstrCost;
  InstructionCost getVectorInstrCost(unsigned Opcode, Type *ValTy,
                                     TTI::TargetCostKind CostKind,
                                     unsigned Index, const Value *Op0,
                                     const Value *Op1) const override;

  bool isReadRegisterSourceOfDivergence(const IntrinsicInst *ReadReg) const;
  bool isSourceOfDivergence(const Value *V) const override;
  bool isAlwaysUniform(const Value *V) const override;

  bool isValidAddrSpaceCast(unsigned FromAS, unsigned ToAS) const override {
    // Address space casts must cast between different address spaces.
    if (FromAS == ToAS)
      return false;

    if (FromAS == AMDGPUAS::FLAT_ADDRESS)
      return AMDGPU::isExtendedGlobalAddrSpace(ToAS) ||
             ToAS == AMDGPUAS::LOCAL_ADDRESS ||
             ToAS == AMDGPUAS::PRIVATE_ADDRESS;

    if (AMDGPU::isExtendedGlobalAddrSpace(FromAS))
      return AMDGPU::isFlatGlobalAddrSpace(ToAS) ||
             ToAS == AMDGPUAS::CONSTANT_ADDRESS_32BIT;

    if (FromAS == AMDGPUAS::LOCAL_ADDRESS ||
        FromAS == AMDGPUAS::PRIVATE_ADDRESS)
      return ToAS == AMDGPUAS::FLAT_ADDRESS;

    return false;
  }

  bool addrspacesMayAlias(unsigned AS0, unsigned AS1) const override {
    return AMDGPU::addrspacesMayAlias(AS0, AS1);
  }

  unsigned getFlatAddressSpace() const override {
    // Don't bother running InferAddressSpaces pass on graphics shaders which
    // don't use flat addressing.
    if (IsGraphics)
      return -1;
    return AMDGPUAS::FLAT_ADDRESS;
  }

  bool collectFlatAddressOperands(SmallVectorImpl<int> &OpIndexes,
                                  Intrinsic::ID IID) const override;

  bool
  canHaveNonUndefGlobalInitializerInAddressSpace(unsigned AS) const override {
    return AS != AMDGPUAS::LOCAL_ADDRESS && AS != AMDGPUAS::REGION_ADDRESS &&
           AS != AMDGPUAS::PRIVATE_ADDRESS;
  }

  Value *rewriteIntrinsicWithAddressSpace(IntrinsicInst *II, Value *OldV,
                                          Value *NewV) const override;

  bool canSimplifyLegacyMulToMul(const Instruction &I, const Value *Op0,
                                 const Value *Op1, InstCombiner &IC) const;

  bool simplifyDemandedLaneMaskArg(InstCombiner &IC, IntrinsicInst &II,
                                   unsigned LaneAgIdx) const;

  std::optional<Instruction *>
  instCombineIntrinsic(InstCombiner &IC, IntrinsicInst &II) const override;

  Value *simplifyAMDGCNLaneIntrinsicDemanded(InstCombiner &IC,
                                             IntrinsicInst &II,
                                             const APInt &DemandedElts,
                                             APInt &UndefElts) const;

  std::optional<Value *> simplifyDemandedVectorEltsIntrinsic(
      InstCombiner &IC, IntrinsicInst &II, APInt DemandedElts, APInt &UndefElts,
      APInt &UndefElts2, APInt &UndefElts3,
      std::function<void(Instruction *, unsigned, APInt, APInt &)>
          SimplifyAndSetOp) const override;

  InstructionCost getVectorSplitCost() const { return 0; }

  InstructionCost
  getShuffleCost(TTI::ShuffleKind Kind, VectorType *Tp, ArrayRef<int> Mask,
                 TTI::TargetCostKind CostKind, int Index, VectorType *SubTp,
                 ArrayRef<const Value *> Args = {},
                 const Instruction *CxtI = nullptr) const override;

  bool isProfitableToSinkOperands(Instruction *I,
                                  SmallVectorImpl<Use *> &Ops) const override;

  bool areInlineCompatible(const Function *Caller,
                           const Function *Callee) const override;

  int getInliningLastCallToStaticBonus() const override;
  unsigned getInliningThresholdMultiplier() const override { return 11; }
  unsigned adjustInliningThreshold(const CallBase *CB) const override;
  unsigned getCallerAllocaCost(const CallBase *CB,
                               const AllocaInst *AI) const override;

  int getInlinerVectorBonusPercent() const override {
    return InlinerVectorBonusPercent;
  }

  InstructionCost
  getArithmeticReductionCost(unsigned Opcode, VectorType *Ty,
                             std::optional<FastMathFlags> FMF,
                             TTI::TargetCostKind CostKind) const override;

  InstructionCost
  getIntrinsicInstrCost(const IntrinsicCostAttributes &ICA,
                        TTI::TargetCostKind CostKind) const override;
  InstructionCost
  getMinMaxReductionCost(Intrinsic::ID IID, VectorType *Ty, FastMathFlags FMF,
                         TTI::TargetCostKind CostKind) const override;

  /// Data cache line size for LoopDataPrefetch pass. Has no use before GFX12.
  unsigned getCacheLineSize() const override { return 128; }

  /// How much before a load we should place the prefetch instruction.
  /// This is currently measured in number of IR instructions.
  unsigned getPrefetchDistance() const override;

  /// \return if target want to issue a prefetch in address space \p AS.
  bool shouldPrefetchAddressSpace(unsigned AS) const override;
  void collectKernelLaunchBounds(
      const Function &F,
      SmallVectorImpl<std::pair<StringRef, int64_t>> &LB) const override;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUTARGETTRANSFORMINFO_H
