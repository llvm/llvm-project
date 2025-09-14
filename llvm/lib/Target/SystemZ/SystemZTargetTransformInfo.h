//===-- SystemZTargetTransformInfo.h - SystemZ-specific TTI ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZTARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZTARGETTRANSFORMINFO_H

#include "SystemZTargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"

namespace llvm {

class SystemZTTIImpl final : public BasicTTIImplBase<SystemZTTIImpl> {
  typedef BasicTTIImplBase<SystemZTTIImpl> BaseT;
  typedef TargetTransformInfo TTI;
  friend BaseT;

  const SystemZSubtarget *ST;
  const SystemZTargetLowering *TLI;

  const SystemZSubtarget *getST() const { return ST; }
  const SystemZTargetLowering *getTLI() const { return TLI; }

  unsigned const LIBCALL_COST = 30;

  bool isInt128InVR(Type *Ty) const {
    return Ty->isIntegerTy(128) && ST->hasVector();
  }

public:
  explicit SystemZTTIImpl(const SystemZTargetMachine *TM, const Function &F)
      : BaseT(TM, F.getDataLayout()), ST(TM->getSubtargetImpl(F)),
        TLI(ST->getTargetLowering()) {}

  /// \name Scalar TTI Implementations
  /// @{

  unsigned adjustInliningThreshold(const CallBase *CB) const override;

  InstructionCost getIntImmCost(const APInt &Imm, Type *Ty,
                                TTI::TargetCostKind CostKind) const override;

  InstructionCost getIntImmCostInst(unsigned Opcode, unsigned Idx,
                                    const APInt &Imm, Type *Ty,
                                    TTI::TargetCostKind CostKind,
                                    Instruction *Inst = nullptr) const override;
  InstructionCost
  getIntImmCostIntrin(Intrinsic::ID IID, unsigned Idx, const APInt &Imm,
                      Type *Ty, TTI::TargetCostKind CostKind) const override;

  TTI::PopcntSupportKind getPopcntSupport(unsigned TyWidth) const override;

  void getUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                               TTI::UnrollingPreferences &UP,
                               OptimizationRemarkEmitter *ORE) const override;

  void getPeelingPreferences(Loop *L, ScalarEvolution &SE,
                             TTI::PeelingPreferences &PP) const override;

  bool isLSRCostLess(const TargetTransformInfo::LSRCost &C1,
                     const TargetTransformInfo::LSRCost &C2) const override;

  /// @}

  /// \name Vector TTI Implementations
  /// @{

  unsigned getNumberOfRegisters(unsigned ClassID) const override;
  TypeSize
  getRegisterBitWidth(TargetTransformInfo::RegisterKind K) const override;

  unsigned getCacheLineSize() const override { return 256; }
  unsigned getPrefetchDistance() const override { return 4500; }
  unsigned getMinPrefetchStride(unsigned NumMemAccesses,
                                unsigned NumStridedMemAccesses,
                                unsigned NumPrefetches,
                                bool HasCall) const override;
  bool enableWritePrefetching() const override { return true; }

  bool hasDivRemOp(Type *DataType, bool IsSigned) const override;
  bool prefersVectorizedAddressing() const override { return false; }
  bool LSRWithInstrQueries() const override { return true; }
  InstructionCost getScalarizationOverhead(
      VectorType *Ty, const APInt &DemandedElts, bool Insert, bool Extract,
      TTI::TargetCostKind CostKind, bool ForPoisonSrc = true,
      ArrayRef<Value *> VL = {}) const override;
  bool supportsEfficientVectorElementLoadStore() const override { return true; }
  bool enableInterleavedAccessVectorization() const override { return true; }

  InstructionCost getArithmeticInstrCost(
      unsigned Opcode, Type *Ty, TTI::TargetCostKind CostKind,
      TTI::OperandValueInfo Op1Info = {TTI::OK_AnyValue, TTI::OP_None},
      TTI::OperandValueInfo Op2Info = {TTI::OK_AnyValue, TTI::OP_None},
      ArrayRef<const Value *> Args = {},
      const Instruction *CxtI = nullptr) const override;
  InstructionCost
  getShuffleCost(TTI::ShuffleKind Kind, VectorType *DstTy, VectorType *SrcTy,
                 ArrayRef<int> Mask, TTI::TargetCostKind CostKind, int Index,
                 VectorType *SubTp, ArrayRef<const Value *> Args = {},
                 const Instruction *CxtI = nullptr) const override;
  unsigned getVectorTruncCost(Type *SrcTy, Type *DstTy) const;
  unsigned getVectorBitmaskConversionCost(Type *SrcTy, Type *DstTy) const;
  unsigned getBoolVecToIntConversionCost(unsigned Opcode, Type *Dst,
                                         const Instruction *I) const;
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
  bool isFoldableLoad(const LoadInst *Ld,
                      const Instruction *&FoldedValue) const;
  InstructionCost getMemoryOpCost(
      unsigned Opcode, Type *Src, Align Alignment, unsigned AddressSpace,
      TTI::TargetCostKind CostKind,
      TTI::OperandValueInfo OpInfo = {TTI::OK_AnyValue, TTI::OP_None},
      const Instruction *I = nullptr) const override;

  InstructionCost getInterleavedMemoryOpCost(
      unsigned Opcode, Type *VecTy, unsigned Factor, ArrayRef<unsigned> Indices,
      Align Alignment, unsigned AddressSpace, TTI::TargetCostKind CostKind,
      bool UseMaskForCond = false, bool UseMaskForGaps = false) const override;

  InstructionCost
  getArithmeticReductionCost(unsigned Opcode, VectorType *Ty,
                             std::optional<FastMathFlags> FMF,
                             TTI::TargetCostKind CostKind) const override;
  InstructionCost
  getMinMaxReductionCost(Intrinsic::ID IID, VectorType *Ty, FastMathFlags FMF,
                         TTI::TargetCostKind CostKind) const override;

  InstructionCost
  getIntrinsicInstrCost(const IntrinsicCostAttributes &ICA,
                        TTI::TargetCostKind CostKind) const override;

  bool shouldExpandReduction(const IntrinsicInst *II) const override;
  /// @}
};

} // end namespace llvm

#endif
