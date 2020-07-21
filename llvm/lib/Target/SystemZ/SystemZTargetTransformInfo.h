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

class SystemZTTIImpl : public BasicTTIImplBase<SystemZTTIImpl> {
  typedef BasicTTIImplBase<SystemZTTIImpl> BaseT;
  typedef TargetTransformInfo TTI;
  friend BaseT;

  const SystemZSubtarget *ST;
  const SystemZTargetLowering *TLI;

  const SystemZSubtarget *getST() const { return ST; }
  const SystemZTargetLowering *getTLI() const { return TLI; }

  unsigned const LIBCALL_COST = 30;

public:
  explicit SystemZTTIImpl(const SystemZTargetMachine *TM, const Function &F)
      : BaseT(TM, F.getParent()->getDataLayout()), ST(TM->getSubtargetImpl(F)),
        TLI(ST->getTargetLowering()) {}

  /// \name Scalar TTI Implementations
  /// @{

  unsigned getInliningThresholdMultiplier() { return 3; }

  int getIntImmCost(const APInt &Imm, Type *Ty, TTI::TargetCostKind CostKind);

  int getIntImmCostInst(unsigned Opcode, unsigned Idx, const APInt &Imm,
                        Type *Ty, TTI::TargetCostKind CostKind);
  int getIntImmCostIntrin(Intrinsic::ID IID, unsigned Idx, const APInt &Imm,
                          Type *Ty, TTI::TargetCostKind CostKind);

  TTI::PopcntSupportKind getPopcntSupport(unsigned TyWidth);

  void getUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                               TTI::UnrollingPreferences &UP);

  void getPeelingPreferences(Loop *L, ScalarEvolution &SE,
                             TTI::PeelingPreferences &PP);

  bool isLSRCostLess(TargetTransformInfo::LSRCost &C1,
                     TargetTransformInfo::LSRCost &C2);
  /// @}

  /// \name Vector TTI Implementations
  /// @{

  unsigned getNumberOfRegisters(unsigned ClassID) const;
  unsigned getRegisterBitWidth(bool Vector) const;

  unsigned getCacheLineSize() const override { return 256; }
  unsigned getPrefetchDistance() const override { return 4500; }
  unsigned getMinPrefetchStride(unsigned NumMemAccesses,
                                unsigned NumStridedMemAccesses,
                                unsigned NumPrefetches,
                                bool HasCall) const override;
  bool enableWritePrefetching() const override { return true; }

  bool hasDivRemOp(Type *DataType, bool IsSigned);
  bool prefersVectorizedAddressing() { return false; }
  bool LSRWithInstrQueries() { return true; }
  bool supportsEfficientVectorElementLoadStore() { return true; }
  bool enableInterleavedAccessVectorization() { return true; }

  int getArithmeticInstrCost(
      unsigned Opcode, Type *Ty,
      TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput,
      TTI::OperandValueKind Opd1Info = TTI::OK_AnyValue,
      TTI::OperandValueKind Opd2Info = TTI::OK_AnyValue,
      TTI::OperandValueProperties Opd1PropInfo = TTI::OP_None,
      TTI::OperandValueProperties Opd2PropInfo = TTI::OP_None,
      ArrayRef<const Value *> Args = ArrayRef<const Value *>(),
      const Instruction *CxtI = nullptr);
  int getShuffleCost(TTI::ShuffleKind Kind, VectorType *Tp, int Index,
                     VectorType *SubTp);
  unsigned getVectorTruncCost(Type *SrcTy, Type *DstTy);
  unsigned getVectorBitmaskConversionCost(Type *SrcTy, Type *DstTy);
  unsigned getBoolVecToIntConversionCost(unsigned Opcode, Type *Dst,
                                         const Instruction *I);
  int getCastInstrCost(unsigned Opcode, Type *Dst, Type *Src,
                       TTI::TargetCostKind CostKind,
                       const Instruction *I = nullptr);
  int getCmpSelInstrCost(unsigned Opcode, Type *ValTy, Type *CondTy,
                         TTI::TargetCostKind CostKind,
                         const Instruction *I = nullptr);
  int getVectorInstrCost(unsigned Opcode, Type *Val, unsigned Index);
  bool isFoldableLoad(const LoadInst *Ld, const Instruction *&FoldedValue);
  int getMemoryOpCost(unsigned Opcode, Type *Src, MaybeAlign Alignment,
                      unsigned AddressSpace, TTI::TargetCostKind CostKind,
                      const Instruction *I = nullptr);

  int getInterleavedMemoryOpCost(
      unsigned Opcode, Type *VecTy, unsigned Factor, ArrayRef<unsigned> Indices,
      Align Alignment, unsigned AddressSpace,
      TTI::TargetCostKind CostKind = TTI::TCK_SizeAndLatency,
      bool UseMaskForCond = false, bool UseMaskForGaps = false);

  int getIntrinsicInstrCost(const IntrinsicCostAttributes &ICA,
                            TTI::TargetCostKind CostKind);
  /// @}
};

} // end namespace llvm

#endif
