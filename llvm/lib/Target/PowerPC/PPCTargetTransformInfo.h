//===-- PPCTargetTransformInfo.h - PPC specific TTI -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file a TargetTransformInfoImplBase conforming object specific to the
/// PPC target machine. It uses the target's detailed information to
/// provide more precise answers to certain TTI queries, while letting the
/// target independent and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_POWERPC_PPCTARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_POWERPC_PPCTARGETTRANSFORMINFO_H

#include "PPCTargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/CodeGen/TargetLowering.h"
#include <optional>

namespace llvm {

class PPCTTIImpl final : public BasicTTIImplBase<PPCTTIImpl> {
  typedef BasicTTIImplBase<PPCTTIImpl> BaseT;
  typedef TargetTransformInfo TTI;
  friend BaseT;

  const PPCSubtarget *ST;
  const PPCTargetLowering *TLI;

  const PPCSubtarget *getST() const { return ST; }
  const PPCTargetLowering *getTLI() const { return TLI; }

public:
  explicit PPCTTIImpl(const PPCTargetMachine *TM, const Function &F)
      : BaseT(TM, F.getDataLayout()), ST(TM->getSubtargetImpl(F)),
        TLI(ST->getTargetLowering()) {}

  std::optional<Instruction *>
  instCombineIntrinsic(InstCombiner &IC, IntrinsicInst &II) const override;

  /// \name Scalar TTI Implementations
  /// @{

  using BaseT::getIntImmCost;
  InstructionCost getIntImmCost(const APInt &Imm, Type *Ty,
                                TTI::TargetCostKind CostKind) const override;

  InstructionCost getIntImmCostInst(unsigned Opcode, unsigned Idx,
                                    const APInt &Imm, Type *Ty,
                                    TTI::TargetCostKind CostKind,
                                    Instruction *Inst = nullptr) const override;
  InstructionCost
  getIntImmCostIntrin(Intrinsic::ID IID, unsigned Idx, const APInt &Imm,
                      Type *Ty, TTI::TargetCostKind CostKind) const override;

  InstructionCost
  getInstructionCost(const User *U, ArrayRef<const Value *> Operands,
                     TTI::TargetCostKind CostKind) const override;

  TTI::PopcntSupportKind getPopcntSupport(unsigned TyWidth) const override;
  bool isHardwareLoopProfitable(Loop *L, ScalarEvolution &SE,
                                AssumptionCache &AC, TargetLibraryInfo *LibInfo,
                                HardwareLoopInfo &HWLoopInfo) const override;
  bool canSaveCmp(Loop *L, BranchInst **BI, ScalarEvolution *SE, LoopInfo *LI,
                  DominatorTree *DT, AssumptionCache *AC,
                  TargetLibraryInfo *LibInfo) const override;
  bool getTgtMemIntrinsic(IntrinsicInst *Inst,
                          MemIntrinsicInfo &Info) const override;
  void getUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                               TTI::UnrollingPreferences &UP,
                               OptimizationRemarkEmitter *ORE) const override;
  void getPeelingPreferences(Loop *L, ScalarEvolution &SE,
                             TTI::PeelingPreferences &PP) const override;
  bool isLSRCostLess(const TargetTransformInfo::LSRCost &C1,
                     const TargetTransformInfo::LSRCost &C2) const override;
  bool isNumRegsMajorCostOfLSR() const override;
  bool shouldBuildRelLookupTables() const override;
  /// @}

  /// \name Vector TTI Implementations
  /// @{
  bool useColdCCForColdCall(Function &F) const override;
  bool enableAggressiveInterleaving(bool LoopHasReductions) const override;
  TTI::MemCmpExpansionOptions
  enableMemCmpExpansion(bool OptSize, bool IsZeroCmp) const override;
  bool enableInterleavedAccessVectorization() const override;

  enum PPCRegisterClass {
    GPRRC, FPRRC, VRRC, VSXRC
  };
  unsigned getNumberOfRegisters(unsigned ClassID) const override;
  unsigned getRegisterClassForType(bool Vector,
                                   Type *Ty = nullptr) const override;
  const char *getRegisterClassName(unsigned ClassID) const override;
  TypeSize
  getRegisterBitWidth(TargetTransformInfo::RegisterKind K) const override;
  unsigned getCacheLineSize() const override;
  unsigned getPrefetchDistance() const override;
  unsigned getMaxInterleaveFactor(ElementCount VF) const override;
  InstructionCost vectorCostAdjustmentFactor(unsigned Opcode, Type *Ty1,
                                             Type *Ty2) const;
  InstructionCost getArithmeticInstrCost(
      unsigned Opcode, Type *Ty, TTI::TargetCostKind CostKind,
      TTI::OperandValueInfo Op1Info = {TTI::OK_AnyValue, TTI::OP_None},
      TTI::OperandValueInfo Op2Info = {TTI::OK_AnyValue, TTI::OP_None},
      ArrayRef<const Value *> Args = {},
      const Instruction *CxtI = nullptr) const override;
  InstructionCost
  getShuffleCost(TTI::ShuffleKind Kind, VectorType *Tp, ArrayRef<int> Mask,
                 TTI::TargetCostKind CostKind, int Index, VectorType *SubTp,
                 ArrayRef<const Value *> Args = {},
                 const Instruction *CxtI = nullptr) const override;
  InstructionCost
  getCastInstrCost(unsigned Opcode, Type *Dst, Type *Src,
                   TTI::CastContextHint CCH, TTI::TargetCostKind CostKind,
                   const Instruction *I = nullptr) const override;
  InstructionCost getCFInstrCost(unsigned Opcode, TTI::TargetCostKind CostKind,
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
  getIntrinsicInstrCost(const IntrinsicCostAttributes &ICA,
                        TTI::TargetCostKind CostKind) const override;
  bool areInlineCompatible(const Function *Caller,
                           const Function *Callee) const override;
  bool areTypesABICompatible(const Function *Caller, const Function *Callee,
                             const ArrayRef<Type *> &Types) const override;
  bool hasActiveVectorLength(unsigned Opcode, Type *DataType,
                             Align Alignment) const override;
  InstructionCost
  getVPMemoryOpCost(unsigned Opcode, Type *Src, Align Alignment,
                    unsigned AddressSpace, TTI::TargetCostKind CostKind,
                    const Instruction *I = nullptr) const override;
  bool supportsTailCallFor(const CallBase *CB) const override;

private:
  // The following constant is used for estimating costs on power9.
  static const InstructionCost::CostType P9PipelineFlushEstimate = 80;

  /// @}
};

} // end namespace llvm

#endif
