//===-- PPCTargetTransformInfo.h - PPC specific TTI -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file a TargetTransformInfo::Concept conforming object specific to the
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

namespace llvm {

class PPCTTIImpl : public BasicTTIImplBase<PPCTTIImpl> {
  typedef BasicTTIImplBase<PPCTTIImpl> BaseT;
  typedef TargetTransformInfo TTI;
  friend BaseT;

  const PPCSubtarget *ST;
  const PPCTargetLowering *TLI;

  const PPCSubtarget *getST() const { return ST; }
  const PPCTargetLowering *getTLI() const { return TLI; }
  bool mightUseCTR(BasicBlock *BB, TargetLibraryInfo *LibInfo,
                   SmallPtrSetImpl<const Value *> &Visited);

public:
  explicit PPCTTIImpl(const PPCTargetMachine *TM, const Function &F)
      : BaseT(TM, F.getParent()->getDataLayout()), ST(TM->getSubtargetImpl(F)),
        TLI(ST->getTargetLowering()) {}

  Optional<Instruction *> instCombineIntrinsic(InstCombiner &IC,
                                               IntrinsicInst &II) const;

  /// \name Scalar TTI Implementations
  /// @{

  using BaseT::getIntImmCost;
  int getIntImmCost(const APInt &Imm, Type *Ty,
                    TTI::TargetCostKind CostKind);

  int getIntImmCostInst(unsigned Opcode, unsigned Idx, const APInt &Imm,
                        Type *Ty, TTI::TargetCostKind CostKind,
                        Instruction *Inst = nullptr);
  int getIntImmCostIntrin(Intrinsic::ID IID, unsigned Idx, const APInt &Imm,
                          Type *Ty, TTI::TargetCostKind CostKind);

  unsigned getUserCost(const User *U, ArrayRef<const Value *> Operands,
                       TTI::TargetCostKind CostKind);

  TTI::PopcntSupportKind getPopcntSupport(unsigned TyWidth);
  bool isHardwareLoopProfitable(Loop *L, ScalarEvolution &SE,
                                AssumptionCache &AC,
                                TargetLibraryInfo *LibInfo,
                                HardwareLoopInfo &HWLoopInfo);
  bool canSaveCmp(Loop *L, BranchInst **BI, ScalarEvolution *SE, LoopInfo *LI,
                  DominatorTree *DT, AssumptionCache *AC,
                  TargetLibraryInfo *LibInfo);
  bool getTgtMemIntrinsic(IntrinsicInst *Inst, MemIntrinsicInfo &Info);
  void getUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                               TTI::UnrollingPreferences &UP);
  void getPeelingPreferences(Loop *L, ScalarEvolution &SE,
                             TTI::PeelingPreferences &PP);
  bool isLSRCostLess(TargetTransformInfo::LSRCost &C1,
                     TargetTransformInfo::LSRCost &C2);
  bool isNumRegsMajorCostOfLSR();

  /// @}

  /// \name Vector TTI Implementations
  /// @{
  bool useColdCCForColdCall(Function &F);
  bool enableAggressiveInterleaving(bool LoopHasReductions);
  TTI::MemCmpExpansionOptions enableMemCmpExpansion(bool OptSize,
                                                    bool IsZeroCmp) const;
  bool enableInterleavedAccessVectorization();

  enum PPCRegisterClass {
    GPRRC, FPRRC, VRRC, VSXRC
  };
  unsigned getNumberOfRegisters(unsigned ClassID) const;
  unsigned getRegisterClassForType(bool Vector, Type *Ty = nullptr) const;
  const char* getRegisterClassName(unsigned ClassID) const;
  unsigned getRegisterBitWidth(bool Vector) const;
  unsigned getCacheLineSize() const override;
  unsigned getPrefetchDistance() const override;
  unsigned getMaxInterleaveFactor(unsigned VF);
  int vectorCostAdjustment(int Cost, unsigned Opcode, Type *Ty1, Type *Ty2);
  int getArithmeticInstrCost(
      unsigned Opcode, Type *Ty,
      TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput,
      TTI::OperandValueKind Opd1Info = TTI::OK_AnyValue,
      TTI::OperandValueKind Opd2Info = TTI::OK_AnyValue,
      TTI::OperandValueProperties Opd1PropInfo = TTI::OP_None,
      TTI::OperandValueProperties Opd2PropInfo = TTI::OP_None,
      ArrayRef<const Value *> Args = ArrayRef<const Value *>(),
      const Instruction *CxtI = nullptr);
  int getShuffleCost(TTI::ShuffleKind Kind, Type *Tp, ArrayRef<int> Mask,
                     int Index, Type *SubTp);
  int getCastInstrCost(unsigned Opcode, Type *Dst, Type *Src,
                       TTI::CastContextHint CCH, TTI::TargetCostKind CostKind,
                       const Instruction *I = nullptr);
  int getCFInstrCost(unsigned Opcode, TTI::TargetCostKind CostKind);
  int getCmpSelInstrCost(unsigned Opcode, Type *ValTy, Type *CondTy,
                         CmpInst::Predicate VecPred,
                         TTI::TargetCostKind CostKind,
                         const Instruction *I = nullptr);
  int getVectorInstrCost(unsigned Opcode, Type *Val, unsigned Index);
  int getMemoryOpCost(unsigned Opcode, Type *Src, MaybeAlign Alignment,
                      unsigned AddressSpace,
                      TTI::TargetCostKind CostKind,
                      const Instruction *I = nullptr);
  int getInterleavedMemoryOpCost(
      unsigned Opcode, Type *VecTy, unsigned Factor, ArrayRef<unsigned> Indices,
      Align Alignment, unsigned AddressSpace,
      TTI::TargetCostKind CostKind = TTI::TCK_SizeAndLatency,
      bool UseMaskForCond = false, bool UseMaskForGaps = false);
  unsigned getIntrinsicInstrCost(const IntrinsicCostAttributes &ICA,
                                 TTI::TargetCostKind CostKind);

  /// @}
};

} // end namespace llvm

#endif
