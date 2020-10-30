//==- HexagonTargetTransformInfo.cpp - Hexagon specific TTI pass -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
/// \file
/// This file implements a TargetTransformInfo analysis pass specific to the
/// Hexagon target machine. It uses the target's detailed information to provide
/// more precise answers to certain TTI queries, while letting the target
/// independent and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_HEXAGON_HEXAGONTARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_HEXAGON_HEXAGONTARGETTRANSFORMINFO_H

#include "Hexagon.h"
#include "HexagonSubtarget.h"
#include "HexagonTargetMachine.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/IR/Function.h"

namespace llvm {

class Loop;
class ScalarEvolution;
class User;
class Value;

class HexagonTTIImpl : public BasicTTIImplBase<HexagonTTIImpl> {
  using BaseT = BasicTTIImplBase<HexagonTTIImpl>;
  using TTI = TargetTransformInfo;

  friend BaseT;

  const HexagonSubtarget &ST;
  const HexagonTargetLowering &TLI;

  const HexagonSubtarget *getST() const { return &ST; }
  const HexagonTargetLowering *getTLI() const { return &TLI; }

  bool useHVX() const;
  bool isTypeForHVX(Type *VecTy) const;

  // Returns the number of vector elements of Ty, if Ty is a vector type,
  // or 1 if Ty is a scalar type. It is incorrect to call this function
  // with any other type.
  unsigned getTypeNumElements(Type *Ty) const;

public:
  explicit HexagonTTIImpl(const HexagonTargetMachine *TM, const Function &F)
      : BaseT(TM, F.getParent()->getDataLayout()),
        ST(*TM->getSubtargetImpl(F)), TLI(*ST.getTargetLowering()) {}

  /// \name Scalar TTI Implementations
  /// @{

  TTI::PopcntSupportKind getPopcntSupport(unsigned IntTyWidthInBit) const;

  // The Hexagon target can unroll loops with run-time trip counts.
  void getUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                               TTI::UnrollingPreferences &UP);

  void getPeelingPreferences(Loop *L, ScalarEvolution &SE,
                             TTI::PeelingPreferences &PP);

  /// Bias LSR towards creating post-increment opportunities.
  bool shouldFavorPostInc() const;

  // L1 cache prefetch.
  unsigned getPrefetchDistance() const override;
  unsigned getCacheLineSize() const override;

  /// @}

  /// \name Vector TTI Implementations
  /// @{

  unsigned getNumberOfRegisters(bool vector) const;
  unsigned getMaxInterleaveFactor(unsigned VF);
  unsigned getRegisterBitWidth(bool Vector) const;
  unsigned getMinVectorRegisterBitWidth() const;
  unsigned getMinimumVF(unsigned ElemWidth) const;

  bool shouldMaximizeVectorBandwidth(bool OptSize) const {
    return true;
  }
  bool supportsEfficientVectorElementLoadStore() {
    return false;
  }
  bool hasBranchDivergence() {
    return false;
  }
  bool enableAggressiveInterleaving(bool LoopHasReductions) {
    return false;
  }
  bool prefersVectorizedAddressing() {
    return false;
  }
  bool enableInterleavedAccessVectorization() {
    return true;
  }

  unsigned getScalarizationOverhead(VectorType *Ty, const APInt &DemandedElts,
                                    bool Insert, bool Extract);
  unsigned getOperandsScalarizationOverhead(ArrayRef<const Value *> Args,
                                            unsigned VF);
  unsigned getCallInstrCost(Function *F, Type *RetTy, ArrayRef<Type*> Tys,
                            TTI::TargetCostKind CostKind);
  unsigned getIntrinsicInstrCost(const IntrinsicCostAttributes &ICA,
                                 TTI::TargetCostKind CostKind);
  unsigned getAddressComputationCost(Type *Tp, ScalarEvolution *SE,
            const SCEV *S);
  unsigned getMemoryOpCost(unsigned Opcode, Type *Src, MaybeAlign Alignment,
                           unsigned AddressSpace,
                           TTI::TargetCostKind CostKind,
                           const Instruction *I = nullptr);
  unsigned
  getMaskedMemoryOpCost(unsigned Opcode, Type *Src, Align Alignment,
                        unsigned AddressSpace,
                        TTI::TargetCostKind CostKind = TTI::TCK_SizeAndLatency);
  unsigned getShuffleCost(TTI::ShuffleKind Kind, Type *Tp, int Index,
            Type *SubTp);
  unsigned getGatherScatterOpCost(unsigned Opcode, Type *DataTy,
                                  const Value *Ptr, bool VariableMask,
                                  Align Alignment, TTI::TargetCostKind CostKind,
                                  const Instruction *I);
  unsigned getInterleavedMemoryOpCost(
      unsigned Opcode, Type *VecTy, unsigned Factor, ArrayRef<unsigned> Indices,
      Align Alignment, unsigned AddressSpace,
      TTI::TargetCostKind CostKind = TTI::TCK_SizeAndLatency,
      bool UseMaskForCond = false, bool UseMaskForGaps = false);
  unsigned getCmpSelInstrCost(unsigned Opcode, Type *ValTy, Type *CondTy,
                              TTI::TargetCostKind CostKind,
                              const Instruction *I = nullptr);
  unsigned getArithmeticInstrCost(
      unsigned Opcode, Type *Ty,
      TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput,
      TTI::OperandValueKind Opd1Info = TTI::OK_AnyValue,
      TTI::OperandValueKind Opd2Info = TTI::OK_AnyValue,
      TTI::OperandValueProperties Opd1PropInfo = TTI::OP_None,
      TTI::OperandValueProperties Opd2PropInfo = TTI::OP_None,
      ArrayRef<const Value *> Args = ArrayRef<const Value *>(),
      const Instruction *CxtI = nullptr);
  unsigned getCastInstrCost(unsigned Opcode, Type *Dst, Type *Src,
                            TTI::CastContextHint CCH,
                            TTI::TargetCostKind CostKind,
                            const Instruction *I = nullptr);
  unsigned getVectorInstrCost(unsigned Opcode, Type *Val, unsigned Index);

  unsigned getCFInstrCost(unsigned Opcode, TTI::TargetCostKind CostKind) {
    return 1;
  }

  bool isLegalMaskedStore(Type *DataType, Align Alignment);
  bool isLegalMaskedLoad(Type *DataType, Align Alignment);

  /// @}

  int getUserCost(const User *U, ArrayRef<const Value *> Operands,
                  TTI::TargetCostKind CostKind);

  // Hexagon specific decision to generate a lookup table.
  bool shouldBuildLookupTables() const;
};

} // end namespace llvm
#endif // LLVM_LIB_TARGET_HEXAGON_HEXAGONTARGETTRANSFORMINFO_H
