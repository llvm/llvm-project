//==- HexagonTargetTransformInfo.cpp - Hexagon specific TTI pass -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

  const HexagonSubtarget *ST;
  const HexagonTargetLowering *TLI;

  const HexagonSubtarget *getST() const { return ST; }
  const HexagonTargetLowering *getTLI() const { return TLI; }

public:
  explicit HexagonTTIImpl(const HexagonTargetMachine *TM, const Function &F)
      : BaseT(TM, F.getParent()->getDataLayout()), ST(TM->getSubtargetImpl(F)),
        TLI(ST->getTargetLowering()) {}

  /// \name Scalar TTI Implementations
  /// @{

  TTI::PopcntSupportKind getPopcntSupport(unsigned IntTyWidthInBit) const;

  // The Hexagon target can unroll loops with run-time trip counts.
  void getUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                               TTI::UnrollingPreferences &UP);

  /// Bias LSR towards creating post-increment opportunities.
  bool shouldFavorPostInc() const;

  // L1 cache prefetch.
  unsigned getPrefetchDistance() const;
  unsigned getCacheLineSize() const;

  /// @}

  /// \name Vector TTI Implementations
  /// @{

  unsigned getNumberOfRegisters(bool vector) const;
  unsigned getMaxInterleaveFactor(unsigned VF);
  unsigned getRegisterBitWidth(bool Vector) const;
  unsigned getMinVectorRegisterBitWidth() const;
  bool shouldMaximizeVectorBandwidth(bool OptSize) const { return true; }

  bool supportsEfficientVectorElementLoadStore() {
    return false;
  }

  unsigned getScalarizationOverhead(Type *Ty, bool Insert, bool Extract) {
    return 0;
  }

  unsigned getOperandsScalarizationOverhead(ArrayRef<const Value*> Args,
                                            unsigned VF) {
    return 0;
  }

  unsigned getCallInstrCost(Function *F, Type *RetTy, ArrayRef<Type*> Tys) {
    return 1;
  }

  unsigned getIntrinsicInstrCost(Intrinsic::ID ID, Type *RetTy,
            ArrayRef<Value*> Args, FastMathFlags FMF, unsigned VF) {
    return BaseT::getIntrinsicInstrCost(ID, RetTy, Args, FMF, VF);
  }
  unsigned getIntrinsicInstrCost(Intrinsic::ID ID, Type *RetTy,
            ArrayRef<Type*> Tys, FastMathFlags FMF,
            unsigned ScalarizationCostPassed = UINT_MAX) {
    return 1;
  }

  bool hasBranchDivergence() {
    return false;
  }

  bool enableAggressiveInterleaving(bool LoopHasReductions) {
    return false;
  }

  unsigned getCFInstrCost(unsigned Opcode) {
    return 1;
  }

  unsigned getAddressComputationCost(Type *Tp, ScalarEvolution *,
                                     const SCEV *) {
    return 0;
  }

  unsigned getMemoryOpCost(unsigned Opcode, Type *Src, unsigned Alignment,
            unsigned AddressSpace, const Instruction *I = nullptr);

  unsigned getMaskedMemoryOpCost(unsigned Opcode, Type *Src, unsigned Alignment,
                                 unsigned AddressSpace) {
    return 1;
  }

  unsigned getShuffleCost(TTI::ShuffleKind Kind, Type *Tp, int Index,
                          Type *SubTp) {
    return 1;
  }

  unsigned getGatherScatterOpCost(unsigned Opcode, Type *DataTy, Value *Ptr,
                                  bool VariableMask,
                                  unsigned Alignment) {
    return 1;
  }

  unsigned getInterleavedMemoryOpCost(unsigned Opcode, Type *VecTy,
                                      unsigned Factor,
                                      ArrayRef<unsigned> Indices,
                                      unsigned Alignment,
                                      unsigned AddressSpace) {
    return BaseT::getInterleavedMemoryOpCost(Opcode, VecTy, Factor, Indices,
                                             Alignment, AddressSpace);
  }

  unsigned getNumberOfParts(Type *Tp) {
    return BaseT::getNumberOfParts(Tp);
  }

  bool prefersVectorizedAddressing() {
    return true;
  }

  unsigned getCmpSelInstrCost(unsigned Opcode, Type *ValTy, Type *CondTy,
                              const Instruction *I) {
    return BaseT::getCmpSelInstrCost(Opcode, ValTy, CondTy, I);
  }

  unsigned getArithmeticInstrCost(unsigned Opcode, Type *Ty,
            TTI::OperandValueKind Opd1Info = TTI::OK_AnyValue,
            TTI::OperandValueKind Opd2Info = TTI::OK_AnyValue,
            TTI::OperandValueProperties Opd1PropInfo = TTI::OP_None,
            TTI::OperandValueProperties Opd2PropInfo = TTI::OP_None,
            ArrayRef<const Value *> Args = ArrayRef<const Value *>()) {
    return BaseT::getArithmeticInstrCost(Opcode, Ty, Opd1Info, Opd2Info,
                                         Opd1PropInfo, Opd2PropInfo, Args);
  }

  unsigned getCastInstrCost(unsigned Opcode, Type *Dst, Type *Src,
                            const Instruction *I = nullptr) {
    return 1;
  }

  unsigned getVectorInstrCost(unsigned Opcode, Type *Val, unsigned Index) {
    return 1;
  }

  /// @}

  int getUserCost(const User *U, ArrayRef<const Value *> Operands);

  // Hexagon specific decision to generate a lookup table.
  bool shouldBuildLookupTables() const;
};

} // end namespace llvm
#endif // LLVM_LIB_TARGET_HEXAGON_HEXAGONTARGETTRANSFORMINFO_H
