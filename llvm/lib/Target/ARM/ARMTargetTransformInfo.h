//===-- ARMTargetTransformInfo.h - ARM specific TTI -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file a TargetTransformInfo::Concept conforming object specific to the
/// ARM target machine. It uses the target's detailed information to
/// provide more precise answers to certain TTI queries, while letting the
/// target independent and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ARM_ARMTARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_ARM_ARMTARGETTRANSFORMINFO_H

#include "ARM.h"
#include "ARMTargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/Target/TargetLowering.h"

namespace llvm {

class ARMTTIImpl : public BasicTTIImplBase<ARMTTIImpl> {
  typedef BasicTTIImplBase<ARMTTIImpl> BaseT;
  typedef TargetTransformInfo TTI;
  friend BaseT;

  const ARMSubtarget *ST;
  const ARMTargetLowering *TLI;

  /// Estimate the overhead of scalarizing an instruction. Insert and Extract
  /// are set if the result needs to be inserted and/or extracted from vectors.
  unsigned getScalarizationOverhead(Type *Ty, bool Insert, bool Extract);

  const ARMSubtarget *getST() const { return ST; }
  const ARMTargetLowering *getTLI() const { return TLI; }

public:
  explicit ARMTTIImpl(const ARMBaseTargetMachine *TM, const Function &F)
      : BaseT(TM, F.getParent()->getDataLayout()), ST(TM->getSubtargetImpl(F)),
        TLI(ST->getTargetLowering()) {}

  // Provide value semantics. MSVC requires that we spell all of these out.
  ARMTTIImpl(const ARMTTIImpl &Arg)
      : BaseT(static_cast<const BaseT &>(Arg)), ST(Arg.ST), TLI(Arg.TLI) {}
  ARMTTIImpl(ARMTTIImpl &&Arg)
      : BaseT(std::move(static_cast<BaseT &>(Arg))), ST(std::move(Arg.ST)),
        TLI(std::move(Arg.TLI)) {}

  bool enableInterleavedAccessVectorization() { return true; }

  /// Floating-point computation using ARMv8 AArch32 Advanced
  /// SIMD instructions remains unchanged from ARMv7. Only AArch64 SIMD
  /// is IEEE-754 compliant, but it's not covered in this target.
  bool isFPVectorizationPotentiallyUnsafe() {
    return !ST->isTargetDarwin();
  }

  /// \name Scalar TTI Implementations
  /// @{

  using BaseT::getIntImmCost;
  int getIntImmCost(const APInt &Imm, Type *Ty);

  int getIntImmCost(unsigned Opcode, unsigned Idx, const APInt &Imm, Type *Ty);

  /// @}

  /// \name Vector TTI Implementations
  /// @{

  unsigned getNumberOfRegisters(bool Vector) {
    if (Vector) {
      if (ST->hasNEON())
        return 16;
      return 0;
    }

    if (ST->isThumb1Only())
      return 8;
    return 13;
  }

  unsigned getRegisterBitWidth(bool Vector) {
    if (Vector) {
      if (ST->hasNEON())
        return 128;
      return 0;
    }

    return 32;
  }

  unsigned getMaxInterleaveFactor(unsigned VF) {
    return ST->getMaxInterleaveFactor();
  }

  int getShuffleCost(TTI::ShuffleKind Kind, Type *Tp, int Index, Type *SubTp);

  int getCastInstrCost(unsigned Opcode, Type *Dst, Type *Src);

  int getCmpSelInstrCost(unsigned Opcode, Type *ValTy, Type *CondTy);

  int getVectorInstrCost(unsigned Opcode, Type *Val, unsigned Index);

  int getAddressComputationCost(Type *Val, bool IsComplex);

  int getFPOpCost(Type *Ty);

  int getArithmeticInstrCost(
      unsigned Opcode, Type *Ty,
      TTI::OperandValueKind Op1Info = TTI::OK_AnyValue,
      TTI::OperandValueKind Op2Info = TTI::OK_AnyValue,
      TTI::OperandValueProperties Opd1PropInfo = TTI::OP_None,
      TTI::OperandValueProperties Opd2PropInfo = TTI::OP_None);

  int getMemoryOpCost(unsigned Opcode, Type *Src, unsigned Alignment,
                      unsigned AddressSpace);

  int getInterleavedMemoryOpCost(unsigned Opcode, Type *VecTy, unsigned Factor,
                                 ArrayRef<unsigned> Indices, unsigned Alignment,
                                 unsigned AddressSpace);
  /// @}
};

} // end namespace llvm

#endif
