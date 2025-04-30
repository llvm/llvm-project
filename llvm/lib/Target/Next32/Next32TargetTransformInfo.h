//===-- Next32TargetTransformInfo.h - Next32 specific TTI -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file a TargetTransformInfo::Concept conforming object specific to the
/// Next32 target machine. It uses the target's detailed information to
/// provide more precise answers to certain TTI queries, while letting the
/// target independent and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NEXT32_NEXT32TARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_NEXT32_NEXT32TARGETTRANSFORMINFO_H

#include "Next32TargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/CodeGen/TargetLowering.h"

namespace llvm {

class Next32TTIImpl : public BasicTTIImplBase<Next32TTIImpl> {
  typedef BasicTTIImplBase<Next32TTIImpl> BaseT;
  typedef TargetTransformInfo TTI;
  friend BaseT;

  const Next32Subtarget *ST;
  const Next32TargetLowering *TLI;

  const Next32Subtarget *getST() const { return ST; };
  const Next32TargetLowering *getTLI() const { return TLI; };

  bool enableVectorization() const;

public:
  explicit Next32TTIImpl(const Next32TargetMachine *TM, const Function &F)
      : BaseT(TM, F.getParent()->getDataLayout()), ST(TM->getSubtargetImpl()),
        TLI(ST->getTargetLowering()) {}

  bool isValidVectorLoadStore(FixedVectorType *VTy) const;
  bool shouldForceVectorizeInst(Instruction *I, Type *Ty) const;
  bool enableInterleavedAccessVectorization();
  unsigned getNumberOfRegisters(unsigned ClassID) const;
  TypeSize getRegisterBitWidth(TargetTransformInfo::RegisterKind K) const;
  InstructionCost getMemoryOpCost(
      unsigned Opcode, Type *Src, MaybeAlign Alignment, unsigned AddressSpace,
      TTI::TargetCostKind CostKind,
      TTI::OperandValueInfo OpdInfo = {TTI::OK_AnyValue, TTI::OP_None},
      const Instruction *I = nullptr);
  InstructionCost getInterleavedMemoryOpCost(
      unsigned Opcode, Type *VecTy, unsigned Factor, ArrayRef<unsigned> Indices,
      Align Alignment, unsigned AddressSpace, TTI::TargetCostKind CostKind,
      bool UseMaskForCond = false, bool UseMaskForGaps = false);
  InstructionCost getShuffleCost(TTI::ShuffleKind Kind, VectorType *Tp,
                                 ArrayRef<int> Mask,
                                 TTI::TargetCostKind CostKind, int Index,
                                 VectorType *SubTp,
                                 ArrayRef<const Value *> Args = std::nullopt,
                                 const Instruction *CxtI = nullptr);
  using BaseT::getVectorInstrCost;
  InstructionCost getVectorInstrCost(unsigned Opcode, Type *Val,
                                     TTI::TargetCostKind CostKind,
                                     unsigned Index, Value *Op0, Value *Op1);
  bool hasDivRemOp(Type *DataType, bool IsSigned) const;

  bool isLegalMaskedLoadStore(Type *DataType, Align Alignment) {
    bool IsMaskedLoadStoreSupported = false;
    if (ST->hasVectorInst() && DataType->isVectorTy())
      IsMaskedLoadStoreSupported = true;
    return IsMaskedLoadStoreSupported;
  }

  bool isLegalMaskedLoad(Type *DataType, Align Alignment) {
    return isLegalMaskedLoadStore(DataType, Alignment);
  }
  bool isLegalMaskedStore(Type *DataType, Align Alignment) {
    return isLegalMaskedLoadStore(DataType, Alignment);
  }
};
} // end namespace llvm

#endif
