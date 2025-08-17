//===- R600TargetTransformInfo.h - R600 specific TTI --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file a TargetTransformInfoImplBase conforming object specific to the
/// R600 target machine. It uses the target's detailed information to
/// provide more precise answers to certain TTI queries, while letting the
/// target independent and default TTI implementations handle the rest.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_R600TARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_AMDGPU_R600TARGETTRANSFORMINFO_H

#include "AMDGPUTargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"

namespace llvm {

class R600Subtarget;
class AMDGPUTargetLowering;

class R600TTIImpl final : public BasicTTIImplBase<R600TTIImpl> {
  using BaseT = BasicTTIImplBase<R600TTIImpl>;
  using TTI = TargetTransformInfo;

  friend BaseT;

  const R600Subtarget *ST;
  const AMDGPUTargetLowering *TLI;
  AMDGPUTTIImpl CommonTTI;

public:
  explicit R600TTIImpl(const AMDGPUTargetMachine *TM, const Function &F);

  const R600Subtarget *getST() const { return ST; }
  const AMDGPUTargetLowering *getTLI() const { return TLI; }

  void getUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                               TTI::UnrollingPreferences &UP,
                               OptimizationRemarkEmitter *ORE) const override;
  void getPeelingPreferences(Loop *L, ScalarEvolution &SE,
                             TTI::PeelingPreferences &PP) const override;
  unsigned getHardwareNumberOfRegisters(bool Vec) const;
  unsigned getNumberOfRegisters(unsigned ClassID) const override;
  TypeSize
  getRegisterBitWidth(TargetTransformInfo::RegisterKind Vector) const override;
  unsigned getMinVectorRegisterBitWidth() const override;
  unsigned getLoadStoreVecRegBitWidth(unsigned AddrSpace) const override;
  bool isLegalToVectorizeMemChain(unsigned ChainSizeInBytes, Align Alignment,
                                  unsigned AddrSpace) const;
  bool isLegalToVectorizeLoadChain(unsigned ChainSizeInBytes, Align Alignment,
                                   unsigned AddrSpace) const override;
  bool isLegalToVectorizeStoreChain(unsigned ChainSizeInBytes, Align Alignment,
                                    unsigned AddrSpace) const override;
  unsigned getMaxInterleaveFactor(ElementCount VF) const override;
  InstructionCost getCFInstrCost(unsigned Opcode, TTI::TargetCostKind CostKind,
                                 const Instruction *I = nullptr) const override;
  using BaseT::getVectorInstrCost;
  InstructionCost getVectorInstrCost(unsigned Opcode, Type *ValTy,
                                     TTI::TargetCostKind CostKind,
                                     unsigned Index, const Value *Op0,
                                     const Value *Op1) const override;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_R600TARGETTRANSFORMINFO_H
