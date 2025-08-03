//===- AVRTargetTransformInfo.h - AVR specific TTI ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file defines a TargetTransformInfoImplBase conforming object specific
/// to the AVR target machine. It uses the target's detailed information to
/// provide more precise answers to certain TTI queries, while letting the
/// target independent and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AVR_AVRTARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_AVR_AVRTARGETTRANSFORMINFO_H

#include "AVRSubtarget.h"
#include "AVRTargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/IR/Function.h"
#include <optional>

namespace llvm {

class AVRTTIImpl final : public BasicTTIImplBase<AVRTTIImpl> {
  using BaseT = BasicTTIImplBase<AVRTTIImpl>;
  using TTI = TargetTransformInfo;

  friend BaseT;

  const AVRSubtarget *ST;
  const AVRTargetLowering *TLI;

  const AVRSubtarget *getST() const { return ST; }
  const AVRTargetLowering *getTLI() const { return TLI; }

public:
  explicit AVRTTIImpl(const AVRTargetMachine *TM, const Function &F)
      : BaseT(TM, F.getDataLayout()), ST(TM->getSubtargetImpl(F)),
        TLI(ST->getTargetLowering()) {}

#if 0 // TODO Examine if these options result in better code generation
  /// Return the cost of materializing an immediate for a value operand of
  /// a store instruction.
  InstructionCost getStoreImmCost(Type *VecTy, TTI::OperandValueInfo OpInfo,
                                  TTI::TargetCostKind CostKind) const;

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
  getIntrinsicInstrCost(const IntrinsicCostAttributes &ICA,
                        TTI::TargetCostKind CostKind) const override;

  InstructionCost getCmpSelInstrCost(
      unsigned Opcode, Type *ValTy, Type *CondTy, CmpInst::Predicate VecPred,
      TTI::TargetCostKind CostKind,
      TTI::OperandValueInfo Op1Info = {TTI::OK_AnyValue, TTI::OP_None},
      TTI::OperandValueInfo Op2Info = {TTI::OK_AnyValue, TTI::OP_None},
      const Instruction *I = nullptr) const override;
#endif

  bool isLSRCostLess(const TargetTransformInfo::LSRCost &C1,
                     const TargetTransformInfo::LSRCost &C2) const override {return C1.Insns < C2.Insns;}
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AVR_AVRTARGETTRANSFORMINFO_H
