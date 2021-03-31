//===------ BPFTargetTransformInfo.h - BPF specific TTI ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file uses the target's specific information to
// provide more precise answers to certain TTI queries, while letting the
// target independent and default TTI implementations handle the rest.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_BPF_BPFTARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_BPF_BPFTARGETTRANSFORMINFO_H

#include "BPFTargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

namespace llvm {
class BPFTTIImpl : public BasicTTIImplBase<BPFTTIImpl> {
  typedef BasicTTIImplBase<BPFTTIImpl> BaseT;
  typedef TargetTransformInfo TTI;
  friend BaseT;

  const BPFSubtarget *ST;
  const BPFTargetLowering *TLI;

  const BPFSubtarget *getST() const { return ST; }
  const BPFTargetLowering *getTLI() const { return TLI; }

public:
  explicit BPFTTIImpl(const BPFTargetMachine *TM, const Function &F)
      : BaseT(TM, F.getParent()->getDataLayout()), ST(TM->getSubtargetImpl(F)),
        TLI(ST->getTargetLowering()) {}

  int getIntImmCost(const APInt &Imm, Type *Ty, TTI::TargetCostKind CostKind) {
    if (Imm.getBitWidth() <= 64 && isInt<32>(Imm.getSExtValue()))
      return TTI::TCC_Free;

    return TTI::TCC_Basic;
  }

  int getCmpSelInstrCost(unsigned Opcode, Type *ValTy, Type *CondTy,
                         CmpInst::Predicate VecPred,
                         TTI::TargetCostKind CostKind,
                         const llvm::Instruction *I = nullptr) {
    if (Opcode == Instruction::Select)
      return SCEVCheapExpansionBudget;

    return BaseT::getCmpSelInstrCost(Opcode, ValTy, CondTy, VecPred, CostKind,
                                     I);
  }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_BPF_BPFTARGETTRANSFORMINFO_H
