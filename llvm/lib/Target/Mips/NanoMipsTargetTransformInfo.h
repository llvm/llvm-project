//===- NanoMipsTargetTransformInfo.h - nanoMIPS specific TTI ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file defines a TargetTransformInfo::Concept conforming object specific
/// to the nanoMIPS target machine. It uses the target's detailed information to
/// provide more precise answers to certain TTI queries, while letting the
/// target independent and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MIPS_NANOMIPSTARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_MIPS_NANOMIPSTARGETTRANSFORMINFO_H

#include "MipsSubtarget.h"
#include "MipsTargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/IR/Function.h"

namespace llvm {

class NanoMipsTTIImpl : public BasicTTIImplBase<NanoMipsTTIImpl> {
  using BaseT = BasicTTIImplBase<NanoMipsTTIImpl>;
  using TTI = TargetTransformInfo;

  friend BaseT;

  const MipsSubtarget *ST;
  const MipsTargetLowering *TLI;

  const MipsSubtarget *getST() const { return ST; }
  const MipsTargetLowering *getTLI() const { return TLI; }

public:
  explicit NanoMipsTTIImpl(const MipsTargetMachine *TM, const Function &F)
      : BaseT(TM, F.getParent()->getDataLayout()), ST(TM->getSubtargetImpl(F)),
        TLI(ST->getTargetLowering()) {}

  InstructionCost getIntImmCost(const APInt &Imm, Type *Ty,
                                TTI::TargetCostKind CostKind);
  InstructionCost getIntImmCostInst(unsigned Opcode, unsigned Idx,
                                    const APInt &Imm, Type *Ty,
                                    TTI::TargetCostKind CostKind,
                                    Instruction *Inst = nullptr);
  void getUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                               TTI::UnrollingPreferences &UP);

  bool canMacroFuseCmp(void) const { return true; } // true; }

  InstructionCost getCmpSelInstrCost (
      unsigned Opcode, Type *ValTy, Type *CondTy, CmpInst::Predicate VecPred,
      TTI::TargetCostKind CostKind=TTI::TCK_RecipThroughput,
      const Instruction *I=nullptr,
      ArrayRef<const Value *> Operands = ArrayRef<const Value*>()) const;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_MIPS_NANOMIPSTARGETTRANSFORMINFO_H
