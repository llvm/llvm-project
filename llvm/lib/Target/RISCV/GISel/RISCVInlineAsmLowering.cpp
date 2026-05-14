//===-- RISCVInlineAsmLowering.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the lowering from LLVM IR inline asm to MIR INLINEASM
///
//===----------------------------------------------------------------------===//

#include "RISCVInlineAsmLowering.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/GlobalISel/InlineAsmLowering.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

RISCVInlineAsmLowering::RISCVInlineAsmLowering(const TargetLowering *TLI)
    : InlineAsmLowering(TLI) {}

bool RISCVInlineAsmLowering::lowerAsmOperandForConstraint(
    Value *Val, StringRef Constraint, std::vector<MachineOperand> &Ops,
    MachineIRBuilder &MIRBuilder) const {
  if (Constraint.size() != 1)
    return false;

  // RISC-V specific constraints.
  switch (Constraint[0]) {
  case 'I': // 12-bit signed immediate operand.
    if (ConstantInt *CI = dyn_cast<ConstantInt>(Val)) {
      int64_t ExtVal = CI->getSExtValue();
      if (isInt<12>(ExtVal)) {
        Ops.push_back(MachineOperand::CreateImm(ExtVal));
        return true;
      }
    }
    return false;
  case 'J': // Integer zero operand.
    if (ConstantInt *CI = dyn_cast<ConstantInt>(Val)) {
      if (CI->isZero()) {
        Ops.push_back(MachineOperand::CreateImm(0));
        return true;
      }
    }
    return false;
  case 'K': // 5-bit unsigned immediate operand.
    if (ConstantInt *CI = dyn_cast<ConstantInt>(Val)) {
      uint64_t ExtVal = CI->getZExtValue();
      if (isUInt<5>(ExtVal)) {
        Ops.push_back(MachineOperand::CreateImm(ExtVal));
        return true;
      }
    }
    return false;
  case 'S': // Alias for s.
    return InlineAsmLowering::lowerAsmOperandForConstraint(Val, "s", Ops,
                                                           MIRBuilder);
  default:
    // Target-independent constraints.
    return InlineAsmLowering::lowerAsmOperandForConstraint(Val, Constraint, Ops,
                                                           MIRBuilder);
  }
}
