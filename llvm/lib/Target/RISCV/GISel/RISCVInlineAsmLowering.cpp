//===-- lib/CodeGen/GlobalISel/InlineAsmLowering.cpp ----------------------===//
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
#include "llvm/IR/Constants.h"

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
      unsigned ConstBitWidth = CI->getBitWidth();
      if (ConstBitWidth <= 12) {
        bool IsBool = ConstBitWidth == 1;
        int64_t ExtVal = IsBool ? CI->getZExtValue() : CI->getSExtValue();
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
      unsigned ConstBitWidth = CI->getBitWidth();
      if (ConstBitWidth <= 5) {
        bool IsBool = ConstBitWidth == 1;
        int64_t ExtVal = IsBool ? CI->getZExtValue() : CI->getSExtValue();
        Ops.push_back(MachineOperand::CreateImm(ExtVal));
        return true;
      }
    }
    return false;
  case 'S': // Alias for s.
    return InlineAsmLowering::lowerAsmOperandForConstraint(Val, "s", Ops,
                                                           MIRBuilder);
  default:
    // Target-indepnedent constraints.
    return InlineAsmLowering::lowerAsmOperandForConstraint(Val, Constraint, Ops,
                                                           MIRBuilder);
  }
}
