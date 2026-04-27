//===-- RISCVInlineAsmLowering.h - Inline asm lowering ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes how to lower LLVM inline asm to machine code INLINEASM.
///
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/InlineAsmLowering.h"

namespace llvm {

class RISCVInlineAsmLowering : public InlineAsmLowering {
public:
  RISCVInlineAsmLowering(const TargetLowering *TLI);
  bool
  lowerAsmOperandForConstraint(Value *Val, StringRef Constraint,
                               std::vector<MachineOperand> &Ops,
                               MachineIRBuilder &MIRBuilder) const override;
};

} // namespace llvm
