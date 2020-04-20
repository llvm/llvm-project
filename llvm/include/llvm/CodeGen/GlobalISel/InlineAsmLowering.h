//===- llvm/CodeGen/GlobalISel/InlineAsmLowering.h --------------*- C++ -*-===//
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

#ifndef LLVM_CODEGEN_GLOBALISEL_INLINEASMLOWERING_H
#define LLVM_CODEGEN_GLOBALISEL_INLINEASMLOWERING_H

namespace llvm {
class CallBase;
class MachineIRBuilder;
class TargetLowering;

class InlineAsmLowering {
  const TargetLowering *TLI;

  virtual void anchor();

public:
  bool lowerInlineAsm(MachineIRBuilder &MIRBuilder, const CallBase &CB) const;

protected:
  /// Getter for generic TargetLowering class.
  const TargetLowering *getTLI() const { return TLI; }

  /// Getter for target specific TargetLowering class.
  template <class XXXTargetLowering> const XXXTargetLowering *getTLI() const {
    return static_cast<const XXXTargetLowering *>(TLI);
  }

public:
  InlineAsmLowering(const TargetLowering *TLI) : TLI(TLI) {}
  virtual ~InlineAsmLowering() = default;
};

} // end namespace llvm

#endif // LLVM_CODEGEN_GLOBALISEL_INLINEASMLOWERING_H
