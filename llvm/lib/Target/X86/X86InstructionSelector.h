//===- X86InstructionSelector --------------------------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares the targeting of the InstructionSelector class for X86.
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_X86_X86INSTRUCTIONSELECTOR_H
#define LLVM_LIB_TARGET_X86_X86INSTRUCTIONSELECTOR_H

#include "llvm/CodeGen/GlobalISel/InstructionSelector.h"

namespace llvm {

class X86InstrInfo;
class X86RegisterBankInfo;
class X86RegisterInfo;
class X86Subtarget;
class X86TargetMachine;

class X86InstructionSelector : public InstructionSelector {
public:
  X86InstructionSelector(const X86Subtarget &STI,
                         const X86RegisterBankInfo &RBI);

  bool select(MachineInstr &I) const override;

private:
  /// tblgen-erated 'select' implementation, used as the initial selector for
  /// the patterns that don't require complex C++.
  bool selectImpl(MachineInstr &I) const;

  const X86InstrInfo &TII;
  const X86RegisterInfo &TRI;
  const X86RegisterBankInfo &RBI;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_X86_X86INSTRUCTIONSELECTOR_H
