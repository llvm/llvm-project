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

#ifdef LLVM_BUILD_GLOBAL_ISEL
#ifndef LLVM_LIB_TARGET_X86_X86INSTRUCTIONSELECTOR_H
#define LLVM_LIB_TARGET_X86_X86INSTRUCTIONSELECTOR_H

#include "llvm/CodeGen/GlobalISel/InstructionSelector.h"
#include "llvm/CodeGen/MachineOperand.h"

namespace llvm {

class X86InstrInfo;
class X86RegisterBankInfo;
class X86RegisterInfo;
class X86Subtarget;
class X86TargetMachine;
class LLT;
class RegisterBank;
class MachineRegisterInfo;
class MachineFunction;

class X86InstructionSelector : public InstructionSelector {
public:
  X86InstructionSelector(const X86Subtarget &STI,
                         const X86RegisterBankInfo &RBI);

  bool select(MachineInstr &I) const override;

private:
  /// tblgen-erated 'select' implementation, used as the initial selector for
  /// the patterns that don't require complex C++.
  bool selectImpl(MachineInstr &I) const;

  // TODO: remove after selectImpl support pattern with a predicate.
  unsigned getFAddOp(LLT &Ty, const RegisterBank &RB) const;
  unsigned getFSubOp(LLT &Ty, const RegisterBank &RB) const;
  unsigned getAddOp(LLT &Ty, const RegisterBank &RB) const;
  unsigned getSubOp(LLT &Ty, const RegisterBank &RB) const;
  unsigned getLoadStoreOp(LLT &Ty, const RegisterBank &RB, unsigned Opc,
                          uint64_t Alignment) const;

  bool selectBinaryOp(MachineInstr &I, MachineRegisterInfo &MRI,
                      MachineFunction &MF) const;
  bool selectLoadStoreOp(MachineInstr &I, MachineRegisterInfo &MRI,
                         MachineFunction &MF) const;
  bool selectFrameIndex(MachineInstr &I, MachineRegisterInfo &MRI,
                        MachineFunction &MF) const;

  const X86Subtarget &STI;
  const X86InstrInfo &TII;
  const X86RegisterInfo &TRI;
  const X86RegisterBankInfo &RBI;

#define GET_GLOBALISEL_TEMPORARIES_DECL
#include "X86GenGlobalISel.inc"
#undef GET_GLOBALISEL_TEMPORARIES_DECL
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_X86_X86INSTRUCTIONSELECTOR_H
#endif // LLVM_BUILD_GLOBAL_ISEL
