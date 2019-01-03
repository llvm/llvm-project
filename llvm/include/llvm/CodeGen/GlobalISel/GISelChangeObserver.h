//===----- llvm/CodeGen/GlobalISel/GISelChangeObserver.h ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// This contains common code to allow clients to notify changes to machine
/// instr.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CODEGEN_GLOBALISEL_GISELCHANGEOBSERVER_H
#define LLVM_CODEGEN_GLOBALISEL_GISELCHANGEOBSERVER_H

#include "llvm/ADT/SmallPtrSet.h"

namespace llvm {
class MachineInstr;
class MachineRegisterInfo;

/// Abstract class that contains various methods for clients to notify about
/// changes. This should be the preferred way for APIs to notify changes.
/// Typically calling erasingInstr/createdInstr multiple times should not affect
/// the result. The observer would likely need to check if it was already
/// notified earlier (consider using GISelWorkList).
class GISelChangeObserver {
  SmallPtrSet<MachineInstr *, 4> ChangingAllUsesOfReg;

public:
  virtual ~GISelChangeObserver() {}

  /// An instruction is about to be erased.
  virtual void erasingInstr(const MachineInstr &MI) = 0;
  /// An instruction was created and inserted into the function.
  virtual void createdInstr(const MachineInstr &MI) = 0;
  /// This instruction is about to be mutated in some way.
  virtual void changingInstr(const MachineInstr &MI) = 0;
  /// This instruction was mutated in some way.
  virtual void changedInstr(const MachineInstr &MI) = 0;

  /// All the instructions using the given register are being changed.
  /// For convenience, finishedChangingAllUsesOfReg() will report the completion
  /// of the changes. The use list may change between this call and
  /// finishedChangingAllUsesOfReg().
  void changingAllUsesOfReg(const MachineRegisterInfo &MRI, unsigned Reg);
  /// All instructions reported as changing by changingAllUsesOfReg() have
  /// finished being changed.
  void finishedChangingAllUsesOfReg();

};

} // namespace llvm
#endif
