//== ----- llvm/CodeGen/GlobalISel/GISelChangeObserver.h ---------------------
//== //
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

namespace llvm {
/// Abstract class that contains various methods for clients to notify about
/// changes. This should be the preferred way for APIs to notify changes.
/// Typically calling erasedInstr/createdInstr multiple times should not affect
/// the result. The observer would likely need to check if it was already
/// notified earlier (consider using GISelWorkList).
class MachineInstr;
class GISelChangeObserver {
public:
  virtual ~GISelChangeObserver() {}

  /// An instruction was erased.
  virtual void erasedInstr(MachineInstr &MI) = 0;
  /// An instruction was created and inserted into the function.
  virtual void createdInstr(MachineInstr &MI) = 0;
  /// This instruction was mutated in some way.
  virtual void changedInstr(MachineInstr &MI) = 0;
};

} // namespace llvm
#endif
