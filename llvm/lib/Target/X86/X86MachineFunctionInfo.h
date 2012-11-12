//===-- X86MachineFuctionInfo.h - X86 machine function info -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares X86-specific per-machine-function information.
//
//===----------------------------------------------------------------------===//

#ifndef X86MACHINEFUNCTIONINFO_H
#define X86MACHINEFUNCTIONINFO_H

#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {

/// X86MachineFunctionInfo - This class is derived from MachineFunction and
/// contains private X86 target-specific information for each MachineFunction.
class X86MachineFunctionInfo : public MachineFunctionInfo {
  virtual void anchor();

  /// ForceFramePointer - True if the function is required to use of frame
  /// pointer for reasons other than it containing dynamic allocation or
  /// that FP eliminatation is turned off. For example, Cygwin main function
  /// contains stack pointer re-alignment code which requires FP.
  bool ForceFramePointer;

  /// CalleeSavedFrameSize - Size of the callee-saved register portion of the
  /// stack frame in bytes.
  unsigned CalleeSavedFrameSize;

  /// BytesToPopOnReturn - Number of bytes function pops on return (in addition
  /// to the space used by the return address).
  /// Used on windows platform for stdcall & fastcall name decoration
  unsigned BytesToPopOnReturn;

  /// ReturnAddrIndex - FrameIndex for return slot.
  int ReturnAddrIndex;

  /// TailCallReturnAddrDelta - The number of bytes by which return address
  /// stack slot is moved as the result of tail call optimization.
  int TailCallReturnAddrDelta;

  /// SRetReturnReg - Some subtargets require that sret lowering includes
  /// returning the value of the returned struct in a register. This field
  /// holds the virtual register into which the sret argument is passed.
  unsigned SRetReturnReg;

  /// GlobalBaseReg - keeps track of the virtual register initialized for
  /// use as the global base register. This is used for PIC in some PIC
  /// relocation models.
  unsigned GlobalBaseReg;

  /// VarArgsFrameIndex - FrameIndex for start of varargs area.
  int VarArgsFrameIndex;
  /// RegSaveFrameIndex - X86-64 vararg func register save area.
  int RegSaveFrameIndex;
  /// VarArgsGPOffset - X86-64 vararg func int reg offset.
  unsigned VarArgsGPOffset;
  /// VarArgsFPOffset - X86-64 vararg func fp reg offset.
  unsigned VarArgsFPOffset;
  /// ArgumentStackSize - The number of bytes on stack consumed by the arguments
  /// being passed on the stack.
  unsigned ArgumentStackSize;
  /// NumLocalDynamics - Number of local-dynamic TLS accesses.
  unsigned NumLocalDynamics;

public:
  X86MachineFunctionInfo() : ForceFramePointer(false),
                             CalleeSavedFrameSize(0),
                             BytesToPopOnReturn(0),
                             ReturnAddrIndex(0),
                             TailCallReturnAddrDelta(0),
                             SRetReturnReg(0),
                             GlobalBaseReg(0),
                             VarArgsFrameIndex(0),
                             RegSaveFrameIndex(0),
                             VarArgsGPOffset(0),
                             VarArgsFPOffset(0),
                             ArgumentStackSize(0),
                             NumLocalDynamics(0) {}

  explicit X86MachineFunctionInfo(MachineFunction &MF)
    : ForceFramePointer(false),
      CalleeSavedFrameSize(0),
      BytesToPopOnReturn(0),
      ReturnAddrIndex(0),
      TailCallReturnAddrDelta(0),
      SRetReturnReg(0),
      GlobalBaseReg(0),
      VarArgsFrameIndex(0),
      RegSaveFrameIndex(0),
      VarArgsGPOffset(0),
      VarArgsFPOffset(0),
      ArgumentStackSize(0),
      NumLocalDynamics(0) {}

  bool getForceFramePointer() const { return ForceFramePointer;}
  void setForceFramePointer(bool forceFP) { ForceFramePointer = forceFP; }

  unsigned getCalleeSavedFrameSize() const { return CalleeSavedFrameSize; }
  void setCalleeSavedFrameSize(unsigned bytes) { CalleeSavedFrameSize = bytes; }

  unsigned getBytesToPopOnReturn() const { return BytesToPopOnReturn; }
  void setBytesToPopOnReturn (unsigned bytes) { BytesToPopOnReturn = bytes;}

  int getRAIndex() const { return ReturnAddrIndex; }
  void setRAIndex(int Index) { ReturnAddrIndex = Index; }

  int getTCReturnAddrDelta() const { return TailCallReturnAddrDelta; }
  void setTCReturnAddrDelta(int delta) {TailCallReturnAddrDelta = delta;}

  unsigned getSRetReturnReg() const { return SRetReturnReg; }
  void setSRetReturnReg(unsigned Reg) { SRetReturnReg = Reg; }

  unsigned getGlobalBaseReg() const { return GlobalBaseReg; }
  void setGlobalBaseReg(unsigned Reg) { GlobalBaseReg = Reg; }

  int getVarArgsFrameIndex() const { return VarArgsFrameIndex; }
  void setVarArgsFrameIndex(int Idx) { VarArgsFrameIndex = Idx; }

  int getRegSaveFrameIndex() const { return RegSaveFrameIndex; }
  void setRegSaveFrameIndex(int Idx) { RegSaveFrameIndex = Idx; }

  unsigned getVarArgsGPOffset() const { return VarArgsGPOffset; }
  void setVarArgsGPOffset(unsigned Offset) { VarArgsGPOffset = Offset; }

  unsigned getVarArgsFPOffset() const { return VarArgsFPOffset; }
  void setVarArgsFPOffset(unsigned Offset) { VarArgsFPOffset = Offset; }

  unsigned getArgumentStackSize() const { return ArgumentStackSize; }
  void setArgumentStackSize(unsigned size) { ArgumentStackSize = size; }

  unsigned getNumLocalDynamicTLSAccesses() const { return NumLocalDynamics; }
  void incNumLocalDynamicTLSAccesses() { ++NumLocalDynamics; }

};

} // End llvm namespace

#endif
