//====- X86MachineFuctionInfo.h - X86 machine function info -----*- C++ -*-===//
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
  /// ForceFramePointer - True if the function is required to use of frame
  /// pointer for reasons other than it containing dynamic allocation or 
  /// that FP eliminatation is turned off. For example, Cygwin main function
  /// contains stack pointer re-alignment code which requires FP.
  bool ForceFramePointer;

  /// CalleeSavedFrameSize - Size of the callee-saved register portion of the
  /// stack frame in bytes.
  unsigned CalleeSavedFrameSize;

  /// BytesToPopOnReturn - Number of bytes function pops on return.
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

  /// ReserveFP - whether the function should reserve the frame pointer
  /// when allocating, even if there may not actually be a frame pointer used.
  bool ReserveFP;

public:
  X86MachineFunctionInfo() : ForceFramePointer(false),
                             CalleeSavedFrameSize(0),
                             BytesToPopOnReturn(0),
                             ReturnAddrIndex(0),
                             TailCallReturnAddrDelta(0),
                             SRetReturnReg(0),
                             GlobalBaseReg(0) {}
  
  explicit X86MachineFunctionInfo(MachineFunction &MF)
    : ForceFramePointer(false),
      CalleeSavedFrameSize(0),
      BytesToPopOnReturn(0),
      ReturnAddrIndex(0),
      TailCallReturnAddrDelta(0),
      SRetReturnReg(0),
      GlobalBaseReg(0),
      ReserveFP(false) {}
  
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

  bool getReserveFP() const { return ReserveFP; }
  void setReserveFP(bool reserveFP) { ReserveFP = reserveFP; }
};

} // End llvm namespace

#endif
