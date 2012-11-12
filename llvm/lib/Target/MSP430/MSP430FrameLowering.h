//==- MSP430FrameLowering.h - Define frame lowering for MSP430 --*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//
//===----------------------------------------------------------------------===//

#ifndef MSP430_FRAMEINFO_H
#define MSP430_FRAMEINFO_H

#include "MSP430.h"
#include "MSP430Subtarget.h"
#include "llvm/Target/TargetFrameLowering.h"

namespace llvm {
  class MSP430Subtarget;

class MSP430FrameLowering : public TargetFrameLowering {
protected:
  const MSP430Subtarget &STI;

public:
  explicit MSP430FrameLowering(const MSP430Subtarget &sti)
    : TargetFrameLowering(TargetFrameLowering::StackGrowsDown, 2, -2), STI(sti) {
  }

  /// emitProlog/emitEpilog - These methods insert prolog and epilog code into
  /// the function.
  void emitPrologue(MachineFunction &MF) const;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;

  bool spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MI,
                                 const std::vector<CalleeSavedInfo> &CSI,
                                 const TargetRegisterInfo *TRI) const;
  bool restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MI,
                                   const std::vector<CalleeSavedInfo> &CSI,
                                   const TargetRegisterInfo *TRI) const;

  bool hasFP(const MachineFunction &MF) const;
  bool hasReservedCallFrame(const MachineFunction &MF) const;
  void processFunctionBeforeFrameFinalized(MachineFunction &MF) const;
};

} // End llvm namespace

#endif
