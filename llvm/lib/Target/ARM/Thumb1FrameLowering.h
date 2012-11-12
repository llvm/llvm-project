//===-- Thumb1FrameLowering.h - Thumb1-specific frame info stuff --*- C++ -*-=//
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

#ifndef __THUMB_FRAMEINFO_H_
#define __THUMB_FRAMEINFO_H_

#include "ARM.h"
#include "ARMFrameLowering.h"
#include "ARMSubtarget.h"
#include "Thumb1InstrInfo.h"
#include "Thumb1RegisterInfo.h"
#include "llvm/Target/TargetFrameLowering.h"

namespace llvm {
  class ARMSubtarget;

class Thumb1FrameLowering : public ARMFrameLowering {
public:
  explicit Thumb1FrameLowering(const ARMSubtarget &sti)
    : ARMFrameLowering(sti) {
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

  bool hasReservedCallFrame(const MachineFunction &MF) const;
};

} // End llvm namespace

#endif
