//===-- XCoreFrameLowering.h - Frame info for XCore Target ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains XCore frame information that doesn't fit anywhere else
// cleanly...
//
//===----------------------------------------------------------------------===//

#ifndef XCOREFRAMEINFO_H
#define XCOREFRAMEINFO_H

#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
  class XCoreSubtarget;

  class XCoreFrameLowering: public TargetFrameLowering {
  public:
    XCoreFrameLowering(const XCoreSubtarget &STI);

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

    void processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                              RegScavenger *RS = NULL) const;

    //! Stack slot size (4 bytes)
    static int stackSlotSize() {
      return 4;
    }
  };
}

#endif // XCOREFRAMEINFO_H
