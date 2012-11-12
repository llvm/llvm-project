//=- MBlazeFrameLowering.h - Define frame lowering for MicroBlaze -*- C++ -*-=//
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

#ifndef MBLAZE_FRAMEINFO_H
#define MBLAZE_FRAMEINFO_H

#include "MBlaze.h"
#include "llvm/Target/TargetFrameLowering.h"

namespace llvm {
class MBlazeSubtarget;

class MBlazeFrameLowering : public TargetFrameLowering {
protected:
  const MBlazeSubtarget &STI;

public:
  explicit MBlazeFrameLowering(const MBlazeSubtarget &sti)
    : TargetFrameLowering(TargetFrameLowering::StackGrowsUp, 4, 0), STI(sti) {
  }

  /// targetHandlesStackFrameRounding - Returns true if the target is
  /// responsible for rounding up the stack frame (probably at emitPrologue
  /// time).
  bool targetHandlesStackFrameRounding() const { return true; }

  /// emitProlog/emitEpilog - These methods insert prolog and epilog code into
  /// the function.
  void emitPrologue(MachineFunction &MF) const;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;

  bool hasFP(const MachineFunction &MF) const;

  int getFrameIndexOffset(const MachineFunction &MF, int FI) const;

  virtual void processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                                    RegScavenger *RS) const;
};

} // End llvm namespace

#endif
