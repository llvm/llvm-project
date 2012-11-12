//===-- SparcFrameLowering.h - Define frame lowering for Sparc --*- C++ -*-===//
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

#ifndef SPARC_FRAMEINFO_H
#define SPARC_FRAMEINFO_H

#include "Sparc.h"
#include "SparcSubtarget.h"
#include "llvm/Target/TargetFrameLowering.h"

namespace llvm {
  class SparcSubtarget;

class SparcFrameLowering : public TargetFrameLowering {
public:
  explicit SparcFrameLowering(const SparcSubtarget &/*sti*/)
    : TargetFrameLowering(TargetFrameLowering::StackGrowsDown, 8, 0) {
  }

  /// emitProlog/emitEpilog - These methods insert prolog and epilog code into
  /// the function.
  void emitPrologue(MachineFunction &MF) const;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;

  bool hasFP(const MachineFunction &MF) const { return false; }
};

} // End llvm namespace

#endif
