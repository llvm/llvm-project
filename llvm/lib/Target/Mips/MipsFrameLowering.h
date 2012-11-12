//===-- MipsFrameLowering.h - Define frame lowering for Mips ----*- C++ -*-===//
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

#ifndef MIPS_FRAMEINFO_H
#define MIPS_FRAMEINFO_H

#include "Mips.h"
#include "MipsSubtarget.h"
#include "llvm/Target/TargetFrameLowering.h"

namespace llvm {
  class MipsSubtarget;

class MipsFrameLowering : public TargetFrameLowering {
protected:
  const MipsSubtarget &STI;

public:
  explicit MipsFrameLowering(const MipsSubtarget &sti)
    : TargetFrameLowering(StackGrowsDown, sti.hasMips64() ? 16 : 8, 0,
                          sti.hasMips64() ? 16 : 8), STI(sti) {}

  static const MipsFrameLowering *create(MipsTargetMachine &TM,
                                         const MipsSubtarget &ST);

  bool hasFP(const MachineFunction &MF) const;

protected:
  uint64_t estimateStackSize(const MachineFunction &MF) const;
};

/// Create MipsInstrInfo objects.
const MipsFrameLowering *createMips16FrameLowering(const MipsSubtarget &ST);
const MipsFrameLowering *createMipsSEFrameLowering(const MipsSubtarget &ST);

} // End llvm namespace

#endif
