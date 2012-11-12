//===-- MipsSEFrameLowering.h - Mips32/64 frame lowering --------*- C++ -*-===//
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

#ifndef MIPSSE_FRAMEINFO_H
#define MIPSSE_FRAMEINFO_H

#include "MipsFrameLowering.h"

namespace llvm {

class MipsSEFrameLowering : public MipsFrameLowering {
public:
  explicit MipsSEFrameLowering(const MipsSubtarget &STI)
    : MipsFrameLowering(STI) {}

  /// emitProlog/emitEpilog - These methods insert prolog and epilog code into
  /// the function.
  void emitPrologue(MachineFunction &MF) const;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;

  bool spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MI,
                                 const std::vector<CalleeSavedInfo> &CSI,
                                 const TargetRegisterInfo *TRI) const;

  bool hasReservedCallFrame(const MachineFunction &MF) const;

  void processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                            RegScavenger *RS) const;
};

} // End llvm namespace

#endif
