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
  const SparcSubtarget &SubTarget;
public:
  explicit SparcFrameLowering(const SparcSubtarget &ST)
    : TargetFrameLowering(TargetFrameLowering::StackGrowsDown,
                          ST.is64Bit() ? 16 : 8, 0, ST.is64Bit() ? 16 : 8),
      SubTarget(ST) {}

  /// emitProlog/emitEpilog - These methods insert prolog and epilog code into
  /// the function.
  void emitPrologue(MachineFunction &MF) const;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;

  void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I) const;

  bool hasReservedCallFrame(const MachineFunction &MF) const;
  bool hasFP(const MachineFunction &MF) const;
  void processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                            RegScavenger *RS = NULL) const;

private:
  // Remap input registers to output registers for leaf procedure.
  void remapRegsForLeafProc(MachineFunction &MF) const;

  // Returns true if MF is a leaf procedure.
  bool isLeafProc(MachineFunction &MF) const;


  // Emits code for adjusting SP in function prologue/epilogue.
  void emitSPAdjustment(MachineFunction &MF,
                        MachineBasicBlock &MBB,
                        MachineBasicBlock::iterator MBBI,
                        int NumBytes, unsigned ADDrr, unsigned ADDri) const;

};

} // End llvm namespace

#endif
