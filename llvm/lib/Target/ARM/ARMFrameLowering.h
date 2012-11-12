//==-- ARMTargetFrameLowering.h - Define frame lowering for ARM --*- C++ -*-==//
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

#ifndef ARM_FRAMEINFO_H
#define ARM_FRAMEINFO_H

#include "ARM.h"
#include "ARMSubtarget.h"
#include "llvm/Target/TargetFrameLowering.h"

namespace llvm {
  class ARMSubtarget;

class ARMFrameLowering : public TargetFrameLowering {
protected:
  const ARMSubtarget &STI;

public:
  explicit ARMFrameLowering(const ARMSubtarget &sti)
    : TargetFrameLowering(StackGrowsDown, sti.getStackAlignment(), 0, 4),
      STI(sti) {
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
  bool canSimplifyCallFramePseudos(const MachineFunction &MF) const;
  int getFrameIndexReference(const MachineFunction &MF, int FI,
                             unsigned &FrameReg) const;
  int ResolveFrameIndexReference(const MachineFunction &MF,
                                 int FI,
                                 unsigned &FrameReg, int SPAdj) const;
  int getFrameIndexOffset(const MachineFunction &MF, int FI) const;

  void processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                            RegScavenger *RS) const;

 private:
  void emitPushInst(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                    const std::vector<CalleeSavedInfo> &CSI, unsigned StmOpc,
                    unsigned StrOpc, bool NoGap,
                    bool(*Func)(unsigned, bool), unsigned NumAlignedDPRCS2Regs,
                    unsigned MIFlags = 0) const;
  void emitPopInst(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                   const std::vector<CalleeSavedInfo> &CSI, unsigned LdmOpc,
                   unsigned LdrOpc, bool isVarArg, bool NoGap,
                   bool(*Func)(unsigned, bool),
                   unsigned NumAlignedDPRCS2Regs) const;
};

} // End llvm namespace

#endif
