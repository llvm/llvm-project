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

#ifndef LLVM_LIB_TARGET_ARM_ARMFRAMELOWERING_H
#define LLVM_LIB_TARGET_ARM_ARMFRAMELOWERING_H

#include "llvm/Target/TargetFrameLowering.h"

namespace llvm {
  class ARMSubtarget;

class ARMFrameLowering : public TargetFrameLowering {
protected:
  const ARMSubtarget &STI;

public:
  explicit ARMFrameLowering(const ARMSubtarget &sti);

  /// emitProlog/emitEpilog - These methods insert prolog and epilog code into
  /// the function.
  void emitPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const override;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const override;

  bool spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MI,
                                 const std::vector<CalleeSavedInfo> &CSI,
                                 const TargetRegisterInfo *TRI) const override;

  bool restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator MI,
                                  const std::vector<CalleeSavedInfo> &CSI,
                                  const TargetRegisterInfo *TRI) const override;

  bool noFramePointerElim(const MachineFunction &MF) const override;

  bool hasFP(const MachineFunction &MF) const override;
  bool hasReservedCallFrame(const MachineFunction &MF) const override;
  bool canSimplifyCallFramePseudos(const MachineFunction &MF) const override;
  int getFrameIndexReference(const MachineFunction &MF, int FI,
                             unsigned &FrameReg) const override;
  int ResolveFrameIndexReference(const MachineFunction &MF, int FI,
                                 unsigned &FrameReg, int SPAdj) const;

  void determineCalleeSaves(MachineFunction &MF, BitVector &SavedRegs,
                            RegScavenger *RS) const override;

  void adjustForSegmentedStacks(MachineFunction &MF,
                                MachineBasicBlock &MBB) const override;

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

  void
  eliminateCallFramePseudoInstr(MachineFunction &MF,
                                MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator MI) const override;
};

} // End llvm namespace

#endif
