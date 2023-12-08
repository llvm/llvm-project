//===-- SystemZFrameLowering.h - Frame lowering for SystemZ -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZFRAMELOWERING_H
#define LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZFRAMELOWERING_H

#include "MCTargetDesc/SystemZMCTargetDesc.h"
#include "SystemZInstrBuilder.h"
#include "SystemZMachineFunctionInfo.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/Support/TypeSize.h"

namespace llvm {
class SystemZSubtarget;

class SystemZFrameLowering : public TargetFrameLowering {
public:
  SystemZFrameLowering(StackDirection D, Align StackAl, int LAO, Align TransAl,
                       bool StackReal);

  static std::unique_ptr<SystemZFrameLowering>
  create(const SystemZSubtarget &STI);

  // Override TargetFrameLowering.
  bool allocateScavengingFrameIndexesNearIncomingSP(
    const MachineFunction &MF) const override {
    // SystemZ wants normal register scavenging slots, as close to the stack or
    // frame pointer as possible.
    // The default implementation assumes an x86-like layout, where the frame
    // pointer is at the opposite end of the frame from the stack pointer.
    // This meant that when frame pointer elimination was disabled,
    // the slots ended up being as close as possible to the incoming
    // stack pointer, which is the opposite of what we want on SystemZ.
    return false;
  }

  bool hasReservedCallFrame(const MachineFunction &MF) const override;
  MachineBasicBlock::iterator
  eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator MI) const override;
};

class SystemZELFFrameLowering : public SystemZFrameLowering {
  IndexedMap<unsigned> RegSpillOffsets;

public:
  SystemZELFFrameLowering();

  // Override TargetFrameLowering.
  bool
  assignCalleeSavedSpillSlots(MachineFunction &MF,
                              const TargetRegisterInfo *TRI,
                              std::vector<CalleeSavedInfo> &CSI) const override;
  void determineCalleeSaves(MachineFunction &MF, BitVector &SavedRegs,
                            RegScavenger *RS) const override;
  bool spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MBBI,
                                 ArrayRef<CalleeSavedInfo> CSI,
                                 const TargetRegisterInfo *TRI) const override;
  bool
  restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MBBII,
                              MutableArrayRef<CalleeSavedInfo> CSI,
                              const TargetRegisterInfo *TRI) const override;
  void processFunctionBeforeFrameFinalized(MachineFunction &MF,
                                           RegScavenger *RS) const override;
  void emitPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const override;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const override;
  void inlineStackProbe(MachineFunction &MF,
                        MachineBasicBlock &PrologMBB) const override;
  bool hasFP(const MachineFunction &MF) const override;
  StackOffset getFrameIndexReference(const MachineFunction &MF, int FI,
                                     Register &FrameReg) const override;
  void
  orderFrameObjects(const MachineFunction &MF,
                    SmallVectorImpl<int> &ObjectsToAllocate) const override;

  // Return the byte offset from the incoming stack pointer of Reg's
  // ABI-defined save slot.  Return 0 if no slot is defined for Reg.  Adjust
  // the offset in case MF has packed-stack.
  unsigned getRegSpillOffset(MachineFunction &MF, Register Reg) const;

  bool usePackedStack(MachineFunction &MF) const;

  // Return the offset of the backchain.
  unsigned getBackchainOffset(MachineFunction &MF) const {
    // The back chain is stored topmost with packed-stack.
    return usePackedStack(MF) ? SystemZMC::ELFCallFrameSize - 8 : 0;
  }

  // Get or create the frame index of where the old frame pointer is stored.
  int getOrCreateFramePointerSaveIndex(MachineFunction &MF) const;
};

class SystemZXPLINKFrameLowering : public SystemZFrameLowering {
  IndexedMap<unsigned> RegSpillOffsets;

public:
  SystemZXPLINKFrameLowering();

  bool
  assignCalleeSavedSpillSlots(MachineFunction &MF,
                              const TargetRegisterInfo *TRI,
                              std::vector<CalleeSavedInfo> &CSI) const override;

  void determineCalleeSaves(MachineFunction &MF, BitVector &SavedRegs,
                            RegScavenger *RS) const override;

  bool spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MBBI,
                                 ArrayRef<CalleeSavedInfo> CSI,
                                 const TargetRegisterInfo *TRI) const override;

  bool
  restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MBBII,
                              MutableArrayRef<CalleeSavedInfo> CSI,
                              const TargetRegisterInfo *TRI) const override;

  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const override;

  void emitPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const override;

  void inlineStackProbe(MachineFunction &MF,
                        MachineBasicBlock &PrologMBB) const override;

  bool hasFP(const MachineFunction &MF) const override;

  void processFunctionBeforeFrameFinalized(MachineFunction &MF,
                                           RegScavenger *RS) const override;

  void determineFrameLayout(MachineFunction &MF) const;
};
} // end namespace llvm

#endif
