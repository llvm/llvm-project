//==-- AArch64FrameLowering.h - TargetFrameLowering for AArch64 --*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AARCH64_AARCH64FRAMELOWERING_H
#define LLVM_LIB_TARGET_AARCH64_AARCH64FRAMELOWERING_H

#include "AArch64StackOffset.h"
#include "llvm/CodeGen/TargetFrameLowering.h"

namespace llvm {

class AArch64FrameLowering : public TargetFrameLowering {
public:
  explicit AArch64FrameLowering()
      : TargetFrameLowering(StackGrowsDown, Align(16), 0, Align(16),
                            true /*StackRealignable*/) {}

  void
  emitCalleeSavedFrameMoves(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MBBI) const override;

  MachineBasicBlock::iterator
  eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator I) const override;

  /// emitProlog/emitEpilog - These methods insert prolog and epilog code into
  /// the function.
  void emitPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const override;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const override;

  bool canUseAsPrologue(const MachineBasicBlock &MBB) const override;

  int getFrameIndexReference(const MachineFunction &MF, int FI,
                             Register &FrameReg) const override;
  StackOffset resolveFrameIndexReference(const MachineFunction &MF, int FI,
                                         Register &FrameReg, bool PreferFP,
                                         bool ForSimm) const;
  StackOffset resolveFrameOffsetReference(const MachineFunction &MF,
                                          int64_t ObjectOffset, bool isFixed,
                                          bool isSVE, Register &FrameReg,
                                          bool PreferFP, bool ForSimm) const;
  bool spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MI,
                                 ArrayRef<CalleeSavedInfo> CSI,
                                 const TargetRegisterInfo *TRI) const override;

  bool
  restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MI,
                              MutableArrayRef<CalleeSavedInfo> CSI,
                              const TargetRegisterInfo *TRI) const override;

  /// Can this function use the red zone for local allocations.
  bool canUseRedZone(const MachineFunction &MF) const;

  bool hasFP(const MachineFunction &MF) const override;
  bool hasReservedCallFrame(const MachineFunction &MF) const override;

  void determineCalleeSaves(MachineFunction &MF, BitVector &SavedRegs,
                            RegScavenger *RS) const override;

  /// Returns true if the target will correctly handle shrink wrapping.
  bool enableShrinkWrapping(const MachineFunction &MF) const override {
    return true;
  }

  bool enableStackSlotScavenging(const MachineFunction &MF) const override;
  TargetStackID::Value getStackIDForScalableVectors() const override;

  void processFunctionBeforeFrameFinalized(MachineFunction &MF,
                                             RegScavenger *RS) const override;

  void
  processFunctionBeforeFrameIndicesReplaced(MachineFunction &MF,
                                            RegScavenger *RS) const override;

  unsigned getWinEHParentFrameOffset(const MachineFunction &MF) const override;

  unsigned getWinEHFuncletFrameSize(const MachineFunction &MF) const;

  int getFrameIndexReferencePreferSP(const MachineFunction &MF, int FI,
                                     Register &FrameReg,
                                     bool IgnoreSPUpdates) const override;
  int getNonLocalFrameIndexReference(const MachineFunction &MF,
                               int FI) const override;
  int getSEHFrameIndexOffset(const MachineFunction &MF, int FI) const;

  bool isSupportedStackID(TargetStackID::Value ID) const override {
    switch (ID) {
    default:
      return false;
    case TargetStackID::Default:
    case TargetStackID::SVEVector:
    case TargetStackID::NoAlloc:
      return true;
    }
  }

  bool isStackIdSafeForLocalArea(unsigned StackId) const override {
    // We don't support putting SVE objects into the pre-allocated local
    // frame block at the moment.
    return StackId != TargetStackID::SVEVector;
  }

private:
  bool shouldCombineCSRLocalStackBump(MachineFunction &MF,
                                      uint64_t StackBumpBytes) const;

  int64_t estimateSVEStackObjectOffsets(MachineFrameInfo &MF) const;
  int64_t assignSVEStackObjectOffsets(MachineFrameInfo &MF,
                                      int &MinCSFrameIndex,
                                      int &MaxCSFrameIndex) const;
  bool shouldCombineCSRLocalStackBumpInEpilogue(MachineBasicBlock &MBB,
                                                unsigned StackBumpBytes) const;
};

} // End llvm namespace

#endif
