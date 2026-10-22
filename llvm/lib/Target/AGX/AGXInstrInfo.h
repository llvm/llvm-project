//===-- AGXInstrInfo.h - AGX instruction information ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Interface definition for AGXInstrInfo.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AGX_AGXINSTRINFO_H
#define LLVM_LIB_TARGET_AGX_AGXINSTRINFO_H

#include "AGXRegisterInfo.h"
#include "Utils/AGXBaseInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "AGXGenInstrInfo.inc"

namespace llvm {

class AGXSubtarget;

class AGXInstrInfo final : public AGXGenInstrInfo {
  const AGXRegisterInfo RI;
  const AGXSubtarget &ST;

  void copy32(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
              const DebugLoc &DL, MCRegister Dst, MCRegister Src,
              bool KillSrc) const;

public:
  explicit AGXInstrInfo(const AGXSubtarget &ST);

  const AGXRegisterInfo &getRegisterInfo() const { return RI; }

  void copyPhysReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                   const DebugLoc &DL, Register DestReg, Register SrcReg,
                   bool KillSrc, bool RenamableDest = false,
                   bool RenamableSrc = false) const override;

  void storeRegToStackSlot(
      MachineBasicBlock &MBB, MachineBasicBlock::iterator MI, Register SrcReg,
      bool IsKill, int FrameIndex, const TargetRegisterClass *RC,
      Register VReg,
      MachineInstr::MIFlag Flags = MachineInstr::NoFlags) const override;

  void loadRegFromStackSlot(
      MachineBasicBlock &MBB, MachineBasicBlock::iterator MI, Register DestReg,
      int FrameIndex, const TargetRegisterClass *RC, Register VReg,
      unsigned SubReg = 0,
      MachineInstr::MIFlag Flags = MachineInstr::NoFlags) const override;

  /// Code runs under the execution mask of its block. A block with several
  /// predecessors in the structurized CFG (a Flow block, or the join of an
  /// if) starts with only the lanes of one of them active, so an instruction
  /// sunk into it would not run for the others.
  bool isSafeToSink(MachineInstr &MI, MachineBasicBlock *SuccToSinkTo,
                    MachineCycleInfo *CI) const override {
    return SuccToSinkTo->pred_size() <= 1;
  }

  Register isLoadFromStackSlot(const MachineInstr &MI,
                               int &FrameIndex) const override;
  Register isStoreToStackSlot(const MachineInstr &MI,
                              int &FrameIndex) const override;

  bool analyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                     MachineBasicBlock *&FBB,
                     SmallVectorImpl<MachineOperand> &Cond,
                     bool AllowModify = false) const override;
  unsigned removeBranch(MachineBasicBlock &MBB,
                        int *BytesRemoved = nullptr) const override;
  unsigned insertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                        MachineBasicBlock *FBB, ArrayRef<MachineOperand> Cond,
                        const DebugLoc &DL,
                        int *BytesAdded = nullptr) const override;
  bool
  reverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const override {
    return true;
  }

  /// Execution mask instructions at the start of a block (the reconvergence
  /// of the enclosing construct) must stay before everything else.
  bool isBasicBlockPrologue(const MachineInstr &MI,
                            Register Reg = Register()) const override;

  unsigned getInstSizeInBytes(const MachineInstr &MI) const override;

  /// The real instruction of the subtarget's encoding family for the pseudo
  /// instruction \p Opcode, or -1 if there is none.
  int pseudoToMCOpcode(unsigned Opcode) const;

  static bool isAsync(const MachineInstr &MI) {
    return AGX::TSFlags::isAsync(MI.getDesc().TSFlags);
  }
  static bool isALU(const MachineInstr &MI) {
    return AGX::TSFlags::isALU(MI.getDesc().TSFlags);
  }
  static bool isExecMaskOp(const MachineInstr &MI) {
    return AGX::TSFlags::isExecMaskOp(MI.getDesc().TSFlags);
  }
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AGX_AGXINSTRINFO_H
