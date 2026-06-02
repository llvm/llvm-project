//===- EZHInstrInfo.h - EZH Instruction Information ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the EZH implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_EZH_EZHINSTRINFO_H
#define LLVM_LIB_TARGET_EZH_EZHINSTRINFO_H

#include "EZHRegisterInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/Support/BranchProbability.h"

#define GET_INSTRINFO_HEADER
#include "EZHGenInstrInfo.inc"

namespace llvm {

class EZHSubtarget;

/// EZH Instruction Information.
class EZHInstrInfo : public EZHGenInstrInfo {
  const EZHRegisterInfo RegisterInfo;

public:
  EZHInstrInfo(const EZHSubtarget &STI);

  const EZHRegisterInfo &getRegisterInfo() const { return RegisterInfo; }

  void copyPhysReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator Position,
                   const DebugLoc &DL, Register DestinationRegister,
                   Register SourceRegister, bool KillSource,
                   bool RenamableDest = false,
                   bool RenamableSrc = false) const override;

  void storeRegToStackSlot(
      MachineBasicBlock &MBB, MachineBasicBlock::iterator Position,
      Register SourceRegister, bool IsKill, int FrameIndex,
      const TargetRegisterClass *RegisterClass, Register VReg,
      MachineInstr::MIFlag Flags = MachineInstr::NoFlags) const override;

  void loadRegFromStackSlot(
      MachineBasicBlock &MBB, MachineBasicBlock::iterator Position,
      Register DestinationRegister, int FrameIndex,
      const TargetRegisterClass *RegisterClass, Register VReg,
      unsigned SubReg = 0,
      MachineInstr::MIFlag Flags = MachineInstr::NoFlags) const override;

  bool expandPostRAPseudo(MachineInstr &MI) const override;

  bool analyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TrueBlock,
                     MachineBasicBlock *&FalseBlock,
                     SmallVectorImpl<MachineOperand> &Condition,
                     bool AllowModify) const override;

  unsigned removeBranch(MachineBasicBlock &MBB,
                        int *BytesRemoved = nullptr) const override;

  unsigned insertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TrueBlock,
                        MachineBasicBlock *FalseBlock,
                        ArrayRef<MachineOperand> Condition, const DebugLoc &DL,
                        int *BytesAdded = nullptr) const override;

  bool
  reverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const override;
  bool isPredicated(const MachineInstr &MI) const override;
  bool isPredicable(const MachineInstr &MI) const override;
  bool canPredicatePredicatedInstr(const MachineInstr &MI) const override;
  bool PredicateInstruction(MachineInstr &MI,
                            ArrayRef<MachineOperand> Pred) const override;
  bool isProfitableToIfCvt(MachineBasicBlock &MBB, unsigned NumCycles,
                           unsigned ExtraPredCycles,
                           BranchProbability Probability) const override;
  bool isProfitableToIfCvt(MachineBasicBlock &TMBB, unsigned NumTCycles,
                           unsigned ExtraTCycles, MachineBasicBlock &FMBB,
                           unsigned NumFCycles, unsigned ExtraFCycles,
                           BranchProbability Probability) const override;
  unsigned getInstSizeInBytes(const MachineInstr &MI) const override;
  int getJumpTableIndex(const MachineInstr &MI) const override;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_EZH_EZHINSTRINFO_H
