//===-- PISAInstrInfo.h - PISA Instruction Information --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PISA_PISAINSTRINFO_H
#define LLVM_LIB_TARGET_PISA_PISAINSTRINFO_H
#include "PISARegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "PISAGenInstrInfo.inc"

#define GET_INSTRINFO_OPERAND_ENUM
#include "PISAGenInstrInfo.inc"

namespace llvm {
class PISASubtarget;

class PISAInstrInfo : public PISAGenInstrInfo {
  const PISARegisterInfo RI;

public:
  PISAInstrInfo(const PISASubtarget &STI);

  const PISARegisterInfo &getRegisterInfo() const { return RI; }
  bool isNoEmissionInstr(const MachineInstr &MI) const;
  bool isFunctionParamInstr(const MachineInstr &MI) const;

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
  reverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const override;

  bool expandPostRAPseudo(MachineInstr &MI) const override;

  bool isSafeToMove(const MachineInstr &MI, const MachineBasicBlock *MBB,
                    const MachineFunction &MF) const override;

  void copyPhysReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                   const DebugLoc &DL, Register DestReg, Register SrcReg,
                   bool KillSrc, bool RenamableDest = false,
                   bool RenamableSrc = false) const override;
};
} // namespace llvm

#endif // LLVM_LIB_TARGET_PISA_PISAINSTRINFO_H
