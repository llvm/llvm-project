//===-- CSKYInstrInfo.h - CSKY Instruction Information --------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the CSKY implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_CSKY_CSKYINSTRINFO_H
#define LLVM_LIB_TARGET_CSKY_CSKYINSTRINFO_H

#include "MCTargetDesc/CSKYMCTargetDesc.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "CSKYGenInstrInfo.inc"

namespace llvm {

class CSKYSubtarget;

class CSKYInstrInfo : public CSKYGenInstrInfo {
protected:
  const CSKYSubtarget &STI;

public:
  explicit CSKYInstrInfo(CSKYSubtarget &STI);

  unsigned isLoadFromStackSlot(const MachineInstr &MI,
                               int &FrameIndex) const override;
  unsigned isStoreToStackSlot(const MachineInstr &MI,
                              int &FrameIndex) const override;

  void storeRegToStackSlot(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MI, Register SrcReg,
                           bool IsKill, int FrameIndex,
                           const TargetRegisterClass *RC,
                           const TargetRegisterInfo *TRI) const override;

  void loadRegFromStackSlot(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MI, Register DestReg,
                            int FrameIndex, const TargetRegisterClass *RC,
                            const TargetRegisterInfo *TRI) const override;

  void copyPhysReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                   const DebugLoc &DL, MCRegister DestReg, MCRegister SrcReg,
                   bool KillSrc) const override;

  // Materializes the given integer Val into DstReg.
  Register movImm(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                  const DebugLoc &DL, int64_t Val,
                  MachineInstr::MIFlag Flag = MachineInstr::NoFlags) const;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_CSKY_CSKYINSTRINFO_H
