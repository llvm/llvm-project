//===-- ConnexInstrInfo.h - Connex Instruction Information ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Connex implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_CONNEX_CONNEXINSTRINFO_H
#define LLVM_LIB_TARGET_CONNEX_CONNEXINSTRINFO_H

#include "Connex.h"
#include "ConnexRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "ConnexGenInstrInfo.inc"

namespace llvm {
class ConnexSubtarget; // Inspired from BPF/BPFInstrInfo.h (from Oct 2025)

class ConnexInstrInfo : public ConnexGenInstrInfo {
  const ConnexRegisterInfo RI;

public:
  // Inspired from BPFInstrInfo.cpp (from Oct 2025)
  explicit ConnexInstrInfo(const ConnexSubtarget &STI);

  const ConnexRegisterInfo &getRegisterInfo() const { return RI; }

  // Got a bit inspired from lib/Target/AMDGPU/SIInstrInfo.cpp
  bool expandPostRAPseudo(MachineInstr &MI) const override;

  // Note: we do not use Pre-RA hazard recognizer since it works on the
  //   MachineInstr immediately after 1st scheduling pass, which is before the,
  //   RA, TwoAddressInstructionPass, etc - so a lot of other instructions
  //   will be added after 1st scheduling pass.
  // We would like our post-RA Hazard recognizer to be able to reschedule
  //   instructions in a different order (with the ScoreBoardHazardRecognizer)
  //   in order to avoid inserting useless NOPs.

  // USE_POSTRA_SCHED
  // Got inspired from llvm/lib/Target/PowerPC/PPCInstrInfo.h
  ScheduleHazardRecognizer *
  CreateTargetPostRAHazardRecognizer(const InstrItineraryData *II,
                                     const ScheduleDAG *DAG) const override;

  ScheduleHazardRecognizer *
  CreateTargetMIHazardRecognizer(const InstrItineraryData *II,
                                 // 2021_02_09: const ScheduleDAG *DAG
                                 const ScheduleDAGMI *DAG // 2021_02_09
  ) const override;

  void insertNoop(MachineBasicBlock &MBB,
                  MachineBasicBlock::iterator MI) const override;

  // Inspired from BPF/BPFInstrInfo.h (from Oct 2025)
  void copyPhysReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                   const DebugLoc &DL, Register DestReg, Register SrcReg,
                   bool KillSrc, bool RenamableDest = false,
                   bool RenamableSrc = false) const override;

  // Inspired from BPF/BPFInstrInfo.h (from Oct 2025)
  void storeRegToStackSlot(
      MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI, Register SrcReg,
      bool isKill, int FrameIndex, const TargetRegisterClass *RC,
      const TargetRegisterInfo *TRI, Register VReg,
      MachineInstr::MIFlag Flags = MachineInstr::NoFlags) const override;

  // Inspired from BPF/BPFInstrInfo.h (from Oct 2025)
  void loadRegFromStackSlot(
      MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
      Register DestReg, int FrameIndex, const TargetRegisterClass *RC,
      const TargetRegisterInfo *TRI, Register VReg,
      MachineInstr::MIFlag Flags = MachineInstr::NoFlags) const override;

  bool analyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                     MachineBasicBlock *&FBB,
                     SmallVectorImpl<MachineOperand> &Cond,
                     bool AllowModify) const override;

  unsigned removeBranch(MachineBasicBlock &MBB,
                        int *BytesRemoved = nullptr) const override;

  unsigned insertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                        MachineBasicBlock *FBB, ArrayRef<MachineOperand> Cond,
                        const DebugLoc &DL,
                        int *BytesAdded = nullptr) const override;

  bool isPredicable(MachineInstr &MI) const;

protected:
  MachineMemOperand *GetMemOperand(MachineBasicBlock &MBB, int FI,
                                   MachineMemOperand::Flags Flag) const;
}; // end class ConnexInstrInfo
} // end namespace llvm

#endif
