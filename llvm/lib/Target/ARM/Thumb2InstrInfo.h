//===-- Thumb2InstrInfo.h - Thumb-2 Instruction Information -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Thumb-2 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef THUMB2INSTRUCTIONINFO_H
#define THUMB2INSTRUCTIONINFO_H

#include "ARM.h"
#include "ARMBaseInstrInfo.h"
#include "Thumb2RegisterInfo.h"

namespace llvm {
class ARMSubtarget;
class ScheduleHazardRecognizer;

class Thumb2InstrInfo : public ARMBaseInstrInfo {
  Thumb2RegisterInfo RI;
public:
  explicit Thumb2InstrInfo(const ARMSubtarget &STI);

  /// getNoopForMachoTarget - Return the noop instruction to use for a noop.
  void getNoopForMachoTarget(MCInst &NopInst) const;

  // Return the non-pre/post incrementing version of 'Opc'. Return 0
  // if there is not such an opcode.
  unsigned getUnindexedOpcode(unsigned Opc) const;

  void ReplaceTailWithBranchTo(MachineBasicBlock::iterator Tail,
                               MachineBasicBlock *NewDest) const;

  bool isLegalToSplitMBBAt(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI) const;

  void copyPhysReg(MachineBasicBlock &MBB,
                   MachineBasicBlock::iterator I, DebugLoc DL,
                   unsigned DestReg, unsigned SrcReg,
                   bool KillSrc) const;

  void storeRegToStackSlot(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI,
                           unsigned SrcReg, bool isKill, int FrameIndex,
                           const TargetRegisterClass *RC,
                           const TargetRegisterInfo *TRI) const;

  void loadRegFromStackSlot(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MBBI,
                            unsigned DestReg, int FrameIndex,
                            const TargetRegisterClass *RC,
                            const TargetRegisterInfo *TRI) const;

  /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
  /// such, whenever a client has an instance of instruction info, it should
  /// always be able to get register info as well (through this method).
  ///
  const Thumb2RegisterInfo &getRegisterInfo() const { return RI; }
};

/// getITInstrPredicate - Valid only in Thumb2 mode. This function is identical
/// to llvm::getInstrPredicate except it returns AL for conditional branch
/// instructions which are "predicated", but are not in IT blocks.
ARMCC::CondCodes getITInstrPredicate(const MachineInstr *MI, unsigned &PredReg);


}

#endif // THUMB2INSTRUCTIONINFO_H
