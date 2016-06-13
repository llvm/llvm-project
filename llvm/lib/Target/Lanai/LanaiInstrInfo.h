//===- LanaiInstrInfo.h - Lanai Instruction Information ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Lanai implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_LANAI_LANAIINSTRINFO_H
#define LLVM_LIB_TARGET_LANAI_LANAIINSTRINFO_H

#include "LanaiRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "LanaiGenInstrInfo.inc"

namespace llvm {

class LanaiInstrInfo : public LanaiGenInstrInfo {
  const LanaiRegisterInfo RegisterInfo;

public:
  LanaiInstrInfo();

  // getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
  // such, whenever a client has an instance of instruction info, it should
  // always be able to get register info as well (through this method).
  virtual const LanaiRegisterInfo &getRegisterInfo() const {
    return RegisterInfo;
  }

  bool areMemAccessesTriviallyDisjoint(MachineInstr *MIa, MachineInstr *MIb,
                                       AliasAnalysis *AA) const override;

  unsigned isLoadFromStackSlot(const MachineInstr *MI,
                               int &FrameIndex) const override;

  unsigned isLoadFromStackSlotPostFE(const MachineInstr *MI,
                                     int &FrameIndex) const override;

  unsigned isStoreToStackSlot(const MachineInstr *MI,
                              int &FrameIndex) const override;

  void copyPhysReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator Position,
                   const DebugLoc &DL, unsigned DestinationRegister,
                   unsigned SourceRegister, bool KillSource) const override;

  void
  storeRegToStackSlot(MachineBasicBlock &MBB,
                      MachineBasicBlock::iterator Position,
                      unsigned SourceRegister, bool IsKill, int FrameIndex,
                      const TargetRegisterClass *RegisterClass,
                      const TargetRegisterInfo *RegisterInfo) const override;

  void
  loadRegFromStackSlot(MachineBasicBlock &MBB,
                       MachineBasicBlock::iterator Position,
                       unsigned DestinationRegister, int FrameIndex,
                       const TargetRegisterClass *RegisterClass,
                       const TargetRegisterInfo *RegisterInfo) const override;

  bool expandPostRAPseudo(MachineBasicBlock::iterator MI) const override;

  bool getMemOpBaseRegImmOfs(MachineInstr *LdSt, unsigned &BaseReg,
                             int64_t &Offset,
                             const TargetRegisterInfo *TRI) const override;

  bool getMemOpBaseRegImmOfsWidth(MachineInstr *LdSt, unsigned &BaseReg,
                                  int64_t &Offset, unsigned &Width,
                                  const TargetRegisterInfo *TRI) const;

  bool AnalyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TrueBlock,
                     MachineBasicBlock *&FalseBlock,
                     SmallVectorImpl<MachineOperand> &Condition,
                     bool AllowModify) const override;

  unsigned RemoveBranch(MachineBasicBlock &MBB) const override;

  bool ReverseBranchCondition(
      SmallVectorImpl<MachineOperand> &Condition) const override;

  unsigned InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TrueBlock,
                        MachineBasicBlock *FalseBlock,
                        ArrayRef<MachineOperand> Condition,
                        const DebugLoc &DL) const override;
};

static inline bool isSPLSOpcode(unsigned Opcode) {
  switch (Opcode) {
  case Lanai::LDBs_RI:
  case Lanai::LDBz_RI:
  case Lanai::LDHs_RI:
  case Lanai::LDHz_RI:
  case Lanai::STB_RI:
  case Lanai::STH_RI:
    return true;
  default:
    return false;
  }
}

static inline bool isRMOpcode(unsigned Opcode) {
  switch (Opcode) {
  case Lanai::LDW_RI:
  case Lanai::SW_RI:
    return true;
  default:
    return false;
  }
}

static inline bool isRRMOpcode(unsigned Opcode) {
  switch (Opcode) {
  case Lanai::LDBs_RR:
  case Lanai::LDBz_RR:
  case Lanai::LDHs_RR:
  case Lanai::LDHz_RR:
  case Lanai::LDWz_RR:
  case Lanai::LDW_RR:
  case Lanai::STB_RR:
  case Lanai::STH_RR:
  case Lanai::SW_RR:
    return true;
  default:
    return false;
  }
}

} // namespace llvm

#endif // LLVM_LIB_TARGET_LANAI_LANAIINSTRINFO_H
