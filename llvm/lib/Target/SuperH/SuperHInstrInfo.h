//===-- SuperHInstrInfo.h - SuperH Instruction Information ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the SuperH implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SUPERH_SUPERHINSTRINFO_H
#define LLVM_LIB_TARGET_SUPERH_SUPERHINSTRINFO_H

#include "SuperHRegisterInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/ISDOpcodes.h"

#define GET_INSTRINFO_HEADER
#include "SuperHGenInstrInfo.inc"

namespace llvm {
class SuperHSubtarget;

namespace SHCC {

/// SuperH specific condition codes.
/// These correspond to `SH_*_COND` in `SuperHInstrInfo.td`.
/// They must be kept in sync.
enum CondCode {
  COND_EQ, //!< Equal
  COND_GT, //!< Greater than
  COND_GE, //!< Greater than or equal
  COND_HI, //!< Higher than
  COND_HS, //!< Higher than or same
  COND_PL, //!< Greater than 0
  COND_PZ, //!< Greater than or equal to 0
  COND_T,  //!< True
  COND_F,  //!< False
  COND_INVALID
};

} // end of namespace SHCC

class SuperHInstrInfo : public SuperHGenInstrInfo {
  const SuperHRegisterInfo RI;
  const SuperHSubtarget &Subtarget;
  virtual void anchor();

public:
  explicit SuperHInstrInfo(const SuperHSubtarget &STI);

  /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
  /// such, whenever a client has an instance of instruction info, it should
  /// always be able to get register info as well (through this method).
  const SuperHRegisterInfo &getRegisterInfo() const { return RI; }

  /// Gets whether a given opcode can fill a delay slot.
  /// 
  /// SuperH does not allow branch instructions of any kind to be situated 
  /// in a delay slot, nor does it allow instructions with delay slots
  /// to be chained together.
  bool canFillDelaySlot(unsigned Opcode) const;
  SHCC::CondCode getCondFromBranchOp(unsigned Op) const;
  SHCC::CondCode getOppositeCondCode(SHCC::CondCode CC) const;
  const MCInstrDesc &getBrCond(SHCC::CondCode CC, bool delaySlot) const;

  // Instruction Info
  unsigned getInstSizeInBytes(const MachineInstr &MI) const override;

  /// Return the noop instruction to use for a noop.
  MCInst getNop() const override;
  void insertNoop(MachineBasicBlock &MBB, 
                  MachineBasicBlock::iterator MI) const override;


  // Stack Frames
  void copyPhysReg(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MI, const DebugLoc &DL,
                           Register DestReg, Register SrcReg, bool KillSrc,
                           bool RenamableDest = false,
                           bool RenamableSrc = false) const override;
  void storeRegToStackSlot(
      MachineBasicBlock &MBB, MachineBasicBlock::iterator MI, Register SrcReg,
      bool isKill, int FrameIndex, const TargetRegisterClass *RC, Register VReg,
      MachineInstr::MIFlag Flags = MachineInstr::NoFlags) const override;
  void loadRegFromStackSlot(
      MachineBasicBlock &MBB, MachineBasicBlock::iterator MI, Register DestReg,
      int FrameIndex, const TargetRegisterClass *RC, Register VReg,
      unsigned SubReg = 0,
      MachineInstr::MIFlag Flags = MachineInstr::NoFlags) const override;
  Register isLoadFromStackSlot(const MachineInstr &MI,
                               int &FrameIndex) const override;
  Register isStoreToStackSlot(const MachineInstr &MI,
                              int &FrameIndex) const override;


  // Branch Analysis
  bool analyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                     MachineBasicBlock *&FBB,
                     SmallVectorImpl<MachineOperand> &Cond,
                     bool AllowModify = false) const override;
  unsigned insertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                        MachineBasicBlock *FBB, ArrayRef<MachineOperand> Cond,
                        const DebugLoc &DL,
                        int *BytesAdded = nullptr) const override;
  unsigned removeBranch(MachineBasicBlock &MBB,
                        int *BytesRemoved = nullptr) const override;
  bool
  reverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const override;

  MachineBasicBlock *getBranchDestBlock(const MachineInstr &MI) const override;

  bool isBranchOffsetInRange(unsigned BranchOpc,
                             int64_t BrOffset) const override;
};

const SuperHInstrInfo *createSuperHInstrInfo(const SuperHSubtarget &STI);
}

#endif // end LLVM_LIB_TARGET_SUPERH_SUPERHINSTRINFO_H