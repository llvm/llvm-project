//===-- RISCVInstrInfo.h - RISCV Instruction Information --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the RISCV implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVINSTRINFO_H
#define LLVM_LIB_TARGET_RISCV_RISCVINSTRINFO_H

#include "RISCVRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "RISCVGenInstrInfo.inc"

namespace llvm {

class RISCVSubtarget;

class RISCVInstrInfo : public RISCVGenInstrInfo {

public:
  explicit RISCVInstrInfo(RISCVSubtarget &STI);

  unsigned isLoadFromStackSlot(const MachineInstr &MI,
                               int &FrameIndex) const override;
  unsigned isStoreToStackSlot(const MachineInstr &MI,
                              int &FrameIndex) const override;

  void copyPhysReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                   const DebugLoc &DL, MCRegister DstReg, MCRegister SrcReg,
                   bool KillSrc) const override;

  void storeRegToStackSlot(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI, Register SrcReg,
                           bool IsKill, int FrameIndex,
                           const TargetRegisterClass *RC,
                           const TargetRegisterInfo *TRI) const override;

  void loadRegFromStackSlot(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MBBI, Register DstReg,
                            int FrameIndex, const TargetRegisterClass *RC,
                            const TargetRegisterInfo *TRI) const override;

  // Materializes the given integer Val into DstReg.
  void movImm(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
              const DebugLoc &DL, Register DstReg, uint64_t Val,
              MachineInstr::MIFlag Flag = MachineInstr::NoFlags) const;

  unsigned getInstSizeInBytes(const MachineInstr &MI) const override;

  bool analyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                     MachineBasicBlock *&FBB,
                     SmallVectorImpl<MachineOperand> &Cond,
                     bool AllowModify) const override;

  unsigned insertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                        MachineBasicBlock *FBB, ArrayRef<MachineOperand> Cond,
                        const DebugLoc &dl,
                        int *BytesAdded = nullptr) const override;

  unsigned insertIndirectBranch(MachineBasicBlock &MBB,
                                MachineBasicBlock &NewDestBB,
                                const DebugLoc &DL, int64_t BrOffset,
                                RegScavenger *RS = nullptr) const override;

  unsigned removeBranch(MachineBasicBlock &MBB,
                        int *BytesRemoved = nullptr) const override;

  bool
  reverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const override;

  MachineBasicBlock *getBranchDestBlock(const MachineInstr &MI) const override;

  bool isBranchOffsetInRange(unsigned BranchOpc,
                             int64_t BrOffset) const override;

  bool isAsCheapAsAMove(const MachineInstr &MI) const override;

  bool verifyInstruction(const MachineInstr &MI,
                         StringRef &ErrInfo) const override;

  bool getMemOperandWithOffsetWidth(const MachineInstr &LdSt,
                                    const MachineOperand *&BaseOp,
                                    int64_t &Offset, unsigned &Width,
                                    const TargetRegisterInfo *TRI) const;

  bool areMemAccessesTriviallyDisjoint(const MachineInstr &MIa,
                                       const MachineInstr &MIb) const override;


  std::pair<unsigned, unsigned>
  decomposeMachineOperandsTargetFlags(unsigned TF) const override;

  ArrayRef<std::pair<unsigned, const char *>>
  getSerializableDirectMachineOperandTargetFlags() const override;

  // Return true if the function can safely be outlined from.
  virtual bool
  isFunctionSafeToOutlineFrom(MachineFunction &MF,
                              bool OutlineFromLinkOnceODRs) const override;

  // Return true if MBB is safe to outline from, and return any target-specific
  // information in Flags.
  virtual bool isMBBSafeToOutlineFrom(MachineBasicBlock &MBB,
                                      unsigned &Flags) const override;

  // Calculate target-specific information for a set of outlining candidates.
  outliner::OutlinedFunction getOutliningCandidateInfo(
      std::vector<outliner::Candidate> &RepeatedSequenceLocs) const override;

  // Return if/how a given MachineInstr should be outlined.
  virtual outliner::InstrType
  getOutliningType(MachineBasicBlock::iterator &MBBI,
                   unsigned Flags) const override;

  // Insert a custom frame for outlined functions.
  virtual void
  buildOutlinedFrame(MachineBasicBlock &MBB, MachineFunction &MF,
                     const outliner::OutlinedFunction &OF) const override;

  // Insert a call to an outlined function into a given basic block.
  virtual MachineBasicBlock::iterator
  insertOutlinedCall(Module &M, MachineBasicBlock &MBB,
                     MachineBasicBlock::iterator &It, MachineFunction &MF,
                     const outliner::Candidate &C) const override;
protected:
  const RISCVSubtarget &STI;
};

namespace RISCV {
// Match with the definitions in RISCVInstrFormatsV.td
enum RVVConstraintType {
  NoConstraint = 0,
  VS2Constraint = 0b0001,
  VS1Constraint = 0b0010,
  VMConstraint = 0b0100,
  OneInput = 0b1000,

  // Illegal instructions:
  //
  // * The destination vector register group for a masked vector instruction
  // cannot overlap the source mask register (v0), unless the destination vector
  // register is being written with a mask value (e.g., comparisons) or the
  // scalar result of a reduction.
  //
  // * Widening: The destination vector register group cannot overlap a source
  // vector register group of a different EEW
  //
  // * Narrowing: The destination vector register group cannot overlap the
  // first source vector register group
  //
  // * For vadc and vsbc, an illegal instruction exception is raised if the
  // destination vector register is v0.
  //
  // * For vmadc and vmsbc, an illegal instruction exception is raised if the
  // destination vector register overlaps a source vector register group.
  //
  // * viota: An illegal instruction exception is raised if the destination
  // vector register group overlaps the source vector mask register. If the
  // instruction is masked, an illegal instruction exception is issued if the
  // destination vector register group overlaps v0.
  //
  // * v[f]slide[1]up: The destination vector register group for vslideup cannot
  // overlap the source vector register group.
  //
  // * vrgather: The destination vector register group cannot overlap with the
  // source vector register groups.
  //
  // * vcompress: The destination vector register group cannot overlap the
  // source vector register group or the source mask register
  WidenV = VS2Constraint | VS1Constraint | VMConstraint,
  WidenW = VS1Constraint | VMConstraint,
  WidenCvt = VS2Constraint | VMConstraint | OneInput,
  Narrow = VS2Constraint | VMConstraint,
  NarrowCvt = VS2Constraint | VMConstraint | OneInput,
  Vmadc = VS2Constraint | VS1Constraint,
  Iota = VS2Constraint | VMConstraint | OneInput,
  SlideUp = VS2Constraint | VMConstraint,
  Vrgather = VS2Constraint | VS1Constraint | VMConstraint,
  Vcompress = VS2Constraint | VS1Constraint,

  ConstraintOffset = 5,
  ConstraintMask = 0b1111
};
} // end namespace RISCV

} // end namespace llvm
#endif
