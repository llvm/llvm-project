//===- NVPTXInstrInfo.h - NVPTX Instruction Information----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the niversity of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the NVPTX implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef NVPTXINSTRUCTIONINFO_H
#define NVPTXINSTRUCTIONINFO_H

#include "NVPTX.h"
#include "NVPTXRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "NVPTXGenInstrInfo.inc"

namespace llvm {

class NVPTXInstrInfo : public NVPTXGenInstrInfo
{
  NVPTXTargetMachine &TM;
  const NVPTXRegisterInfo RegInfo;
public:
  explicit NVPTXInstrInfo(NVPTXTargetMachine &TM);

  virtual const NVPTXRegisterInfo &getRegisterInfo() const { return RegInfo; }

  /* The following virtual functions are used in register allocation.
   * They are not implemented because the existing interface and the logic
   * at the caller side do not work for the elementized vector load and store.
   *
   * virtual unsigned isLoadFromStackSlot(const MachineInstr *MI,
   *                                  int &FrameIndex) const;
   * virtual unsigned isStoreToStackSlot(const MachineInstr *MI,
   *                                 int &FrameIndex) const;
   * virtual void storeRegToStackSlot(MachineBasicBlock &MBB,
   *                              MachineBasicBlock::iterator MBBI,
   *                             unsigned SrcReg, bool isKill, int FrameIndex,
   *                              const TargetRegisterClass *RC) const;
   * virtual void loadRegFromStackSlot(MachineBasicBlock &MBB,
   *                               MachineBasicBlock::iterator MBBI,
   *                               unsigned DestReg, int FrameIndex,
   *                               const TargetRegisterClass *RC) const;
   */

  virtual void copyPhysReg(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator I, DebugLoc DL,
                           unsigned DestReg, unsigned SrcReg,
                           bool KillSrc) const ;
  virtual bool isMoveInstr(const MachineInstr &MI,
                           unsigned &SrcReg,
                           unsigned &DestReg) const;
  bool isLoadInstr(const MachineInstr &MI, unsigned &AddrSpace) const;
  bool isStoreInstr(const MachineInstr &MI, unsigned &AddrSpace) const;
  bool isReadSpecialReg(MachineInstr &MI) const;

  virtual bool CanTailMerge(const MachineInstr *MI) const ;
  // Branch analysis.
  virtual bool AnalyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                             MachineBasicBlock *&FBB,
                             SmallVectorImpl<MachineOperand> &Cond,
                             bool AllowModify) const;
  virtual unsigned RemoveBranch(MachineBasicBlock &MBB) const;
  virtual unsigned InsertBranch(MachineBasicBlock &MBB,MachineBasicBlock *TBB,
                                MachineBasicBlock *FBB,
                                const SmallVectorImpl<MachineOperand> &Cond,
                                DebugLoc DL) const;
  unsigned getLdStCodeAddrSpace(const MachineInstr &MI) const {
    return  MI.getOperand(2).getImm();
  }

};

} // namespace llvm

#endif
