//===-- SPUInstrInfo.h - Cell SPU Instruction Information -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the CellSPU implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef SPU_INSTRUCTIONINFO_H
#define SPU_INSTRUCTIONINFO_H

#include "SPU.h"
#include "SPURegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "SPUGenInstrInfo.inc"

namespace llvm {
  //! Cell SPU instruction information class
  class SPUInstrInfo : public SPUGenInstrInfo {
    SPUTargetMachine &TM;
    const SPURegisterInfo RI;
  public:
    explicit SPUInstrInfo(SPUTargetMachine &tm);

    /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
    /// such, whenever a client has an instance of instruction info, it should
    /// always be able to get register info as well (through this method).
    ///
    virtual const SPURegisterInfo &getRegisterInfo() const { return RI; }

    ScheduleHazardRecognizer *
    CreateTargetHazardRecognizer(const TargetMachine *TM,
                                 const ScheduleDAG *DAG) const;

    unsigned isLoadFromStackSlot(const MachineInstr *MI,
                                 int &FrameIndex) const;
    unsigned isStoreToStackSlot(const MachineInstr *MI,
                                int &FrameIndex) const;

    virtual void copyPhysReg(MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator I, DebugLoc DL,
                             unsigned DestReg, unsigned SrcReg,
                             bool KillSrc) const;

    //! Store a register to a stack slot, based on its register class.
    virtual void storeRegToStackSlot(MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator MBBI,
                                     unsigned SrcReg, bool isKill, int FrameIndex,
                                     const TargetRegisterClass *RC,
                                     const TargetRegisterInfo *TRI) const;

    //! Load a register from a stack slot, based on its register class.
    virtual void loadRegFromStackSlot(MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator MBBI,
                                      unsigned DestReg, int FrameIndex,
                                      const TargetRegisterClass *RC,
                                      const TargetRegisterInfo *TRI) const;

    //! Reverses a branch's condition, returning false on success.
    virtual
    bool ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const;

    virtual bool AnalyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                               MachineBasicBlock *&FBB,
                               SmallVectorImpl<MachineOperand> &Cond,
                               bool AllowModify) const;

    virtual unsigned RemoveBranch(MachineBasicBlock &MBB) const;

    virtual unsigned InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                                  MachineBasicBlock *FBB,
                                  const SmallVectorImpl<MachineOperand> &Cond,
                                  DebugLoc DL) const;
   };
}

#endif
