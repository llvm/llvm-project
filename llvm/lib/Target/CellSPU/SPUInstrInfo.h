//===- SPUInstrInfo.h - Cell SPU Instruction Information --------*- C++ -*-===//
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
#include "llvm/Target/TargetInstrInfo.h"
#include "SPURegisterInfo.h"

namespace llvm {
  //! Cell SPU instruction information class
  class SPUInstrInfo : public TargetInstrInfoImpl {
    SPUTargetMachine &TM;
    const SPURegisterInfo RI;
  protected:
    virtual MachineInstr* foldMemoryOperandImpl(MachineFunction &MF,
                                            MachineInstr* MI,
                                            const SmallVectorImpl<unsigned> &Ops,
                                            int FrameIndex) const;

    virtual MachineInstr* foldMemoryOperandImpl(MachineFunction &MF,
                                                MachineInstr* MI,
                                                const SmallVectorImpl<unsigned> &Ops,
                                                MachineInstr* LoadMI) const {
      return 0;
    }

  public:
    explicit SPUInstrInfo(SPUTargetMachine &tm);

    /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
    /// such, whenever a client has an instance of instruction info, it should
    /// always be able to get register info as well (through this method).
    ///
    virtual const SPURegisterInfo &getRegisterInfo() const { return RI; }

    /// getPointerRegClass - Return the register class to use to hold pointers.
    /// This is used for addressing modes.
    virtual const TargetRegisterClass *getPointerRegClass() const;

    /// Return true if the instruction is a register to register move and return
    /// the source and dest operands and their sub-register indices by reference.
    virtual bool isMoveInstr(const MachineInstr &MI,
                             unsigned &SrcReg, unsigned &DstReg,
                             unsigned &SrcSubIdx, unsigned &DstSubIdx) const;

    unsigned isLoadFromStackSlot(const MachineInstr *MI,
                                 int &FrameIndex) const;
    unsigned isStoreToStackSlot(const MachineInstr *MI,
                                int &FrameIndex) const;

    virtual bool copyRegToReg(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MI,
                              unsigned DestReg, unsigned SrcReg,
                              const TargetRegisterClass *DestRC,
                              const TargetRegisterClass *SrcRC) const;

    //! Store a register to a stack slot, based on its register class.
    virtual void storeRegToStackSlot(MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator MBBI,
                                     unsigned SrcReg, bool isKill, int FrameIndex,
                                     const TargetRegisterClass *RC) const;

    //! Store a register to an address, based on its register class
    virtual void storeRegToAddr(MachineFunction &MF, unsigned SrcReg, bool isKill,
                                                  SmallVectorImpl<MachineOperand> &Addr,
                                                  const TargetRegisterClass *RC,
                                                  SmallVectorImpl<MachineInstr*> &NewMIs) const;

    //! Load a register from a stack slot, based on its register class.
    virtual void loadRegFromStackSlot(MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator MBBI,
                                      unsigned DestReg, int FrameIndex,
                                      const TargetRegisterClass *RC) const;

    //! Loqad a register from an address, based on its register class
    virtual void loadRegFromAddr(MachineFunction &MF, unsigned DestReg,
                                                         SmallVectorImpl<MachineOperand> &Addr,
                                                         const TargetRegisterClass *RC,
                                 SmallVectorImpl<MachineInstr*> &NewMIs) const;

    //! Return true if the specified load or store can be folded
    virtual
    bool canFoldMemoryOperand(const MachineInstr *MI,
                              const SmallVectorImpl<unsigned> &Ops) const;

    //! Return true if the specified block does not fall through
    virtual bool BlockHasNoFallThrough(const MachineBasicBlock &MBB) const;

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
                              const SmallVectorImpl<MachineOperand> &Cond) const;
   };
}

#endif
