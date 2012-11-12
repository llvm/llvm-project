//===-- SPURegisterInfo.h - Cell SPU Register Information Impl --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Cell SPU implementation of the TargetRegisterInfo
// class.
//
//===----------------------------------------------------------------------===//

#ifndef SPU_REGISTERINFO_H
#define SPU_REGISTERINFO_H

#include "SPU.h"

#define GET_REGINFO_HEADER
#include "SPUGenRegisterInfo.inc"

namespace llvm {
  class SPUSubtarget;
  class TargetInstrInfo;
  class Type;

  class SPURegisterInfo : public SPUGenRegisterInfo {
  private:
    const SPUSubtarget &Subtarget;
    const TargetInstrInfo &TII;

    //! Predicate: Does the machine function use the link register?
    bool usesLR(MachineFunction &MF) const;

  public:
    SPURegisterInfo(const SPUSubtarget &subtarget, const TargetInstrInfo &tii);
 
    //! Translate a register's enum value to a register number
    /*!
      This method translates a register's enum value to it's regiser number,
      e.g. SPU::R14 -> 14.
     */
    static unsigned getRegisterNumbering(unsigned RegEnum);

    /// getPointerRegClass - Return the register class to use to hold pointers.
    /// This is used for addressing modes.
    virtual const TargetRegisterClass *
    getPointerRegClass(const MachineFunction &MF, unsigned Kind = 0) const;

    /// After allocating this many registers, the allocator should feel
    /// register pressure. The value is a somewhat random guess, based on the
    /// number of non callee saved registers in the C calling convention.
    virtual unsigned getRegPressureLimit( const TargetRegisterClass *RC,
                                          MachineFunction &MF) const{
      return 50;
    }

    //! Return the array of callee-saved registers
    virtual const uint16_t* getCalleeSavedRegs(const MachineFunction *MF) const;

    //! Allow for scavenging, so we can get scratch registers when needed.
    virtual bool requiresRegisterScavenging(const MachineFunction &MF) const
    { return true; }

    //! Enable tracking of liveness after register allocation, since register
    // scavenging is enabled.
    virtual bool trackLivenessAfterRegAlloc(const MachineFunction &MF) const
    { return true; }

    //! Return the reserved registers
    BitVector getReservedRegs(const MachineFunction &MF) const;

    //! Eliminate the call frame setup pseudo-instructions
    void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                       MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator I) const;
    //! Convert frame indicies into machine operands
    void eliminateFrameIndex(MachineBasicBlock::iterator II, int SPAdj,
                             RegScavenger *RS = NULL) const;

    //! Get the stack frame register (SP, aka R1)
    unsigned getFrameRegister(const MachineFunction &MF) const;

    //------------------------------------------------------------------------
    // New methods added:
    //------------------------------------------------------------------------

    //! Convert D-form load/store to X-form load/store
    /*!
      Converts a regiser displacement load/store into a register-indexed
      load/store for large stack frames, when the stack frame exceeds the
      range of a s10 displacement.
     */
    int convertDFormToXForm(int dFormOpcode) const;

    //! Acquire an unused register in an emergency.
    unsigned findScratchRegister(MachineBasicBlock::iterator II,
                                 RegScavenger *RS,
                                 const TargetRegisterClass *RC, 
                                 int SPAdj) const;
    
  };
} // end namespace llvm

#endif
