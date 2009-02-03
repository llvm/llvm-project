//===- SPURegisterInfo.h - Cell SPU Register Information Impl ----*- C++ -*-==//
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
#include "SPUGenRegisterInfo.h.inc"

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

    //! Return the array of callee-saved registers
    virtual const unsigned* getCalleeSavedRegs(const MachineFunction *MF) const;

    //! Return the register class array of the callee-saved registers
    virtual const TargetRegisterClass* const *
      getCalleeSavedRegClasses(const MachineFunction *MF) const;

    //! Return the reserved registers
    BitVector getReservedRegs(const MachineFunction &MF) const;

    //! Prediate: Target has dedicated frame pointer
    bool hasFP(const MachineFunction &MF) const;
    //! Eliminate the call frame setup pseudo-instructions
    void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                       MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator I) const;
    //! Convert frame indicies into machine operands
    void eliminateFrameIndex(MachineBasicBlock::iterator II, int,
                             RegScavenger *RS) const;
    //! Determine the frame's layour
    void determineFrameLayout(MachineFunction &MF) const;

    void processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                              RegScavenger *RS = NULL) const;
    //! Emit the function prologue
    void emitPrologue(MachineFunction &MF) const;
    //! Emit the function epilogue
    void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;
    //! Get return address register (LR, aka R0)
    unsigned getRARegister() const;
    //! Get the stack frame register (SP, aka R1)
    unsigned getFrameRegister(MachineFunction &MF) const;
    //! Perform target-specific stack frame setup.
    void getInitialFrameState(std::vector<MachineMove> &Moves) const;

    //------------------------------------------------------------------------
    // New methods added:
    //------------------------------------------------------------------------

    //! Return the array of argument passing registers
    /*!
      \note The size of this array is returned by getArgRegsSize().
     */
    static const unsigned *getArgRegs();

    //! Return the size of the argument passing register array
    static unsigned getNumArgRegs();

    //! Get DWARF debugging register number
    int getDwarfRegNum(unsigned RegNum, bool isEH) const;
  };
} // end namespace llvm

#endif
