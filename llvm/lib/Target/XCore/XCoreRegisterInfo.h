//===- XCoreRegisterInfo.h - XCore Register Information Impl ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the XCore implementation of the MRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef XCOREREGISTERINFO_H
#define XCOREREGISTERINFO_H

#include "llvm/Target/TargetRegisterInfo.h"
#include "XCoreGenRegisterInfo.h.inc"

namespace llvm {

class TargetInstrInfo;

struct XCoreRegisterInfo : public XCoreGenRegisterInfo {
private:
  const TargetInstrInfo &TII;

  void loadConstant(MachineBasicBlock &MBB,
                  MachineBasicBlock::iterator I,
                  unsigned DstReg, int64_t Value) const;

  void storeToStack(MachineBasicBlock &MBB,
                  MachineBasicBlock::iterator I,
                  unsigned SrcReg, int Offset) const;

  void loadFromStack(MachineBasicBlock &MBB,
                  MachineBasicBlock::iterator I,
                  unsigned DstReg, int Offset) const;

public:
  XCoreRegisterInfo(const TargetInstrInfo &tii);

  /// Code Generation virtual methods...

  const unsigned *getCalleeSavedRegs(const MachineFunction *MF = 0) const;

  const TargetRegisterClass* const* getCalleeSavedRegClasses(
                                     const MachineFunction *MF = 0) const;

  BitVector getReservedRegs(const MachineFunction &MF) const;
  
  bool requiresRegisterScavenging(const MachineFunction &MF) const;

  bool hasFP(const MachineFunction &MF) const;

  void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I) const;

  void eliminateFrameIndex(MachineBasicBlock::iterator II,
                           int SPAdj, RegScavenger *RS = NULL) const;
                           
  void processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                                RegScavenger *RS = NULL) const;

  void processFunctionBeforeFrameFinalized(MachineFunction &MF) const;

  void emitPrologue(MachineFunction &MF) const;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;
  
  // Debug information queries.
  unsigned getRARegister() const;
  unsigned getFrameRegister(MachineFunction &MF) const;
  void getInitialFrameState(std::vector<MachineMove> &Moves) const;

  //! Return the array of argument passing registers
  /*!
    \note The size of this array is returned by getArgRegsSize().
    */
  static const unsigned *getArgRegs(const MachineFunction *MF = 0);

  //! Return the size of the argument passing register array
  static unsigned getNumArgRegs(const MachineFunction *MF = 0);
  
  //! Return whether to emit frame moves
  static bool needsFrameMoves(const MachineFunction &MF);

  //! Get DWARF debugging register number
  int getDwarfRegNum(unsigned RegNum, bool isEH) const;
};

} // end namespace llvm

#endif
