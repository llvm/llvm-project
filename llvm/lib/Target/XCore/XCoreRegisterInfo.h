//===-- XCoreRegisterInfo.h - XCore Register Information Impl ---*- C++ -*-===//
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

#define GET_REGINFO_HEADER
#include "XCoreGenRegisterInfo.inc"

namespace llvm {

class TargetInstrInfo;

struct XCoreRegisterInfo : public XCoreGenRegisterInfo {
private:
  const TargetInstrInfo &TII;

  void loadConstant(MachineBasicBlock &MBB,
                  MachineBasicBlock::iterator I,
                  unsigned DstReg, int64_t Value, DebugLoc dl) const;

  void storeToStack(MachineBasicBlock &MBB,
                  MachineBasicBlock::iterator I,
                  unsigned SrcReg, int Offset, DebugLoc dl) const;

  void loadFromStack(MachineBasicBlock &MBB,
                  MachineBasicBlock::iterator I,
                  unsigned DstReg, int Offset, DebugLoc dl) const;

public:
  XCoreRegisterInfo(const TargetInstrInfo &tii);

  /// Code Generation virtual methods...

  const uint16_t *getCalleeSavedRegs(const MachineFunction *MF = 0) const;

  BitVector getReservedRegs(const MachineFunction &MF) const;
  
  bool requiresRegisterScavenging(const MachineFunction &MF) const;

  bool trackLivenessAfterRegAlloc(const MachineFunction &MF) const;

  bool useFPForScavengingIndex(const MachineFunction &MF) const;

  void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I) const;

  void eliminateFrameIndex(MachineBasicBlock::iterator II,
                           int SPAdj, RegScavenger *RS = NULL) const;

  // Debug information queries.
  unsigned getFrameRegister(const MachineFunction &MF) const;

  //! Return whether to emit frame moves
  static bool needsFrameMoves(const MachineFunction &MF);
};

} // end namespace llvm

#endif
