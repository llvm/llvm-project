//===-- Mips16RegisterInfo.h - Mips16 Register Information ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Mips16 implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef MIPS16REGISTERINFO_H
#define MIPS16REGISTERINFO_H

#include "MipsRegisterInfo.h"

namespace llvm {
class Mips16InstrInfo;

class Mips16RegisterInfo : public MipsRegisterInfo {
  const Mips16InstrInfo &TII;
public:
  Mips16RegisterInfo(const MipsSubtarget &Subtarget, const Mips16InstrInfo &TII);

  void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I) const;
private:
  virtual void eliminateFI(MachineBasicBlock::iterator II, unsigned OpNo,
                           int FrameIndex, uint64_t StackSize,
                           int64_t SPOffset) const;
};

} // end namespace llvm

#endif
