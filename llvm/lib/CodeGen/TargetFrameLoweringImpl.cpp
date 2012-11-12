//===----- TargetFrameLoweringImpl.cpp - Implement target frame interface --==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements the layout of a stack frame on the target machine.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"

#include <cstdlib>
using namespace llvm;

TargetFrameLowering::~TargetFrameLowering() {
}

/// getFrameIndexOffset - Returns the displacement from the frame register to
/// the stack frame of the specified index. This is the default implementation
/// which is overridden for some targets.
int TargetFrameLowering::getFrameIndexOffset(const MachineFunction &MF,
                                         int FI) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  return MFI->getObjectOffset(FI) + MFI->getStackSize() -
    getOffsetOfLocalArea() + MFI->getOffsetAdjustment();
}

int TargetFrameLowering::getFrameIndexReference(const MachineFunction &MF,
                                             int FI, unsigned &FrameReg) const {
  const TargetRegisterInfo *RI = MF.getTarget().getRegisterInfo();

  // By default, assume all frame indices are referenced via whatever
  // getFrameRegister() says. The target can override this if it's doing
  // something different.
  FrameReg = RI->getFrameRegister(MF);
  return getFrameIndexOffset(MF, FI);
}
