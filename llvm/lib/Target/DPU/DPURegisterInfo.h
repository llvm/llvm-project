//===-- DPURegisterInfo.cpp - DPU Register Information --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the DPU implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIB_TARGET_DPU_DPUREGISTERINFO_H
#define LLVM_LIB_TARGET_DPU_DPUREGISTERINFO_H

#include "llvm/CodeGen/TargetRegisterInfo.h"

#define GET_REGINFO_HEADER
#include "DPUGenRegisterInfo.inc"

namespace llvm {
struct DPURegisterInfo : public DPUGenRegisterInfo {
public:
  DPURegisterInfo();

  const MCPhysReg *
  getCalleeSavedRegs(const MachineFunction *MF = nullptr) const override;

  BitVector getReservedRegs(const MachineFunction &MF) const override;

  void eliminateFrameIndex(MachineBasicBlock::iterator II, int SPAdj,
                           unsigned FIOperandNum,
                           RegScavenger *RS) const override;

  unsigned getFrameRegister(const MachineFunction &MF) const override;

  const uint32_t *getCallPreservedMask(const MachineFunction &MF,
                                       CallingConv::ID) const override;
};
} // namespace llvm
#endif
