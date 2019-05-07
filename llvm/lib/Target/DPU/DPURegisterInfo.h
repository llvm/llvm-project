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

// Aliases for the general purpose registers used for special purpose:
//  - R_RADD : functions' return addresses (i.e. where the PC was before
//  calling)
//  - R_STKP : stack pointer register
//  - R_RVAL : value returned by functions
#define R_RADD DPU::RADD
#define R_STKP DPU::STKP
#define R_RVAL DPU::RVAL
#define R_RVAL1 DPU::R20r
#define R_RDVAL DPU::RDVAL

namespace llvm {
struct DPURegisterInfo : public DPUGenRegisterInfo {
public:
  DPURegisterInfo();

  /// Code Generation virtual methods...
  const MCPhysReg *
  getCalleeSavedRegs(const MachineFunction *MF = nullptr) const override;
  BitVector getReservedRegs(const MachineFunction &MF) const override;
  void eliminateFrameIndex(MachineBasicBlock::iterator II, int SPAdj,
                           unsigned FIOperandNum,
                           RegScavenger *RS) const override;
  unsigned getFrameRegister(const MachineFunction &MF) const override;
  // We use a unique, function independent, frame register (which is the stack
  // pointer).
  static unsigned getFrameRegister();

  // This module is in charge of computing the actual stack adjustment upon
  // function calls, when eliminating the frame index. To do this, it needs to
  // be aware of whether the stack pointer is pushed onto the stack before such
  // a call (in O0). It must also consider 64 bits alignments required to
  // support 64 bits emulation. This function returns the actual overhead in
  // bytes to add to the stack when adjusting it.
  unsigned int
  stackAdjustmentDependingOnOptimizationLevel(const MachineFunction &MF,
                                              unsigned int FrameSize) const;
};
} // namespace llvm
#endif
