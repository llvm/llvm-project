//===-- DirectXRegisterInfo.h - Define RegisterInfo for DirectX -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the DirectX specific subclass of TargetRegisterInfo.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DIRECTX_DXILREGISTERINFO_H
#define LLVM_DIRECTX_DXILREGISTERINFO_H

#include "llvm/CodeGen/TargetRegisterInfo.h"

#define GET_REGINFO_HEADER
#include "DirectXGenRegisterInfo.inc"

namespace llvm {
struct DirectXRegisterInfo : public DirectXGenRegisterInfo {
  DirectXRegisterInfo() : DirectXGenRegisterInfo(0) {}
  ~DirectXRegisterInfo();

  const MCPhysReg *getCalleeSavedRegs(const MachineFunction *MF) const override;
  BitVector getReservedRegs(const MachineFunction &MF) const override;
  bool eliminateFrameIndex(MachineBasicBlock::iterator II, int SPAdj,
                           unsigned FIOperandNum,
                           RegScavenger *RS = nullptr) const override;
  // Debug information queries.
  Register getFrameRegister(const MachineFunction &MF) const override;
};
} // namespace llvm

#endif // LLVM_DIRECTX_DXILREGISTERINFO_H
