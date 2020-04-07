//===-- SystemZRegisterInfo.h - SystemZ register information ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZREGISTERINFO_H
#define LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZREGISTERINFO_H

#include "SystemZ.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

#define GET_REGINFO_HEADER
#include "SystemZGenRegisterInfo.inc"

namespace llvm {

class LiveIntervals;

namespace SystemZ {
// Return the subreg to use for referring to the even and odd registers
// in a GR128 pair.  Is32Bit says whether we want a GR32 or GR64.
inline unsigned even128(bool Is32bit) {
  return Is32bit ? subreg_hl32 : subreg_h64;
}
inline unsigned odd128(bool Is32bit) {
  return Is32bit ? subreg_l32 : subreg_l64;
}

// Reg should be a 32-bit GPR.  Return true if it is a high register rather
// than a low register.
inline bool isHighReg(unsigned int Reg) {
  if (SystemZ::GRH32BitRegClass.contains(Reg))
    return true;
  assert(SystemZ::GR32BitRegClass.contains(Reg) && "Invalid GRX32");
  return false;
}
} // end namespace SystemZ

struct SystemZRegisterInfo : public SystemZGenRegisterInfo {
public:
  SystemZRegisterInfo();

  /// getPointerRegClass - Return the register class to use to hold pointers.
  /// This is currently only used by LOAD_STACK_GUARD, which requires a non-%r0
  /// register, hence ADDR64.
  const TargetRegisterClass *
  getPointerRegClass(const MachineFunction &MF,
                     unsigned Kind=0) const override {
    return &SystemZ::ADDR64BitRegClass;
  }

  /// getCrossCopyRegClass - Returns a legal register class to copy a register
  /// in the specified class to or from. Returns NULL if it is possible to copy
  /// between a two registers of the specified class.
  const TargetRegisterClass *
  getCrossCopyRegClass(const TargetRegisterClass *RC) const override;

  bool getRegAllocationHints(Register VirtReg, ArrayRef<MCPhysReg> Order,
                             SmallVectorImpl<MCPhysReg> &Hints,
                             const MachineFunction &MF, const VirtRegMap *VRM,
                             const LiveRegMatrix *Matrix) const override;

  // Override TargetRegisterInfo.h.
  bool requiresRegisterScavenging(const MachineFunction &MF) const override {
    return true;
  }
  bool requiresFrameIndexScavenging(const MachineFunction &MF) const override {
    return true;
  }
  const MCPhysReg *getCalleeSavedRegs(const MachineFunction *MF) const override;
  const uint32_t *getCallPreservedMask(const MachineFunction &MF,
                                       CallingConv::ID CC) const override;
  BitVector getReservedRegs(const MachineFunction &MF) const override;
  void eliminateFrameIndex(MachineBasicBlock::iterator MI,
                           int SPAdj, unsigned FIOperandNum,
                           RegScavenger *RS) const override;

  /// SrcRC and DstRC will be morphed into NewRC if this returns true.
 bool shouldCoalesce(MachineInstr *MI,
                      const TargetRegisterClass *SrcRC,
                      unsigned SubReg,
                      const TargetRegisterClass *DstRC,
                      unsigned DstSubReg,
                      const TargetRegisterClass *NewRC,
                      LiveIntervals &LIS) const override;

  Register getFrameRegister(const MachineFunction &MF) const override;
};

} // end namespace llvm

#endif
