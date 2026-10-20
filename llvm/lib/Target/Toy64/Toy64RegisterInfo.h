//===-- Toy64RegisterInfo.h - Toy64 Register Information -------*- C++ -*-===//

#ifndef LLVM_LIB_TARGET_TOY64_TOY64REGISTERINFO_H
#define LLVM_LIB_TARGET_TOY64_TOY64REGISTERINFO_H

#include "llvm/CodeGen/TargetRegisterInfo.h"

#define GET_REGINFO_HEADER
#include "Toy64GenRegisterInfo.inc"

namespace llvm {

struct Toy64RegisterInfo : public Toy64GenRegisterInfo {
  Toy64RegisterInfo();

  const uint16_t *
  getCalleeSavedRegs(const MachineFunction *MF = nullptr) const override;

  BitVector getReservedRegs(const MachineFunction &MF) const override;

  bool eliminateFrameIndex(
      MachineBasicBlock::iterator II,
      int SPAdj,
      unsigned FIOperandNum,
      RegScavenger *RS = nullptr) const override;

  Register getFrameRegister(const MachineFunction &MF) const override;
};

} // namespace llvm

#endif