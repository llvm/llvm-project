//===-- Toy64FrameLowering.h - Toy64 Frame Lowering -----------*- C++ -*-===//

#ifndef LLVM_LIB_TARGET_TOY64_TOY64FRAMELOWERING_H
#define LLVM_LIB_TARGET_TOY64_TOY64FRAMELOWERING_H

#include "llvm/CodeGen/TargetFrameLowering.h"

namespace llvm {

class MachineFunction;
class MachineBasicBlock;

class Toy64FrameLowering : public TargetFrameLowering {
public:
  Toy64FrameLowering()
      : TargetFrameLowering(
            StackGrowsDown,
            /*StackAlignment=*/Align(16),
            /*LocalAreaOffset=*/0,
            /*TransientStackAlignment=*/Align(16)) {}

  void emitPrologue(
      MachineFunction &MF,
      MachineBasicBlock &MBB) const override;

  void emitEpilogue(
      MachineFunction &MF,
      MachineBasicBlock &MBB) const override;

protected:
  bool hasFPImpl(const MachineFunction &MF) const override {
    return false;
  }
};

} // namespace llvm

#endif