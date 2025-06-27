/* --- PEFrameLowering.h --- */

/* ------------------------------------------
Author: 高宇翔
Date: 4/2/2025
------------------------------------------ */

#ifndef PEFRAMELOWERING_H
#define PEFRAMELOWERING_H

#include "llvm/CodeGen/TargetFrameLowering.h"

namespace llvm {

class PESubtarget;

class PEFrameLowering : public TargetFrameLowering {
  const PESubtarget &STI;

public:
  explicit PEFrameLowering(const PESubtarget &STI)
      : TargetFrameLowering(StackGrowsDown, Align(16), 0, Align(16)), STI(STI) {
  }

  // 函数的汇编代码开头序言
  void emitPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const override;

  // 函数的汇编代码结尾序言
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const override;

  bool hasReservedCallFrame(const MachineFunction &MF) const override {
    return true;
  }

protected:
  bool hasFPImpl(const MachineFunction &MF) const override;

private:
  uint64_t computeStackSize(MachineFunction &MF) const;
};
} // namespace llvm

#endif // PEFRAMELOWERING_H
