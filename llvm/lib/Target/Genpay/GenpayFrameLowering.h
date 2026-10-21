#ifndef LLVM_LIB_TARGET_GENPAY_GENPAYFRAMELOWERING_H
#define LLVM_LIB_TARGET_GENPAY_GENPAYFRAMELOWERING_H

#include "llvm/CodeGen/TargetFrameLowering.h"

namespace llvm {

class GenpayFrameLowering : public TargetFrameLowering {
public:
  GenpayFrameLowering();

  void emitPrologue(MachineFunction &MF,
                    MachineBasicBlock &MBB) const override {}
  void emitEpilogue(MachineFunction &MF,
                    MachineBasicBlock &MBB) const override {}
  bool hasReservedCallFrame(const MachineFunction &MF) const override {
    return true;
  }

protected:
  // LLVM 21.x: hasFP() is a non-virtual wrapper; targets override hasFPImpl().
  bool hasFPImpl(const MachineFunction &MF) const override { return false; }
};

} // namespace llvm

#endif
