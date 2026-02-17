#ifndef LLVM_LIB_TARGET_SC32_SC32FRAMELOWERING_H
#define LLVM_LIB_TARGET_SC32_SC32FRAMELOWERING_H

#include "llvm/CodeGen/TargetFrameLowering.h"

namespace llvm {

class SC32FrameLowering : public TargetFrameLowering {
public:
  SC32FrameLowering();

  void emitPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const override;

  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const override;

  bool hasFPImpl(const MachineFunction &MF) const override;
};

} // namespace llvm

#endif
