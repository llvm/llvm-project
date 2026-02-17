#include "SC32FrameLowering.h"

using namespace llvm;

SC32FrameLowering::SC32FrameLowering()
    : TargetFrameLowering(TargetFrameLowering::StackGrowsDown, Align(4), 0) {}

void SC32FrameLowering::emitPrologue(MachineFunction &MF,
                                     MachineBasicBlock &MBB) const {}

void SC32FrameLowering::emitEpilogue(MachineFunction &MF,
                                     MachineBasicBlock &MBB) const {}

bool SC32FrameLowering::hasFPImpl(const MachineFunction &MF) const {
  return false;
}
