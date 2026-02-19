#ifndef LLVM_LIB_TARGET_SC32_SC32REGISTERINFO_H
#define LLVM_LIB_TARGET_SC32_SC32REGISTERINFO_H

#define GET_REGINFO_HEADER
#include "SC32GenRegisterInfo.inc"

namespace llvm {

class SC32RegisterInfo : public SC32GenRegisterInfo {
public:
  SC32RegisterInfo();

  const MCPhysReg *getCalleeSavedRegs(const MachineFunction *MF) const override;

  const uint32_t *getCallPreservedMask(const MachineFunction &MF,
                                       CallingConv::ID) const override;

  BitVector getReservedRegs(const MachineFunction &MF) const override;

  bool eliminateFrameIndex(MachineBasicBlock::iterator MI, int SPAdj,
                           unsigned FIOperandNum,
                           RegScavenger *RS) const override;

  Register getFrameRegister(const MachineFunction &MF) const override;
};

} // namespace llvm

#endif
