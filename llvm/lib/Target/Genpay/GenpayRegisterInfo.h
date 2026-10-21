#ifndef LLVM_LIB_TARGET_GENPAY_GENPAYREGISTERINFO_H
#define LLVM_LIB_TARGET_GENPAY_GENPAYREGISTERINFO_H

#include "MCTargetDesc/GenpayMCTargetDesc.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/MC/MCRegisterInfo.h"

#define GET_REGINFO_HEADER
#include "GenpayGenRegisterInfo.inc"

namespace llvm {

class MachineFunction;
class RegScavenger;

class GenpayRegisterInfo : public GenpayGenRegisterInfo {
public:
  GenpayRegisterInfo();

  const MCPhysReg *
  getCalleeSavedRegs(const MachineFunction *MF) const override;

  BitVector getReservedRegs(const MachineFunction &MF) const override;

  bool eliminateFrameIndex(MachineBasicBlock::iterator II, int SPAdj,
                           unsigned FIOperandNum,
                           RegScavenger *RS = nullptr) const override;

  Register getFrameRegister(const MachineFunction &MF) const override;
};

} // namespace llvm

#endif
