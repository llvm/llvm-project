#include "SC32RegisterInfo.h"
#include "MCTargetDesc/SC32MCTargetDesc.h"
#include "SC32FrameLowering.h"
#include "SC32InstrInfo.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"

using namespace llvm;

#define GET_REGINFO_TARGET_DESC
#include "SC32GenRegisterInfo.inc"

SC32RegisterInfo::SC32RegisterInfo() : SC32GenRegisterInfo(SC32::NoRegister) {}

const MCPhysReg *
SC32RegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  return CSR_SaveList;
}

BitVector SC32RegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());

  Reserved.set(SC32::GP0);
  Reserved.set(SC32::GP1);
  Reserved.set(SC32::GP27);
  Reserved.set(SC32::GP28);
  Reserved.set(SC32::GP29);
  Reserved.set(SC32::GP30);
  Reserved.set(SC32::GP31);

  return Reserved;
}

bool SC32RegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator MI,
                                           int SPAdj, unsigned FIOperandNum,
                                           RegScavenger *RS) const {
  MachineOperand &MO = MI->getOperand(FIOperandNum);
  MachineBasicBlock &MBB = *MI->getParent();
  MachineFunction &MF = *MBB.getParent();
  const TargetSubtargetInfo &SI = MF.getSubtarget();
  const TargetFrameLowering &FL = *SI.getFrameLowering();
  const TargetInstrInfo &II = *SI.getInstrInfo();
  const DebugLoc &DL = MI->getDebugLoc();

  Register FrameReg;
  StackOffset Offset = FL.getFrameIndexReference(MF, MO.getIndex(), FrameReg);
  int FixedOffset = Offset.getFixed() / 4;

  if (FixedOffset > 0) {
    if (FixedOffset > 0xFFFF) {
      BuildMI(MBB, MI, DL, II.get(SC32::LUI), SC32::GP1)
          .addImm(FixedOffset >> 16);
      BuildMI(MBB, MI, DL, II.get(SC32::ORI), SC32::GP1)
          .addReg(SC32::GP1)
          .addImm(FixedOffset & 0xFFFF);
    } else {
      BuildMI(MBB, MI, DL, II.get(SC32::LLI), SC32::GP1)
          .addImm(FixedOffset & 0xFFFF);
    }

    BuildMI(MBB, MI, DL, II.get(SC32::ADD), SC32::GP1)
        .addReg(SC32::GP1)
        .addReg(FrameReg);

    MO.ChangeToRegister(SC32::GP1, false);
  } else {
    MO.ChangeToRegister(SC32::GP0, false);
  }

  return false;
}

Register SC32RegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  return SC32::GP0;
}
