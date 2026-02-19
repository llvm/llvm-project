#include "SC32RegisterInfo.h"
#include "MCTargetDesc/SC32MCTargetDesc.h"
#include "SC32FrameLowering.h"
#include "SC32InstrInfo.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
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

const uint32_t *
SC32RegisterInfo::getCallPreservedMask(const MachineFunction &MF,
                                       CallingConv::ID) const {
  return CSR_RegMask;
}

BitVector SC32RegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());

  Reserved.set(SC32::GP0);
  Reserved.set(SC32::GP1);
  Reserved.set(SC32::GP27);
  Reserved.set(SC32::GP29);
  Reserved.set(SC32::GP30);
  Reserved.set(SC32::GP31);

  return Reserved;
}

static bool isLoadStoreOpcode(unsigned Opcode) {
  switch (Opcode) {
  case SC32::LD:
  case SC32::LDB:
  case SC32::ST:
  case SC32::STB:
    return true;
  default:
    return false;
  }
}

bool SC32RegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator MI,
                                           int SPAdj, unsigned FIOperandNum,
                                           RegScavenger *RS) const {
  MachineOperand &MO = MI->getOperand(FIOperandNum);
  MachineBasicBlock &MBB = *MI->getParent();
  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  const TargetSubtargetInfo &SI = MF.getSubtarget();
  const TargetFrameLowering &FL = *SI.getFrameLowering();
  const TargetInstrInfo &II = *SI.getInstrInfo();
  const DebugLoc &DL = MI->getDebugLoc();

  Register FrameReg;
  StackOffset Offset = FL.getFrameIndexReference(MF, MO.getIndex(), FrameReg);
  int FixedOffset = Offset.getFixed() - MFI.getStackSize();

  assert(SPAdj == 0);
  assert(FixedOffset < 0);

  if (isLoadStoreOpcode(MI->getOpcode()) && FIOperandNum == 1) {
    int TotalOffset = FixedOffset + MI->getOperand(2).getImm();

    if (TotalOffset >= -0x4000 && TotalOffset < 0x4000) {
      MI->getOperand(1).ChangeToRegister(FrameReg, false);
      MI->getOperand(2).setImm(TotalOffset);
      return false;
    }
  }

  if (FixedOffset < -0xFFFFF) {
    BuildMI(MBB, MI, DL, II.get(SC32::LUI), SC32::GP1)
        .addImm(FixedOffset >> 16);
    BuildMI(MBB, MI, DL, II.get(SC32::ORI), SC32::GP1)
        .addReg(SC32::GP1)
        .addImm(FixedOffset & 0xFFFF);
    BuildMI(MBB, MI, DL, II.get(SC32::ADD), SC32::GP1)
        .addReg(SC32::GP1)
        .addReg(FrameReg);
  } else {
    BuildMI(MBB, MI, DL, II.get(SC32::MOV), SC32::GP1).addReg(FrameReg);
    BuildMI(MBB, MI, DL, II.get(SC32::SUBI), SC32::GP1)
        .addReg(SC32::GP1)
        .addImm(-FixedOffset);
  }

  MO.ChangeToRegister(SC32::GP1, false);
  return false;
}

Register SC32RegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  return SC32::GP0;
}
