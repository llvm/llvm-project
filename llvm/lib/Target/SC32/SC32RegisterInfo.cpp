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

static bool isCommutativeOpcode(unsigned Opcode) {
  switch (Opcode) {
  case SC32::ADD:
  case SC32::MUL:
  case SC32::AND:
  case SC32::OR:
  case SC32::XOR:
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
  const TargetRegisterInfo &RI = *SI.getRegisterInfo();
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

  int TiedDef = MI->getDesc().getOperandConstraint(FIOperandNum, MCOI::TIED_TO);
  Register TiedReg;
  Register ScratchReg = SC32::GP1;

  if (TiedDef >= 0) {
    TiedReg = MI->getOperand(TiedDef).getReg();

    if (!MI->readsRegister(TiedReg, &RI)) {
      ScratchReg = TiedReg;
    }
  }

  if (FixedOffset < -0xFFFFF) {
    BuildMI(MBB, MI, DL, II.get(SC32::LUI), ScratchReg)
        .addImm(FixedOffset >> 16);
    BuildMI(MBB, MI, DL, II.get(SC32::ORI), ScratchReg)
        .addReg(ScratchReg)
        .addImm(FixedOffset & 0xFFFF);
    BuildMI(MBB, MI, DL, II.get(SC32::ADD), ScratchReg)
        .addReg(ScratchReg)
        .addReg(FrameReg);
  } else {
    BuildMI(MBB, MI, DL, II.get(SC32::MOV), ScratchReg).addReg(FrameReg);
    BuildMI(MBB, MI, DL, II.get(SC32::SUBI), ScratchReg)
        .addReg(ScratchReg)
        .addImm(-FixedOffset);
  }

  if (TiedDef >= 0 && ScratchReg != TiedReg) {
    if (isCommutativeOpcode(MI->getOpcode()) &&
        TiedReg == MI->getOperand(2).getReg()) {
      MI->getOperand(2).setReg(ScratchReg);
      MI->getOperand(1).ChangeToRegister(TiedReg, false);
      return false;
    }

    MI->getOperand(TiedDef).setReg(ScratchReg);
    BuildMI(MBB, std::next(MI), DL, II.get(SC32::MOV), TiedReg)
        .addReg(ScratchReg);
  }

  MO.ChangeToRegister(ScratchReg, false);
  return false;
}

Register SC32RegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  return SC32::GP0;
}
