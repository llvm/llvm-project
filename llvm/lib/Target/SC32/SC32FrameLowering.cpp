#include "SC32FrameLowering.h"
#include "MCTargetDesc/SC32MCTargetDesc.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"

using namespace llvm;

SC32FrameLowering::SC32FrameLowering()
    : TargetFrameLowering(TargetFrameLowering::StackGrowsDown, Align(4), 0) {}

void SC32FrameLowering::emitPrologue(MachineFunction &MF,
                                     MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MI = MBB.begin();
  DebugLoc DL;

  const TargetSubtargetInfo &TSI = MF.getSubtarget();
  const TargetInstrInfo &TII = *TSI.getInstrInfo();
  const TargetRegisterInfo &TRI = *TSI.getRegisterInfo();

  Register FrameReg = TRI.getFrameRegister(MF);
  int StackSize = MF.getFrameInfo().getStackSize();

  if (StackSize > 0) {
    BuildMI(MBB, MI, DL, TII.get(SC32::PUSH)).addReg(FrameReg);
    BuildMI(MBB, MI, DL, TII.get(SC32::MOV), FrameReg).addReg(SC32::GP29);

    if (StackSize > 0xFFFFF) {
      BuildMI(MBB, MI, DL, TII.get(SC32::LUI), SC32::GP1)
          .addImm(StackSize >> 16);
      BuildMI(MBB, MI, DL, TII.get(SC32::ORI), SC32::GP1)
          .addReg(SC32::GP1)
          .addImm(StackSize & 0xFFFF);
      BuildMI(MBB, MI, DL, TII.get(SC32::SUB), SC32::GP29)
          .addReg(SC32::GP29)
          .addReg(SC32::GP1);
    } else {
      BuildMI(MBB, MI, DL, TII.get(SC32::SUBI), SC32::GP29)
          .addReg(SC32::GP29)
          .addImm(StackSize);
    }
  }
}

void SC32FrameLowering::emitEpilogue(MachineFunction &MF,
                                     MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MI = MBB.getLastNonDebugInstr();
  DebugLoc DL = MI->getDebugLoc();

  const TargetSubtargetInfo &TSI = MF.getSubtarget();
  const TargetInstrInfo &TII = *TSI.getInstrInfo();
  const TargetRegisterInfo &TRI = *TSI.getRegisterInfo();

  Register FrameReg = TRI.getFrameRegister(MF);
  int StackSize = MF.getFrameInfo().getStackSize();

  if (StackSize > 0) {
    BuildMI(MBB, MI, DL, TII.get(SC32::MOV), SC32::GP29).addReg(FrameReg);
    BuildMI(MBB, MI, DL, TII.get(SC32::POP)).addReg(FrameReg);
  }
}

bool SC32FrameLowering::hasFPImpl(const MachineFunction &MF) const {
  return false;
}
