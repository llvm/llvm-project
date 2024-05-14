#include "InArchRegisterInfo.h"
#include "InArch.h"
#include "InArchInstrInfo.h"
//#include "InArchMachineFunctionInfo.h"
#include "InArchSubtarget.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

using namespace llvm;

#define DEBUG_TYPE "InArch-reg-info"

#define GET_REGINFO_TARGET_DESC
#include "InArchGenRegisterInfo.inc"

InArchRegisterInfo::InArchRegisterInfo() : InArchGenRegisterInfo(InArch::R0) {}

const MCPhysReg *
InArchRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  return CSR_InArch_SaveList;
}

// TODO: check cconv
BitVector InArchRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  InArchFrameLowering const *TFI = getFrameLowering(MF);

  BitVector Reserved(getNumRegs());
  Reserved.set(InArch::GP);
  Reserved.set(InArch::SP);

  if (TFI->hasFP(MF)) {
    Reserved.set(InArch::FP);
  }
  if (TFI->hasBP(MF)) {
    Reserved.set(InArch::BP);
  }
  return Reserved;
}

bool InArchRegisterInfo::requiresRegisterScavenging(
    const MachineFunction &MF) const {
  return false; // TODO: what for?
}

// TODO: rewrite!
bool InArchRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                           int SPAdj, unsigned FIOperandNum,
                                           RegScavenger *RS) const {
  assert(SPAdj == 0 && "Unexpected non-zero SPAdj value");

  MachineInstr &MI = *II;
  MachineFunction &MF = *MI.getParent()->getParent();
  DebugLoc DL = MI.getDebugLoc();

  int FrameIndex = MI.getOperand(FIOperandNum).getIndex();
  Register FrameReg;
  int Offset = getFrameLowering(MF)
                   ->getFrameIndexReference(MF, FrameIndex, FrameReg)
                   .getFixed();
  Offset += MI.getOperand(FIOperandNum + 1).getImm();

  if (!isInt<16>(Offset)) {
    llvm_unreachable("");
  }

  MI.getOperand(FIOperandNum).ChangeToRegister(FrameReg, false, false, false);
  MI.getOperand(FIOperandNum + 1).ChangeToImmediate(Offset);
  return false;
}

Register InArchRegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  const TargetFrameLowering *TFI = getFrameLowering(MF);
  return TFI->hasFP(MF) ? InArch::FP : InArch::SP;
}

const uint32_t *
InArchRegisterInfo::getCallPreservedMask(const MachineFunction &MF,
                                       CallingConv::ID CC) const {
  return CSR_InArch_RegMask;
}