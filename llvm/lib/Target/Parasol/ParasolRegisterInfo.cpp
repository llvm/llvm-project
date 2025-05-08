//===-- ParasolRegisterInfo.cpp - Parasol Register Information ------------===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//
//
// This file contains the Parasol implementation of the TargetRegisterInfo
// class.
//
//===----------------------------------------------------------------------===//

#include "ParasolRegisterInfo.h"
#include "ParasolSubtarget.h"
#include "llvm/Support/Debug.h"

#define GET_REGINFO_TARGET_DESC
#include "ParasolGenRegisterInfo.inc"

#define DEBUG_TYPE "parasol-reginfo"

using namespace llvm;

ParasolRegisterInfo::ParasolRegisterInfo(const ParasolSubtarget &ST)
    : ParasolGenRegisterInfo(Parasol::X1, /*DwarfFlavour*/ 0, /*EHFlavor*/ 0,
                             /*PC*/ 0),
      Subtarget(ST) {}

const MCPhysReg *
ParasolRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  return Parasol_CalleeSavedRegs_SaveList;
}

const TargetRegisterClass *
ParasolRegisterInfo::intRegClass(unsigned Size) const {
  return &Parasol::IRRegClass;
}

const uint32_t *
ParasolRegisterInfo::getCallPreservedMask(const MachineFunction &MF,
                                          CallingConv::ID) const {
  return Parasol_CalleeSavedRegs_RegMask;
}

BitVector
ParasolRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());

  markSuperRegs(Reserved, Parasol::X0); // zero
  markSuperRegs(Reserved, Parasol::X2); // sp
  markSuperRegs(Reserved, Parasol::X3); // gp
  markSuperRegs(Reserved, Parasol::X4); // tp

  return Reserved;
}

bool ParasolRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                              int SPAdj, unsigned FIOperandNum,
                                              RegScavenger *RS) const {
  llvm_unreachable("Unsupported eliminateFrameIndex");
  return false;
}

bool ParasolRegisterInfo::requiresRegisterScavenging(
    const MachineFunction &MF) const {
  return true;
}

bool ParasolRegisterInfo::requiresFrameIndexScavenging(
    const MachineFunction &MF) const {
  return true;
}

bool ParasolRegisterInfo::requiresFrameIndexReplacementScavenging(
    const MachineFunction &MF) const {
  return true;
}

bool ParasolRegisterInfo::trackLivenessAfterRegAlloc(
    const MachineFunction &MF) const {
  return true;
}

Register
ParasolRegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  // llvm_unreachable("Unsupported getFrameRegister");
  return Parasol::X8;
}
