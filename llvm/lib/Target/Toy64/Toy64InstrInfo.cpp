//===-- Toy64InstrInfo.cpp - Toy64 Instruction Information ----------------===//

#include "Toy64InstrInfo.h"
#include "Toy64Subtarget.h"

#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/ErrorHandling.h"

#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineMemOperand.h"

#define GET_INSTRINFO_CTOR_DTOR
#include "Toy64GenInstrInfo.inc"

using namespace llvm;

Toy64InstrInfo::Toy64InstrInfo(
    const Toy64Subtarget &STI,
    const Toy64RegisterInfo &RI)
    : Toy64GenInstrInfo(
          STI,
          RI,
          Toy64::ADJCALLSTACKDOWN,
          Toy64::ADJCALLSTACKUP) {}


void Toy64InstrInfo::copyPhysReg(
    MachineBasicBlock &MBB,
    MachineBasicBlock::iterator I,
    const DebugLoc &DL,
    Register DestReg,
    Register SrcReg,
    bool KillSrc,
    bool RenamableDest,
    bool RenamableSrc) const {

  // Toy64 currently has only one register class: GPR.
  //
  // Copy:
  //
  //     DestReg = SrcReg
  //
  // using:
  //
  //     ADDI DestReg, SrcReg, 0
  //
  if (Toy64::GPRRegClass.contains(DestReg) &&
      Toy64::GPRRegClass.contains(SrcReg)) {

    BuildMI(
        MBB,
        I,
        DL,
        get(Toy64::ADDI),DestReg)
        .addReg(SrcReg,getKillRegState(KillSrc))
        .addImm(0);

    return;
  }

  llvm_unreachable(
      "unsupported Toy64 physical register copy");
}

void Toy64InstrInfo::storeRegToStackSlot(
    MachineBasicBlock &MBB,
    MachineBasicBlock::iterator MI,
    Register SrcReg,
    bool IsKill,
    int FrameIndex,
    const TargetRegisterClass *RC,
    Register VReg,
    MachineInstr::MIFlag Flags) const {

  assert(
      RC == &Toy64::GPRRegClass &&
      "Toy64 can currently spill only GPRs");

  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo &MFI = MF.getFrameInfo();

  MachineMemOperand *MMO =
      MF.getMachineMemOperand(
          MachinePointerInfo::getFixedStack(
              MF,
              FrameIndex),
          MachineMemOperand::MOStore,
          MFI.getObjectSize(FrameIndex),
          MFI.getObjectAlign(FrameIndex));

  BuildMI(
      MBB,
      MI,
      DebugLoc(),
      get(Toy64::SD))
      .addReg(
          SrcReg,
          getKillRegState(IsKill))
      .addFrameIndex(FrameIndex)
      .addImm(0)
      .addMemOperand(MMO)
      .setMIFlag(Flags);
}


void Toy64InstrInfo::loadRegFromStackSlot(
    MachineBasicBlock &MBB,
    MachineBasicBlock::iterator MI,
    Register DestReg,
    int FrameIndex,
    const TargetRegisterClass *RC,
    Register VReg,
    unsigned SubReg,
    MachineInstr::MIFlag Flags) const {

  assert(
      RC == &Toy64::GPRRegClass &&
      "Toy64 can currently reload only GPRs");

  assert(
      SubReg == 0 &&
      "Toy64 does not have subregisters");

  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo &MFI = MF.getFrameInfo();

  MachineMemOperand *MMO =
      MF.getMachineMemOperand(
          MachinePointerInfo::getFixedStack(
              MF,
              FrameIndex),
          MachineMemOperand::MOLoad,
          MFI.getObjectSize(FrameIndex),
          MFI.getObjectAlign(FrameIndex));

  BuildMI(
      MBB,
      MI,
      DebugLoc(),
      get(Toy64::LD),
      DestReg)
      .addFrameIndex(FrameIndex)
      .addImm(0)
      .addMemOperand(MMO)
      .setMIFlag(Flags);
}

Register Toy64InstrInfo::isStoreToStackSlot(
    const MachineInstr &MI,
    int &FrameIndex) const {

  if (MI.getOpcode() != Toy64::SD)
    return Register();

  if (!MI.getOperand(1).isFI())
    return Register();

  if (!MI.getOperand(2).isImm() ||
      MI.getOperand(2).getImm() != 0)
    return Register();

  FrameIndex =
      MI.getOperand(1).getIndex();

  return MI.getOperand(0).getReg();
}

Register Toy64InstrInfo::isLoadFromStackSlot(
    const MachineInstr &MI,
    int &FrameIndex) const {

  if (MI.getOpcode() != Toy64::LD)
    return Register();

  if (!MI.getOperand(1).isFI())
    return Register();

  if (!MI.getOperand(2).isImm() ||
      MI.getOperand(2).getImm() != 0)
    return Register();

  FrameIndex =
      MI.getOperand(1).getIndex();

  return MI.getOperand(0).getReg();
}