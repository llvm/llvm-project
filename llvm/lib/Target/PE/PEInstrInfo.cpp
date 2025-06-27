/* --- PEInstrInfo.cpp --- */

/* ------------------------------------------
author: 高宇翔
date: 4/1/2025
------------------------------------------ */

#include "PEInstrInfo.h"
#include "MCTargetDesc/PEMCTargetDesc.h"

using namespace llvm;

#define GET_INSTRINFO_CTOR_DTOR
#include "PEGenInstrInfo.inc"

PEInstrInfo::PEInstrInfo() : PEGenInstrInfo() {
  // Constructor
}

PEInstrInfo::~PEInstrInfo() {
  // Destructor
}
void PEInstrInfo::storeRegToStackSlot(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MI, Register SrcReg,
    bool isKill, int FrameIndex, const TargetRegisterClass *RC,
    const TargetRegisterInfo *TRI, Register VReg,
    MachineInstr::MIFlag Flags) const {
  // 以你的目标指令为例，假设 ST rd, sp, offset
  BuildMI(MBB, MI, MI != MBB.end() ? MI->getDebugLoc() : DebugLoc(),
          get(PE::STOREWFI))
      .addFrameIndex(FrameIndex)
      .addReg(SrcReg, getKillRegState(isKill))
      .setMIFlag(Flags);
}

void PEInstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator MI,
                                       Register DestReg, int FrameIndex,
                                       const TargetRegisterClass *RC,
                                       const TargetRegisterInfo *TRI,
                                       Register VReg,
                                       MachineInstr::MIFlag Flags) const {
  // 以你的目标指令为例，假设 LD rd, sp, offset
  BuildMI(MBB, MI, MI != MBB.end() ? MI->getDebugLoc() : DebugLoc(),
          get(PE::LOADWFI), DestReg)
      .addFrameIndex(FrameIndex)
      .setMIFlag(Flags);
}
void PEInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MI,
                              const DebugLoc &DL, Register DestReg,
                              Register SrcReg, bool KillSrc,
                              bool RenamableDest,
                              bool RenamableSrc) const {
  BuildMI(MBB, MI, DL, get(PE::ADD), DestReg)
      .addReg(SrcReg, getKillRegState(KillSrc))
      .addReg(PE::RS0)
      .setMIFlag(MachineInstr::NoFlags);
}