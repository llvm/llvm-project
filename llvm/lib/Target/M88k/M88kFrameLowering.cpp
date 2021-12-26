//===-- M88kFrameLowering.cpp - Frame lowering for M88k -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "M88kFrameLowering.h"
//#include "M88kCallingConv.h"
//#include "M88kInstrBuilder.h"
#include "M88kInstrInfo.h"
//#include "M88kMachineFunctionInfo.h"
#include "M88kRegisterInfo.h"
#include "M88kSubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/IR/Function.h"
#include "llvm/Target/TargetMachine.h"

/*
 * The M88k stack layout:
 * +-------------------------+ High Address
 * |                         |
 * |                         |
 * | Argument Area           |
 * +-------------------------+ <- SP before call
 * |                         |    Pointer to last allocated word
 * | Temporary Space /       |    16-byte aligned
 * | Local Variable Space    |
 * | (optional)              |
 * +-------------------------+
 * |                         |
 * | Argument Area           |
 * | (at least 32 bytes,     |
 * |  equals 8 registers)    |
 * +-------------------------+ <- SP after call
 * |                         |
 * |                         |
 * +-------------------------+ <- Low Address
 *
 *
 */

using namespace llvm;

M88kFrameLowering::M88kFrameLowering()
    : TargetFrameLowering(TargetFrameLowering::StackGrowsDown, Align(16),
                          /*LocalAreaOffset=*/0, Align(8),
                          /*StackRealignable0*/ false),
      RegSpillOffsets(0) {}

bool M88kFrameLowering::hasFP(const MachineFunction &MF) const { return false; }

StackOffset
M88kFrameLowering::getFrameIndexReference(const MachineFunction &MF, int FI,
                                          Register &FrameReg) const {
llvm::dbgs() << "Enter " << __FUNCTION__ << " for " << MF.getFunction().getName() << "\n";
  const MachineFrameInfo &MFI = MF.getFrameInfo();

  FrameReg = hasFP(MF) ? M88k::R30 : M88k::R31;
llvm::dbgs() << " -> Index " << FI << "\n";
  return StackOffset::getFixed(MFI.getObjectOffset(FI) + MFI.getStackSize() -
                               getOffsetOfLocalArea() +
                               MFI.getOffsetAdjustment());
}

bool M88kFrameLowering::spillCalleeSavedRegisters(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    ArrayRef<CalleeSavedInfo> CSI, const TargetRegisterInfo *TRI) const {

  MachineFunction *MF = MBB.getParent();
  const TargetSubtargetInfo &STI = MF->getSubtarget();
  const TargetInstrInfo *TII = STI.getInstrInfo();
  const Register RAReg = STI.getRegisterInfo()->getRARegister();

  for (auto &CS : CSI) {
    // Add the callee-saved register as live-in.
    Register Reg = CS.getReg();
    bool IsRetAddrIsTaken =
        Reg == M88k::R1 && MF->getFrameInfo().isReturnAddressTaken();
    if (!IsRetAddrIsTaken)
      MBB.addLiveIn(Reg);

    // Save in the normal TargetInstrInfo way.
    bool IsKill = !IsRetAddrIsTaken;
    // TODO const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
    const TargetRegisterClass &RC = M88k::GPRRCRegClass;
    TII->storeRegToStackSlot(MBB, MBBI, Reg, IsKill, CS.getFrameIdx(), &RC,
                             TRI);
  }

  // TODO Check correct handling of R1.
  // R1 is marked as used in the RET pseudo instruction.
  // R1 is also part of the callee-saved register list.
  // In case R1 needs to be saved, it becomes part of CSI,
  // and is automatically marked as live-in.

  // Mark the return address register as live in.
  MBB.addLiveIn(M88k::R1);
  return true;
}

void M88kFrameLowering::determineCalleeSaves(MachineFunction &MF,
                                             BitVector &SavedRegs,
                                             RegScavenger *RS) const {
  TargetFrameLowering::determineCalleeSaves(MF, SavedRegs, RS);

  MachineFrameInfo &MFFrame = MF.getFrameInfo();

  // If the function calls other functions, record that the return
  // address register will be clobbered.
  if (MFFrame.hasCalls())
    SavedRegs.set(M88k::R1);
}

bool M88kFrameLowering::restoreCalleeSavedRegisters(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
    MutableArrayRef<CalleeSavedInfo> CSI, const TargetRegisterInfo *TRI) const {
  return false;
}

void M88kFrameLowering::emitPrologue(MachineFunction &MF,
                                     MachineBasicBlock &MBB) const {}

MachineBasicBlock::iterator M88kFrameLowering::eliminateCallFramePseudoInstr(
    MachineFunction & /*MF*/, MachineBasicBlock &MBB,
    MachineBasicBlock::iterator I) const {
  // TODO Implementation needed?
  // Discard ADJCALLSTACKDOWN, ADJCALLSTACKUP instructions.
  return MBB.erase(I);
}

void M88kFrameLowering::emitEpilogue(MachineFunction &MF,
                                     MachineBasicBlock &MBB) const {}
